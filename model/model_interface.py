import inspect
import torch
import importlib
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import pytorch_lightning as pl

from transformers import LlamaForCausalLM, LlamaTokenizer
import random
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
import numpy as np
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
import clip
from PIL import Image


class MInterface(pl.LightningModule):
    def __init__(self,
                 image_dir=r"/root/autodl-tmp/LLaRA/data/ref/steam/steam_posters",
                 **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_llm(self.hparams.llm_path)
        self.load_rec_model(self.hparams.rec_model_path)
        self.load_projector()
        self.load_clip()
        # 初始化图像路径映射
        self.image_dir = image_dir
        self.image_path_map = self._build_image_path_map() if image_dir else None

    def _build_image_path_map(self):
        """构建seqID到图片路径的映射"""
        image_map = {}
        for root, _, files in os.walk(self.image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    seq_id = op.splitext(file)[0]  # 假设文件名就是seqID
                    image_map[seq_id] = op.join(root, file)
        return image_map

    def load_clip(self):
        """加载CLIP模型和预处理器"""
        if not hasattr(self, 'clip_model'):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            # 冻结CLIP参数
            for param in self.clip_model.parameters():
                param.requires_grad = False

        print("加载CLIP模型")
        # 新增图像投影层（将CLIP嵌入对齐到LLM空间）
        self.image_projector = nn.Sequential(
            nn.Linear(512, 4096),
            nn.GELU(),
            nn.Linear(4096, 4096)
        )

    def get_image_embedding(self, seq_ids):
        """
        根据seqID获取图像嵌入
        seq_ids: Tensor [batch_size, seq_len] 或 [batch_size]
        """
        if not self.image_path_map:
            return None

        # 转换输入为字符串列表
        if isinstance(seq_ids, torch.Tensor):
            if seq_ids.dim() > 1:  # 如果是二维张量 [batch_size, seq_len]
                seq_ids = seq_ids.flatten()  # 展平为一维
            seq_ids = [str(int(id)) for id in seq_ids.cpu().numpy()]

        images = []
        valid_indices = []
        for i, seq_id in enumerate(seq_ids):
            if seq_id in self.image_path_map:
                try:
                    image = Image.open(self.image_path_map[seq_id])
                    image = self.clip_preprocess(image).unsqueeze(0)
                    images.append(image)
                    valid_indices.append(i)
                except Exception as e:
                    print(f"Error loading image for seq_id {seq_id}: {e}")

        if not images:
            return None

        # 批量处理图像
        images = torch.cat(images).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 投影到LLM空间
        projected_embeds = self.image_projector(image_features)

        # 创建全零张量并填充有效嵌入
        batch_size = len(seq_ids)
        full_embeds = torch.zeros(batch_size, self.llama_model.config.hidden_size,
                                  device=self.device)
        projected_embeds = projected_embeds.float()
        full_embeds[valid_indices] = projected_embeds

        return full_embeds

    def forward(self, batch):
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        )  # [batch_size, max_len]
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:, 1:], -100)
        input_embeds = self.wrap_emb(batch)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        return outputs

    def generate(self, batch, temperature=0.8, do_sample=False, num_beams=1, max_gen_length=64, min_gen_length=1,
                 repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=1):
        input_embeds = self.wrap_emb(batch)
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences
        )
        output_text = self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)
        outputs = [text.strip() for text in output_text]
        return outputs

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        if batch["flag"]:
            for name, param in self.projector.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.projector.named_parameters():
                param.requires_grad = True
        out = self(batch)
        loss = self.configure_loss(out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_content = {
            "generate": [],
            "real": [],
            "cans": [],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output = []
        for i, generate in enumerate(generate_output):
            real = batch['correct_answer'][i]
            cans = batch['cans_name'][i]
            generate = generate.strip().split("\n")[0]
            output.append((generate, real, cans))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real, cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)

    def on_validation_epoch_end(self):
        df = DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'valid.csv'))
        prediction_valid_ratio, hr = self.calculate_hr1(self.val_content)
        metric = hr * prediction_valid_ratio
        self.log('val_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_content = {
            "generate": [],
            "real": [],
            "cans": [],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # 获取 inputs_embeds
        input_embeds = self.wrap_emb(batch)

        # 转换为 NumPy 数组（确保是 float32）
        if isinstance(input_embeds, torch.Tensor):
            input_embeds_np = input_embeds.to(torch.float32).cpu().numpy()
        else:
            input_embeds_np = input_embeds.astype(np.float32)  # 如果已经是 NumPy 数组，确保类型正确

        # 保存到文件
        output_dir = self.hparams.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        embeds_file = os.path.join(output_dir, f"inputs_embeds_batch_{batch_idx}.npy")
        np.save(embeds_file, input_embeds_np)  # 直接保存 NumPy 数组
        print(f"Saved inputs_embeds for batch {batch_idx} to {embeds_file}")
        generate_output = self.generate(batch)
        output = []
        for i, generate in enumerate(generate_output):
            real = batch['correct_answer'][i]
            cans = batch['cans_name'][i]
            generate = generate.strip().split("\n")[0]
            output.append((generate, real, cans))
        return output

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real, cans in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)

    def on_test_epoch_end(self):
        df = DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, 'test.csv'))
        prediction_valid_ratio, hr = self.calculate_hr1(self.test_content)
        metric = hr * prediction_valid_ratio
        self.log('test_prediction_valid', prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_hr', hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam([
            {'params': self.projector.parameters(), 'lr': self.hparams.lr, 'weight_decay': weight_decay},
            {'params': self.llama_model.parameters(), 'lr': self.hparams.lr}
        ])

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                               max_step=max_step,
                                                               min_lr=self.hparams.lr_decay_min_lr,
                                                               init_lr=self.hparams.lr,
                                                               warmup_steps=warmup_steps,
                                                               warmup_start_lr=self.hparams.lr_warmup_start_lr)
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer

    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == 'lm':
            return out.loss
        else:
            raise ValueError("Invalid Loss Type!")

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == 'part':
            checkpoint.pop('optimizer_states')
            to_be_removed = []
            for key, value in checkpoint['state_dict'].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        elif self.hparams.save == 'all':
            pass

    def load_llm(self, llm_path):
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]', '[HistoryEmb]', '[CansEmb]',
                                                                               '[ItemEmb]', '[ImageHere]',
                                                                               '[CansImage]']})
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.hparams.llm_tuning == 'lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj',
                                                             'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == 'freeze':
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        elif self.hparams.llm_tuning == 'freeze_lora':
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llm_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                             inference_mode=False,
                                             r=self.hparams.lora_r,
                                             lora_alpha=self.hparams.lora_alpha,
                                             lora_dropout=self.hparams.lora_dropout,
                                             target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj',
                                                             'up_proj', 'down_proj'])
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()
        else:
            raise NotImplementedError()

        print('Loading LLAMA Done')

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        self.projector = self.instancialize(Model, rec_size=self.hparams.rec_size,
                                            llm_size=self.llama_model.config.hidden_size)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def load_rec_model(self, rec_model_path):
        print('Loading Rec Model')
        self.rec_model = torch.load(rec_model_path, map_location="cpu", weights_only=False)
        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False
        print('Loding Rec model Done')

    def encode_items(self, seq):
        if self.hparams.rec_embed == "SASRec":
            item_rec_embs = self.rec_model.cacu_x(seq)
        elif self.hparams.rec_embed in ['Caser', 'GRU']:
            item_rec_embs = self.rec_model.item_embeddings(seq)
        item_txt_embs = self.projector(item_rec_embs)
        return item_txt_embs

    def embed_tokens(self, token_ids):
        embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    def wrap_emb(self, batch):

        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)

        his_token_id = self.llama_tokenizer("[HistoryEmb]", return_tensors="pt",
                                            add_special_tokens=False).input_ids.item()
        cans_token_id = self.llama_tokenizer("[CansEmb]", return_tensors="pt",
                                             add_special_tokens=False).input_ids.item()
        item_token_id = self.llama_tokenizer("[ItemEmb]", return_tensors="pt",
                                             add_special_tokens=False).input_ids.item()
        image_token_id = self.llama_tokenizer("[ImageHere]", return_tensors="pt",
                                              add_special_tokens=False).input_ids.item()  # 新增图像token
        cans_image_token_id = self.llama_tokenizer("[CansImage]", return_tensors="pt",
                                                   add_special_tokens=False).input_ids.item()  # 新增候选图像token
        his_item_embeds = self.encode_items(batch["seq"])
        cans_item_embeds = self.encode_items(batch["cans"])
        item_embeds = self.encode_items(batch["item_id"])
        if self.image_dir:
            # 历史序列图像嵌入
            his_image_embeds = self.get_image_embedding(batch["seq"])
            # 候选列表图像嵌入
            cans_image_embeds = self.get_image_embedding(batch["cans"])
            # 目标项图像嵌入
            target_image_embeds = self.get_image_embedding(batch["item_id"])

        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i] == his_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == his_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, his_item_embeds[i, :batch["len_seq"][i].item()]):
                    input_embeds[i, idx] = item_emb
            if (batch["tokens"].input_ids[i] == cans_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, cans_item_embeds[i, :batch["len_cans"][i].item()]):
                    input_embeds[i, idx] = item_emb
            if (batch["tokens"].input_ids[i] == item_token_id).nonzero().shape[0] > 0:
                idx = (batch["tokens"].input_ids[i] == item_token_id).nonzero().item()
                input_embeds[i, idx] = item_embeds[i]
            if self.image_dir and (batch["tokens"].input_ids[i] == image_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == image_token_id).nonzero().view(-1)
                if his_image_embeds is not None:
                    for idx, img_emb in zip(idx_tensor, his_image_embeds[i, :batch["len_seq"][i].item()]):
                        input_embeds[i, idx] = img_emb
            # 新增：候选列表图像嵌入替换
            if self.image_dir and (batch["tokens"].input_ids[i] == cans_image_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == cans_image_token_id).nonzero().view(-1)
                if cans_image_embeds is not None:
                    for idx, img_emb in zip(idx_tensor, cans_image_embeds[i, :batch["len_cans"][i].item()]):
                        input_embeds[i, idx] = img_emb

        return input_embeds

    def calculate_hr1(self, eval_content):
        correct_num = 0
        valid_num = 0
        total_num = 0
        for i, generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            cans = eval_content["cans"][i]
            total_num += 1
            generate = generate.strip().lower().strip()
            real = real.strip().lower().strip()
            cans = [item.strip().lower().strip() for item in cans]
            gen_cans_list = []
            for cans_item in cans:
                if cans_item in generate:
                    gen_cans_list.append(cans_item)
            if len(gen_cans_list) == 1:
                valid_num += 1
                if real == gen_cans_list[0]:
                    correct_num += 1
        valid_ratio = valid_num / total_num
        if valid_num > 0:
            hr1 = correct_num / valid_num
        else:
            hr1 = 0
        return valid_ratio, hr1

# def main():
#     # 设置图像目录路径
#     image_dir = "/root/autodl-tmp/LLaRA/data/ref/steam/steam_posters"
#     # 设置 LLM 模型路径
#     llm_path = "/root/autodl-tmp/LLaRA/model/llama2"  # 替换为实际路径
#     llm_tuning = "lora"
#     # 创建 MInterface 实例
#     model = MInterface(image_dir=image_dir, llm_path=llm_path,llm_tuning=llm_tuning)
#     # 创建 MInterface 实例
#     model = MInterface(image_dir=image_dir)

#     # 模拟输入的 seq_ids 列表
#     seq_ids = ["123", "5", "71"]  # 假设这些是 seqID，对应图像文件名

#     # 获取图像嵌入
#     image_embeddings = model.get_image_embedding(seq_ids)

#     if image_embeddings is not None:
#         print("图像嵌入的形状:", image_embeddings.shape)
#         print("图像嵌入的前几个值:", image_embeddings[:2])
#     else:
#         print("未找到任何图像嵌入，可能是因为 seq_ids 中的某些 ID 未找到对应的图像文件。")

# if __name__ == "__main__":
#     main()