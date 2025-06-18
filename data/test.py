import os
import pandas as pd
import torch
from transformers import LlamaTokenizer
import random

class TrainCollater:
    def __init__(self,
                 prompt_list=None,
                 llm_tokenizer=None,
                 train=False,
                 terminator="\n",
                 max_step=1):
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.train = train
        self.terminator = terminator
        self.max_step = max_step
        self.cur_step = 1

    def __call__(self, batch):
        if isinstance(self.prompt_list, list):
            instruction = random.choice(self.prompt_list)
            inputs_text = [instruction] * len(batch)
        else:
            raise ValueError("Prompt list must be provided as a list of strings.")

        for i, sample in enumerate(batch):
            input_text = inputs_text[i]
            if '[HistoryHere]' in input_text:
                insert_prompt = ", ".join([seq_title + ' [HistoryEmb]' for seq_title in sample['seq_name']])
                input_text = input_text.replace('[HistoryHere]', insert_prompt)
            if '[CansHere]' in input_text:
                insert_prompt = ", ".join([can_title + ' [CansEmb]' for can_title in sample['cans_name']])
                input_text = input_text.replace('[CansHere]', insert_prompt)
            inputs_text[i] = input_text

        targets_text = [sample['correct_answer'] for sample in batch]

        batch_tokens = self.llm_tokenizer(
            inputs_text,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True
        )

        new_batch = {
            "tokens": batch_tokens,
            "seq": torch.tensor([sample['seq'] for sample in batch], dtype=torch.long),
            "cans": torch.tensor([sample['cans'] for sample in batch], dtype=torch.long),
            "len_seq": torch.tensor([sample['len_seq'] for sample in batch], dtype=torch.long),
            "len_cans": torch.tensor([sample['len_cans'] for sample in batch], dtype=torch.long),
            "item_id": torch.tensor([sample['item_id'] for sample in batch], dtype=torch.long),
            "correct_answer": targets_text,
            "cans_name": [sample['cans_name'] for sample in batch]
        }

        return new_batch

# 加载 Prompt 文件
def load_prompt(prompt_path):
    if os.path.isfile(prompt_path):
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        prompt_list = [p.strip() for p in raw_prompts]
        print('Load {} prompts'.format(len(prompt_list)))
        print('Prompt Example \n{}'.format(random.choice(prompt_list)))
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_list

# 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)
print("当前脚本的绝对路径:", current_file_path)
# 获取当前脚本所在的目录
current_dir = os.path.dirname(current_file_path)
print("当前脚本所在的目录:", current_dir)

# 构建 CSV 文件的路径
csv_path = r'data/ref/lastfm/lastfm_sample_test.csv'
print("CSV 文件路径:", csv_path)

# 构建 Prompt 文件的路径
prompt_path = r'prompt/artist.txt'
print("Prompt 文件路径:", prompt_path)

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 将 DataFrame 转换为列表形式的样本
samples = df.to_dict(orient='records')

# 加载 Prompt 文件
prompt_list = load_prompt(prompt_path)

# 初始化 LLM Tokenizer
# 指定本地模型路径
local_model_path = "/root/autodl-tmp/LLaRA/model/llama2"

# 从本地加载分词器
llm_tokenizer = LlamaTokenizer.from_pretrained(local_model_path,local_files_only=True)

# 初始化 TrainCollater
collater = TrainCollater(
    prompt_list=prompt_list,
    llm_tokenizer=llm_tokenizer,
    train=False,
    terminator="\n",
    max_step=1
)

# 处理数据
processed_batch = collater(samples)

# 打印处理后的数据
print("处理后的输入文本：")
print(processed_batch['tokens'])
print("\n其他信息：")
print(processed_batch)