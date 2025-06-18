import torch
import os
from PIL import Image
from model_interface import MInterface
import argparse


def test_image_embedding():
    # 初始化参数
    args = argparse.Namespace(
        llm_path="model/llama2",
        rec_model_path="checkpoints/steam/steam.ckpt",
        output_dir="test_output",
        rec_embed="SASRec",  # 根据你的实际模型调整
        rec_size=64,  # 根据你的实际模型调整
        lr=1e-4,
        batch_size=4,
        model_name="mlp_projector",
        # 其他必要参数...
    )

    # 初始化模型
    model = MInterface(
        image_dir=r"/root/autodl-tmp/LLaRA/data/ref/steam/steam_posters",
        **vars(args)
    )
    model = model.cuda() if torch.cuda.is_available() else model.cpu()

    # 测试1: 检查图片映射是否正确构建
    print("\n=== 测试1: 检查图片映射 ===")
    if model.image_path_map:
        print(f"找到 {len(model.image_path_map)} 张图片")
        print("示例映射 (前5项):")
        for i, (k, v) in enumerate(list(model.image_path_map.items())[:5]):
            print(f"{k}: {v}")
            # 验证图片是否能打开
            try:
                img = Image.open(v)
                print(f"  图片验证成功，尺寸: {img.size}")
                img.close()
            except Exception as e:
                print(f"  图片验证失败: {e}")
    else:
        print("警告: 没有找到任何图片！请检查路径和文件")
        return

    # 测试2: 测试get_image_embedding函数
    print("\n=== 测试2: 测试get_image_embedding ===")

    # 创建测试数据 (使用实际找到的seq_ids)
    test_seq_ids = list(model.image_path_map.keys())[:4]  # 测试前4个

    print(f"测试seq_ids: {test_seq_ids}")
    embeddings = model.get_image_embedding(test_seq_ids)

    if embeddings is not None:
        print(f"获取到的嵌入形状: {embeddings.shape}")
        print(f"示例嵌入 (第一个):\n{embeddings[0][:10]}...")  # 只打印前10维

        # 检查是否有无效的嵌入
        zero_embeds = torch.all(embeddings == 0, dim=1)
        print(f"无效嵌入数量: {torch.sum(zero_embeds).item()}")
    else:
        print("错误: 未能获取图片嵌入")

    # 测试3: 测试不存在的seq_id
    print("\n=== 测试3: 测试不存在的seq_id ===")
    fake_seq_ids = ["999999", "000000"] + test_seq_ids[:2]
    print(f"测试seq_ids (包含不存在的ID): {fake_seq_ids}")
    embeddings = model.get_image_embedding(fake_seq_ids)

    if embeddings is not None:
        print(f"获取到的嵌入形状: {embeddings.shape}")
        zero_embeds = torch.all(embeddings == 0, dim=1)
        print(f"无效嵌入数量 (应为2): {torch.sum(zero_embeds).item()}")
    else:
        print("错误: 未能获取图片嵌入")


if __name__ == "__main__":
    # 创建测试输出目录
    os.makedirs("test_output", exist_ok=True)

    print("=== 开始测试图片嵌入功能 ===")
    test_image_embedding()
    print("=== 测试完成 ===")