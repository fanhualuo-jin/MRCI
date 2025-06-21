# 我的M²RCI是基于LLaRA模型进行修改的

首先是LLaRA介绍

## LLaRA

|                | movielens  || steam    || lastfm   ||
|----------------|------------|------|----------|------|----------|------|
|                | ValidRatio | HitRatio@1 | ValidRatio | HitRatio@1 | ValidRatio | HitRatio@1 |
| LLaRA(GRU4Rec) | 0.9684     | 0.4000 | 0.9840 | 0.4916 | 0.9672 | 0.4918 |
| LLaRA(Caser)   | 0.9684     | 0.4211 | 0.9519 | 0.4621 | 0.9754 | 0.4836 |
| LLaRA(SASRec)  | 0.9789     | 0.4526 | 0.9958 | 0.5051 | 0.9754 | 0.5246 |
## 接下来是我的项目的训练过程

##### Preparation

1. Prepare the environment: 

   ```sh
   git clone https://github.com/fanhualuo-jin/MRCI.git
   cd M²RCI
   pip install -r requirements.txt
   ```

2. 本实验使用的是Llama2-7b-hf模型，由于在Huggingface上需要向官方申请，这里我们在魔搭社区下载模型 (https://www.modelscope.cn/models/shakechen/Llama-2-7b-hf).

3. Download the [data](https://www.modelscope.cn/datasets/fanhualuojin/MRCI/files) and checkpoints.在LLaRA的基础上我更新了数据集，加入了图片数据，实现了多模态。

4. Prepare the data and checkpoints:

   Put the data to the dir path `data/ref/` 。

##### Train 

Train  with a single A100 GPU on Steam dataset:

```sh
sh train_steam.sh
```

##### Evaluate 

Test with a single A100 GPU on Steam dataset:

```sh
sh test_steam.sh
```

由于movielens和lastfm海报暂未爬取，所以现在只实现了steam数据集的M²RCI。

整个实验部署在autodl的A100服务器上，经测试，用4090等新型显卡也可以进行模型的训练。由于算力的限制，暂时没有发布cheakpoints