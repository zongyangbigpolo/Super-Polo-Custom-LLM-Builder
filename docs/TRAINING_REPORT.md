# 训练报告：LoRA 风格微调实验

> 返回 [README](../README.md)｜架构见 [ARCHITECTURE.md](ARCHITECTURE.md)｜原理见 [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md)

本报告记录一次在本地 MacBook 上真实完成的 LoRA 微调。所有数字由 [`finetune/evaluate.py`](../finetune/evaluate.py) 自动生成，原始文件都在 [`docs/results/`](results/) 下。

## 1. 实验目的与数据说明

**目的**：验证"聊天数据 → 数据处理 → LoRA 训练 → 评测"这条流水线能真正把模型从"AI 客服腔"调成"真人聊天腔"。

**数据来源（如实说明）**：本次实验**没有使用个人微信聊天记录**，用的是公开中文闲聊语料 **LCCC-base**（清华 CoAI 组，MIT 许可，来自微博等社交平台的真实多轮闲聊），通过 [`prepare_data.py lccc`](../finetune/prepare_data.py) 获取和处理：

| | 数量 | 来源 |
| --- | --- | --- |
| 训练集 | 2,000 段对话 | LCCC-base `train` 切分 |
| 留出评测集 | 200 段对话 | LCCC-base `validation` 切分（与训练集不重叠） |
| 平均上下文轮数 | 1.9 轮（含最后一句回复共 2.9 轮） | |
| 目标回复平均长度 | 10.5 字 | |

数据样例见 [`results/train_sample.jsonl`](results/train_sample.jsonl)，格式与微信数据处理后完全一致（`system / user / assistant`，最后一轮是"我"的回复）。

**换成自己的微信数据**只需替换第一步：

```bash
python finetune/prepare_data.py wechat --input 微信导出.csv --out finetune/data/train.jsonl
```

## 2. 实验配置

| 项目 | 值 |
| --- | --- |
| 基座模型 | `Qwen/Qwen2.5-0.5B-Instruct`（4.94 亿参数） |
| 方法 | LoRA，`r=8`，`alpha=16`，`dropout=0.05` |
| 注入层 | `q/k/v/o_proj` + `gate/up/down_proj`（注意力 + MLP，共 24 层） |
| **可训练参数** | **4,399,104 / 498,431,872 = 0.883%** |
| 监督方式 | 只对最后一句回复计算损失（上下文 token 被 mask） |
| 优化器 | AdamW，lr `2e-4`，5% warmup + cosine 衰减，weight decay 0.01，梯度裁剪 1.0 |
| Batch | 4 × 梯度累积 2 = 等效 8 |
| 训练量 | 2 epoch = 500 步 |
| 最大长度 | 256 token |
| 硬件 | Apple M4，16GB 统一内存，MPS 后端，fp32 |
| **训练耗时** | **1,024.6 秒（约 17 分钟）** |
| 适配器大小 | 17.6 MB（`adapter_model.safetensors`） |
| 软件 | torch 2.15.0，transformers 5.23.0，peft 0.21.3，macOS 27.0 |

完整超参与逐步日志：[`results/train_log.json`](results/train_log.json)、[`results/adapter_config.json`](results/adapter_config.json)。
适配器权重 SHA-256：`36ca5fefd268afca6f30eba889aee67e3d11bb74baa7420e2e48de1cd8c17477`

复现命令：

```bash
pip install -r finetune/requirements.txt
python finetune/prepare_data.py lccc --split train --num 2000 --out finetune/data/train.jsonl
python finetune/prepare_data.py lccc --split validation --num 200 --out finetune/data/eval.jsonl
python finetune/train_lora.py --train finetune/data/train.jsonl --eval finetune/data/eval.jsonl --out outputs/polo-style-lora
python finetune/evaluate.py --adapter outputs/polo-style-lora --eval finetune/data/eval.jsonl --report-dir docs/results
```

## 3. 训练曲线

![loss curve](results/loss_curve.png)

| step | 0 | 50 | 100 | 150 | **200** | 250 | 300 | 350 | 400 | 450 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 留出集 loss | 6.401 | 4.986 | 4.961 | 4.936 | **4.882** | 4.906 | 4.954 | 4.980 | 4.959 | 4.954 | 4.954 |
| 困惑度 | 602.1 | 146.3 | 142.7 | 139.2 | **131.9** | 135.0 | 141.7 | 145.5 | 142.4 | 141.7 | 141.7 |

**分析**：

- 前 50 步下降最快：模型很快学会"闲聊该怎么回"这种整体风格。
- 第 200 步（第 1 个 epoch 共 250 步）留出集 loss 最低；进入第 2 个 epoch 后训练 loss 继续降到约 4.1，但留出集 loss 小幅回升，说明 2,000 条数据下开始**轻微过拟合**。数据量小时建议训练 1 个 epoch，或开启早停保留最优步。

## 4. 微调前后对比

在 200 条留出对话上计算 loss；在其中前 60 条上贪心解码生成回复：

| 指标 | 基座模型 | LoRA 微调后 | 真实人类回复 | 说明 |
| --- | --- | --- | --- | --- |
| 留出集 loss ↓ | 6.401 | **4.954** | — | 真实回复的平均负对数似然 |
| 困惑度 ↓ | 602.1 | **141.7（−76.5%）** | — | 模型对"真人会怎么回"的意外程度 |
| 平均回复长度 | 25.6 字 | **7.5 字** | 9.3 字 | 越接近真人越好 |
| 字级 F1 ↑ | 0.080 | **0.100** | — | 与真实回复的字重合度 |
| distinct-1 | 0.288 | 0.318 | 0.460 | 单字多样性 |
| distinct-2 | **0.730** | 0.625 | 0.894 | 双字多样性 |

原始数据：[`results/metrics.json`](results/metrics.json)

**典型样例**（摘自 [`results/samples.md`](results/samples.md)，该文件按留出集顺序列出前 15 条，未经挑选）：

| 对方说 | 真实回复 | 基座模型 | LoRA 微调后 |
| --- | --- | --- | --- |
| 问哈嘛 | 哦哦 | 你好！有什么我可以帮你解答的吗？ | 我也是 |
| 心疼夫人 | 啥？ | 哎呀，听到你这么说，我也很为你担心。有什么我可以帮忙的吗？ | 你这是在安慰我啊 |
| 除了远之外，挺美的，有机会去玩啊~ | 嗯嗯嗯哪～ | 那太好了！咱们可以一起计划一下，找个时间一起去玩儿吧！ | 嗯嗯 |
| 生日快乐爱你 | 我也是 | 生日快乐！爱你永远。 | 谢谢啦 |
| 刚才请教你剩下西甲两场怎么看？交叉盘吗 | 哪里来的这么多交叉… | 看下双方的胜平负走势……交叉盘就是看主队和客队的赔率高低…… | 我还没看啊 |

## 5. 结论与局限

**结论**

1. 只训练 **0.88%** 的参数、17 分钟、一台 16GB 的 Mac，就能让模型对真人回复的困惑度下降 **76%**。
2. 回复风格明显从"AI 客服腔"（长、礼貌、爱提供帮助）变成"真人聊天腔"（短、口语、"嗯嗯""谢谢啦"），长度从 25.6 字降到 7.5 字，接近真人的 9.3 字。

**局限（如实说明）**

1. 数据是公开闲聊语料，学到的是"普通网友的聊天风格"，不是某个具体的人；要模仿本人，需要换成本人的微信记录。
2. distinct-2 下降、部分回复偏模板化（例如"我也是这么认为的"）：小模型 + 小数据 + 贪心解码导致的"安全回复"倾向。可以通过更多数据、采样解码（temperature 0.7–0.9）、更大的基座（1.5B–7B）缓解。
3. 0.5B 模型知识有限，会出现事实错误（例如"中国人是汉族人"），实际部署建议用更大的基座。
4. 自动指标只能部分反映"像不像"，最好再加人工盲测。

## 6. 部署这个适配器

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 \
  --enable-lora --lora-modules polo-style=outputs/polo-style-lora
```

`config.json` 中设置 `"model": "polo-style"`、`"open_ai_api_base": "http://localhost:8000/v1"` 即可让微信机器人使用微调后的模型。再配合 [`StyleLearner`](../plugins/style_learner/README.md) 插件，机器人会按联系人记录对话、接受纠正，并把这些记录用于下一轮训练。
