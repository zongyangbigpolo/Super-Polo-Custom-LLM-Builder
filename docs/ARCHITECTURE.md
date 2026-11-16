# 架构说明 (Architecture)

> 返回 [README](../README.md)｜技术原理见 [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md)

本项目由三层组成：

| 层 | 作用 | 对应代码 |
| --- | --- | --- |
| ① 训练层 | 用 **LoRA / QLoRA**（低秩适配，即"分秩"微调）把开源大模型调成"像我一样说话"；支持 **FedSA-LoRA** 联邦式低秩微调，数据不出本地 | [`finetune/`](../finetune)、[`docs/FINETUNE_GUIDE.md`](FINETUNE_GUIDE.md)、[`fedSA-LoRA/`](../fedSA-LoRA) |
| ② 推理服务层 | 用 **vLLM / Ollama** 在本地把模型（基座 + LoRA 适配器）发布成 OpenAI 兼容接口 | 外部进程（见 `screencapture/`） |
| ③ 应用接入层 | 多渠道聊天机器人：微信、企业微信、公众号、飞书、钉钉、网页、终端，带插件系统；`StyleLearner` 插件按联系人记录对话并学习风格 | `app.py`、`channel/`、`bridge/`、`bot/`、`plugins/`、`voice/` |

## 1. 总体架构

```mermaid
flowchart LR
    subgraph DATA["数据准备"]
        D1["个人聊天记录<br/>(微信导出等)"]
        D2["公开数据集<br/>(HuggingFace)"]
        D3["清洗 / 脱敏<br/>转成 messages JSONL"]
        D1 --> D3
        D2 --> D3
    end

    subgraph TRAIN["① 训练层：低秩微调"]
        B0["HuggingFace 开源基座<br/>DeepSeek / Qwen / Llama"]
        T1["单机 QLoRA<br/>Unsloth + PEFT + TRL"]
        T2["联邦 FedSA-LoRA<br/>fedSA-LoRA/ (FederatedScope)"]
        A1[("LoRA 适配器<br/>polo-super-lora")]
        B0 --> T1
        B0 --> T2
        D3 --> T1
        D3 --> T2
        T1 --> A1
        T2 --> A1
    end

    subgraph SERVE["② 推理服务层（本地）"]
        S1["vLLM / Ollama<br/>OpenAI 兼容 API :8000/v1"]
    end
    B0 --> S1
    A1 --> S1

    subgraph APP["③ 应用接入层 (app.py)"]
        C["channel/<br/>微信·企微·公众号·飞书·钉钉·Web·终端"]
        P["plugins/<br/>角色·关键词·敏感词·工具·Agent…"]
        BR["bridge/<br/>路由到具体模型"]
        BOT["bot/<br/>chatGPT(OpenAI兼容)·Claude·Gemini·通义…"]
        V["voice/ 语音识别·合成"]
        C <--> P
        P <--> BR
        BR --> BOT
        BR --> V
    end

    U(("用户")) <--> C
    BOT -->|"open_ai_api_base=<br/>http://localhost:8000/v1"| S1
```

## 2. 一条消息的处理流程

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户 (微信/网页…)
    participant CH as Channel<br/>channel/*
    participant CC as ChatChannel<br/>channel/chat_channel.py
    participant PM as PluginManager<br/>plugins/
    participant BR as Bridge<br/>bridge/bridge.py
    participant BOT as ChatGPTBot<br/>bot/chatgpt
    participant LLM as 本地 vLLM / Ollama<br/>(基座 + LoRA)

    U->>CH: 发送消息
    CH->>CC: 封装为 ChatMessage
    CC->>CC: 前缀/群白名单/语音转文字 → Context
    CC->>PM: ON_HANDLE_CONTEXT 事件
    alt 插件已直接处理（如关键词、敏感词）
        PM-->>CC: Reply
    else 交给大模型
        CC->>BR: fetch_reply_content(query, context)
        BR->>BOT: 根据 bot_type / model 选择 bot
        BOT->>BOT: session_manager 拼接历史对话 + character_desc
        BOT->>LLM: POST /v1/chat/completions
        LLM-->>BOT: 回复文本
        BOT-->>CC: Reply
    end
    CC->>PM: ON_DECORATE_REPLY / ON_SEND_REPLY
    CC->>CH: send(reply)
    CH-->>U: 文本 / 语音 / 图片
```

## 3. 训练流程（单机 QLoRA）

```mermaid
flowchart TD
    A["选择基座<br/>unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit"] --> B["4bit 量化加载<br/>FastLanguageModel.from_pretrained"]
    B --> C["注入 LoRA (r=64, alpha=32)<br/>q/k/v/o/gate/up/down_proj"]
    C --> D["SFTTrainer 训练<br/>只更新 LoRA 参数"]
    D --> E{"保存方式"}
    E -->|"小文件，可热插拔"| F["polo-super-lora/<br/>(适配器)"]
    E -->|"推理更快"| G["polo-super-merged/<br/>(合并后完整模型)"]
    F --> H["vLLM --enable-lora<br/>或 Ollama ADAPTER"]
    G --> H
```

## 3.5 边聊边学：StyleLearner 闭环

```mermaid
flowchart LR
    U(("联系人 A / B / C")) -->|消息| BOT["微信机器人"]
    BOT -->|回复| U
    BOT -->|"ON_SEND_REPLY<br/>按联系人记录"| LOG[("plugins/style_learner/data/<br/>&lt;联系人&gt;.jsonl")]
    OWNER(("我")) -->|"$纠正 我会这么回"| LOG
    LOG -->|"即时：最近/被纠正的对话<br/>注入该联系人的 system prompt"| BOT
    LOG -->|"定期：prepare_data.py stylelog"| DS["训练集 JSONL"]
    DS -->|"train_lora.py"| LORA[("新 LoRA 适配器")]
    LORA -->|"vLLM 热加载"| BOT
```

- **即时学习（分钟级）**：每记录 `refresh_every` 轮对话或每次 `$纠正` 后，把该联系人最近的（优先被纠正的）对话作为风格示例写进系统提示，所以对老师、同学、家人会自动用不同的称呼和语气。
- **深度学习（定期）**：攒够数据后执行 `prepare_data.py stylelog`，再用 `train_lora.py` 重新训练，把风格写进 LoRA 权重。system prompt 里带有联系人名，模型能学到"对不同人怎么说"。
- **隐私**：记录只保存在本地，文件名是会话 ID 的哈希，`data/` 已加入 `.gitignore`。

## 4. FedSA-LoRA 联邦低秩微调

多个客户端（例如多台设备、多个人的聊天数据）各自在本地训练，**只上传 LoRA 的 A 矩阵**，服务器对 A 做平均后下发；**B 矩阵始终留在本地**作为个性化部分。

```mermaid
flowchart TB
    subgraph Server["服务器 (FedAvg 聚合)"]
        AGG["A_global = Σ (n_k / n) · A_k"]
    end
    subgraph C1["客户端 1"]
        W1["冻结基座 W₀"]
        A1["A₁ (共享)"]
        B1["B₁ (本地私有)"]
    end
    subgraph C2["客户端 2"]
        W2["冻结基座 W₀"]
        A2["A₂ (共享)"]
        B2["B₂ (本地私有)"]
    end
    subgraph C3["客户端 3"]
        W3["冻结基座 W₀"]
        A3["A₃ (共享)"]
        B3["B₃ (本地私有)"]
    end
    A1 -- 上传 --> AGG
    A2 -- 上传 --> AGG
    A3 -- 上传 --> AGG
    AGG -- 下发 A_global --> A1
    AGG -- 下发 A_global --> A2
    AGG -- 下发 A_global --> A3
```

关键配置（[`fedSA-LoRA/federatedscope/llm/yamls/fedsa-lora.yaml`](../fedSA-LoRA/federatedscope/llm/yamls/fedsa-lora.yaml)）：

```yaml
federate:
  freeze_A: False          # True 时即 FFA-LoRA（冻结 A，只训练 B）
  client_num: 3
personalization:
  local_param: ['lora_B']  # B 不参与聚合 → FedSA-LoRA 的核心
llm:
  adapter:
    args: [{ adapter_package: peft, adapter_method: lora, r: 8, lora_alpha: 16 }]
```

实现位置：

- `federatedscope/core/trainers/trainer.py` → `_param_filter()`：按 `personalization.local_param` 过滤掉不上传的参数。
- `federatedscope/llm/model/model_builder.py` / `glue/model/model_builder.py`：`freeze_A` 为真时冻结 `lora_A`（FFA-LoRA 对照实验）。
- `fedSA-LoRA/exp/`：RoBERTa-large 在 GLUE-MNLI 上的 FedAvg 基线实验记录。

## 5. 目录结构

```text
.
├── app.py                  # 入口：读取 config.json → 创建 channel → 加载插件 → 启动
├── config.py               # 全部可用配置项及默认值
├── config-template.json    # 配置模板，复制为 config.json 使用
├── channel/                # ③ 接入渠道：wechat(itchat/wechaty/wcf)、wechatmp、wechatcom、wework、feishu、dingtalk、web、terminal
├── bridge/                 # 消息上下文 Context / Reply，以及按配置路由到具体 bot
├── bot/                    # 各家大模型适配；本地 vLLM/Ollama 走 chatgpt（OpenAI 兼容）
├── plugins/                # 插件：style_learner(风格学习)、role、keyword、banwords、tool、agent、godcmd…
├── voice/ translate/       # 语音识别/合成、翻译
├── lib/itchat/             # 个人微信协议库
├── finetune/               # ① 数据处理 prepare_data.py / 训练 train_lora.py / 评测 evaluate.py
├── fedSA-LoRA/             # ① 联邦低秩微调（基于 FederatedScope）
├── docs/                   # 文档：架构、技术分析、微调指南、训练报告(results/)
├── demo/ screencapture/    # 演示视频与 vLLM 部署截图
└── docker/ Dockerfile      # 容器化部署
```
