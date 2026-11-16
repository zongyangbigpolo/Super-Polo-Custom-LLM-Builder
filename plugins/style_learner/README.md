## 插件说明

`StyleLearner`（风格学习）：按联系人记录机器人和对方的对话，让机器人对不同的人用不同的语气说话，同时为 LoRA 再训练积累数据。

### 工作方式

1. **记录**：每次回复文本后，把 `对方消息 / 机器人回复` 追加到 `data/<会话ID哈希>.jsonl`。
2. **纠正**：机器人说得不像你时，发送 `$纠正 你会怎么回`，上一条回复会被标记为你的真实说法。
3. **即时生效**：每满 `refresh_every` 轮或每次纠正后，把该联系人最近的对话（优先使用被纠正的）作为示例写进系统提示。
4. **再训练**：
   ```bash
   python finetune/prepare_data.py stylelog --input plugins/style_learner/data --out finetune/data/online.jsonl
   python finetune/train_lora.py --train finetune/data/online.jsonl --eval finetune/data/eval.jsonl
   ```

### 指令

| 指令 | 作用 |
| --- | --- |
| `$纠正 <更好的回复>` | 用你的说法替换机器人对当前联系人的上一条回复 |
| `$风格` | 查看与当前联系人已记录、已纠正的对话数 |

### 配置（`config.json`，可从 `config.json.template` 复制）

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `record` | `true` | 是否记录对话 |
| `inject_examples` | `true` | 是否把历史对话作为风格示例注入系统提示 |
| `max_examples` | `5` | 注入的示例条数 |
| `refresh_every` | `5` | 每多少轮刷新一次示例（刷新会重置该会话的上下文） |
| `owner_nicknames` | `[]` | 允许使用 `$纠正` 的昵称，留空表示不限制 |
| `record_groups` | `false` | 是否记录群聊 |

> 记录中包含他人的聊天内容，请只在本地保存，并告知对方。`data/` 已在 `.gitignore` 中。
