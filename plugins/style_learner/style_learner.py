# encoding:utf-8
"""StyleLearner: record conversations per contact and adapt the reply style to each person.

Two learning loops:
  1. Online (instant): the latest (preferably owner-corrected) exchanges with a contact are
     injected into that contact's system prompt as style examples.
  2. Offline (periodic): records under data/ are converted by
     `finetune/prepare_data.py stylelog` and used to retrain the LoRA adapter.

Commands (plugin_trigger_prefix, default "$"):
  $纠正 <更好的回复>   replace the bot's last reply to this contact with how I would really say it
  $风格               show how many exchanges have been learned for this contact
"""
import hashlib
import json
import os
import threading
import time

import plugins
from bridge.bridge import Bridge
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf
from plugins import *


@plugins.register(
    name="StyleLearner",
    desire_priority=50,
    namecn="风格学习",
    desc="按联系人记录聊天并学习说话风格，支持纠正回复，为 LoRA 再训练积累数据",
    version="1.0",
    author="polo",
)
class StyleLearner(Plugin):
    def __init__(self):
        super().__init__()
        self.config = super().load_config() or self._load_config_template()
        self.data_dir = os.path.join(self.path, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.cache = {}  # contact_key -> list of records
        self.applied = set()  # contacts whose session already carries the style prompt
        self.handlers[Event.ON_HANDLE_CONTEXT] = self.on_handle_context
        self.handlers[Event.ON_SEND_REPLY] = self.on_send_reply
        logger.info("[StyleLearner] inited, data_dir={}".format(self.data_dir))

    # ---------- storage ----------
    @staticmethod
    def _contact(context):
        msg = context.get("msg")
        if context.get("isgroup", False):
            name = getattr(msg, "actual_user_nickname", None) or getattr(msg, "other_user_nickname", None)
        else:
            name = getattr(msg, "other_user_nickname", None) or getattr(msg, "from_user_nickname", None)
        sid = str(context.get("session_id") or name or "unknown")
        return hashlib.md5(sid.encode("utf-8")).hexdigest()[:12], name or "朋友"

    def _path(self, key):
        return os.path.join(self.data_dir, f"{key}.jsonl")

    def _records(self, key):
        if key not in self.cache:
            records = []
            if os.path.exists(self._path(key)):
                with open(self._path(key), encoding="utf-8") as f:
                    records = [json.loads(line) for line in f if line.strip()]
            self.cache[key] = records
        return self.cache[key]

    def _rewrite(self, key):
        with open(self._path(key), "w", encoding="utf-8") as f:
            for r in self.cache[key]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- style prompt ----------
    def _style_prompt(self, records, contact):
        base = conf().get("character_desc", "")
        n = self.config.get("max_examples", 5)
        good = [r for r in records if r.get("corrected")][-n:]
        examples = good + [r for r in records if not r.get("corrected")][-(n - len(good)):] if len(good) < n else good
        if not examples:
            return base
        lines = [f"对方：{r['query']}\n我：{r.get('corrected') or r['reply']}" for r in examples]
        return (f"{base}\n\n你正在和「{contact}」聊天。下面是你们之前的真实对话，"
                f"请保持同样的语气、称呼和回复长度：\n" + "\n".join(lines))

    def _apply_style(self, session_id, key, contact):
        records = self._records(key)
        if not records or not self.config.get("inject_examples", True):
            return
        sessions = getattr(Bridge().get_bot("chat"), "sessions", None)
        if sessions is None:
            return
        sessions.build_session(session_id, system_prompt=self._style_prompt(records, contact))
        self.applied.add(key)

    # ---------- events ----------
    def on_handle_context(self, e_context: EventContext):
        context = e_context["context"]
        if context.type != ContextType.TEXT:
            return
        content = context.content.strip()
        prefix = conf().get("plugin_trigger_prefix", "$")
        key, contact = self._contact(context)

        if content.startswith(prefix + "纠正"):
            e_context["reply"] = Reply(ReplyType.INFO, self._correct(context, key, content[len(prefix) + 2:].strip()))
            e_context.action = EventAction.BREAK_PASS
            return
        if content.startswith(prefix + "风格"):
            records = self._records(key)
            fixed = sum(1 for r in records if r.get("corrected"))
            e_context["reply"] = Reply(ReplyType.INFO, f"与「{contact}」已记录 {len(records)} 轮对话，其中 {fixed} 条被纠正。")
            e_context.action = EventAction.BREAK_PASS
            return
        if context.get("isgroup", False) and not self.config.get("record_groups", False):
            return
        context["style_learner"] = {"key": key, "contact": contact, "query": content}
        if key not in self.applied:  # e.g. first message after a restart
            self._apply_style(context["session_id"], key, contact)

    def on_send_reply(self, e_context: EventContext):
        context, reply = e_context["context"], e_context["reply"]
        info = context.get("style_learner") if context else None
        if not info or not self.config.get("record", True) or not reply or reply.type != ReplyType.TEXT:
            return
        record = {"ts": int(time.time()), "contact": info["contact"], "is_group": context.get("isgroup", False),
                  "query": info["query"], "reply": reply.content}
        with self.lock:
            records = self._records(info["key"])
            records.append(record)
            with open(self._path(info["key"]), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # updating the system prompt resets the session history, so only refresh every N exchanges
        if len(records) % max(1, self.config.get("refresh_every", 5)) == 0:
            self._apply_style(context["session_id"], info["key"], info["contact"])

    def _correct(self, context, key, text):
        owners = self.config.get("owner_nicknames", [])
        msg = context.get("msg")
        sender = getattr(msg, "actual_user_nickname", None) or getattr(msg, "from_user_nickname", None)
        if owners and sender not in owners:
            return "只有主人可以纠正回复"
        if not text:
            return "用法：$纠正 <你会怎么回>"
        with self.lock:
            records = self._records(key)
            if not records:
                return "还没有可以纠正的回复"
            records[-1]["corrected"] = text
            self._rewrite(key)
        self._apply_style(context["session_id"], key, records[-1]["contact"])
        return "已记住，下次会更像你 ✅"

    def get_help_text(self, **kwargs):
        prefix = conf().get("plugin_trigger_prefix", "$")
        return (f"按联系人学习聊天风格。\n{prefix}纠正 <更好的回复>：用你的说法替换机器人上一条回复\n"
                f"{prefix}风格：查看与当前联系人的学习进度")

    def _load_config_template(self):
        path = os.path.join(self.path, "config.json.template")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}
