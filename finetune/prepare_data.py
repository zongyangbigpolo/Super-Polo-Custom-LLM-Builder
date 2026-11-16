"""Build chat-style SFT data (messages JSONL) from several sources.

Sources:
  wechat    CSV exported by WeChatMsg / MemoTrace style tools (one row per message)
  stylelog  conversations recorded online by plugins/style_learner
  lccc      public LCCC chit-chat corpus (used to validate the pipeline)

Every sample is: {"contact": str, "messages": [system, user, assistant, ...]}
where the last message is always an assistant turn (= what "I" said).

Examples:
  python finetune/prepare_data.py wechat --input chats.csv --out data/train.jsonl
  python finetune/prepare_data.py stylelog --input plugins/style_learner/data --out data/online.jsonl
  python finetune/prepare_data.py lccc --out data/lccc_train.jsonl --split train --num 2000
"""
import argparse
import csv
import glob
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

DEFAULT_SYSTEM = "你是Polo，正在用微信和{contact}聊天。说话口语化、简短、自然，像真人一样回复。"

PII_PATTERNS = [
    (re.compile(r"https?://\S+"), "<URL>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL>"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "<ID>"),
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "<CARD>"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "<PHONE>"),
    (re.compile(r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)"), "<PHONE>"),
]
NON_TEXT = re.compile(r"^\[(图片|视频|语音|表情|动画表情|文件|位置|链接|名片|小程序|转账|红包|聊天记录)\]$")
RECALL = re.compile(r"撤回了一条消息|拍了拍")


def anonymize(text: str) -> str:
    for pat, repl in PII_PATTERNS:
        text = pat.sub(repl, text)
    return text.strip()


def clean(text: str) -> str:
    text = (text or "").replace("\r", "").strip()
    if not text or NON_TEXT.match(text) or RECALL.search(text):
        return ""
    return anonymize(text)


def build_samples(turns, contact, system_tpl, max_context, min_len=1, max_len=200):
    """turns: list of (is_me, text). Consecutive messages from the same side are merged.
    One sample per "my" turn, with up to `max_context` previous turns as context."""
    merged = []
    for is_me, text in turns:
        if merged and merged[-1][0] == is_me:
            merged[-1][1] += "\n" + text
        else:
            merged.append([is_me, text])
    samples = []
    for i, (is_me, text) in enumerate(merged):
        if not is_me or i == 0 or not (min_len <= len(text) <= max_len):
            continue
        ctx = merged[max(0, i - max_context):i]
        while ctx and ctx[0][0]:  # context must start with the other person
            ctx = ctx[1:]
        if not ctx:
            continue
        msgs = [{"role": "system", "content": system_tpl.format(contact=contact)}]
        msgs += [{"role": "assistant" if m else "user", "content": t} for m, t in ctx]
        msgs.append({"role": "assistant", "content": text})
        samples.append({"contact": contact, "messages": msgs})
    return samples


def from_wechat(args):
    """Rows are grouped per contact, sorted by time and split into sessions by idle gap."""
    by_contact = {}
    with open(args.input, encoding=args.encoding, newline="") as f:
        for row in csv.DictReader(f):
            if args.col_type and str(row.get(args.col_type, "1")) != "1":
                continue
            text = clean(row.get(args.col_content, ""))
            if not text:
                continue
            is_me = str(row.get(args.col_sender, "")).strip() in ("1", "True", "true")
            contact = row.get(args.col_contact) or "朋友"
            ts = float(row.get(args.col_time) or 0)
            by_contact.setdefault(contact, []).append((ts, is_me, text))
    samples = []
    for idx, (contact, rows) in enumerate(by_contact.items()):
        rows.sort(key=lambda r: r[0])
        alias = contact if args.keep_names else f"联系人{idx:03d}"
        session, last_ts = [], None
        for ts, is_me, text in rows:
            if last_ts is not None and ts - last_ts > args.session_gap:
                samples += build_samples(session, alias, args.system, args.max_context)
                session = []
            session.append((is_me, text))
            last_ts = ts
        samples += build_samples(session, alias, args.system, args.max_context)
    return samples


def from_stylelog(args):
    """Records written by plugins/style_learner: one JSONL file per contact.
    Owner corrections (`corrected`) take precedence over the bot's original reply."""
    samples = []
    for path in sorted(glob.glob(os.path.join(args.input, "*.jsonl"))):
        turns, contact = [], "朋友"
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                contact = r.get("contact", contact)
                q, a = clean(r.get("query", "")), clean(r.get("corrected") or r.get("reply") or "")
                if q and a:
                    turns += [(False, q), (True, a)]
        samples += build_samples(turns, contact, args.system, args.max_context)
    return samples


def from_lccc(args):
    """Fetch LCCC-base rows through the HuggingFace datasets-server API (no full download).
    LCCC text is word-segmented with spaces; spaces next to CJK chars are removed."""
    api = "https://datasets-server.huggingface.co/rows?"
    rows, offset = [], args.offset
    while len(rows) < args.num * 2:
        q = urllib.parse.urlencode({"dataset": "silver/lccc", "config": "base", "split": args.split,
                                    "offset": offset, "length": 100})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(api + q, timeout=30) as resp:
                    batch = json.load(resp)["rows"]
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed to fetch LCCC rows at offset {offset}")
        if not batch:
            break
        rows += [r["row"]["dialog"] for r in batch]
        offset += 100
    join = lambda s: re.sub(r"(?<=[^\x00-\x7f]) | (?=[^\x00-\x7f])", "", s).strip()
    samples = []
    for dialog in rows:
        dialog = [clean(join(t)) for t in dialog]
        if len(dialog) < 2 or not all(dialog):
            continue
        n = len(dialog)  # speakers alternate; the last utterance is treated as "my" reply
        turns = [((n - 1 - i) % 2 == 0, t) for i, t in enumerate(dialog)]
        samples += build_samples(turns, "朋友", args.system, args.max_context)[-1:]
        if len(samples) >= args.num:
            break
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", choices=["wechat", "stylelog", "lccc"])
    p.add_argument("--input", help="CSV file (wechat) or directory (stylelog)")
    p.add_argument("--out", required=True)
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt, may contain {contact}")
    p.add_argument("--max-context", type=int, default=4, help="max previous turns kept as context")
    # wechat csv columns (defaults follow the WeChatMsg / MemoTrace export)
    p.add_argument("--col-content", default="StrContent")
    p.add_argument("--col-sender", default="IsSender")
    p.add_argument("--col-time", default="CreateTime")
    p.add_argument("--col-contact", default="NickName")
    p.add_argument("--col-type", default="Type")
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--session-gap", type=int, default=3600, help="seconds of silence that start a new session")
    p.add_argument("--keep-names", action="store_true", help="keep real contact names (default: aliases)")
    # lccc
    p.add_argument("--split", default="train")
    p.add_argument("--num", type=int, default=2000)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    samples = {"wechat": from_wechat, "stylelog": from_stylelog, "lccc": from_lccc}[args.source](args)
    random.Random(args.seed).shuffle(samples)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    contacts = len({s["contact"] for s in samples})
    print(f"wrote {len(samples)} samples ({contacts} contacts) -> {args.out}")


if __name__ == "__main__":
    main()
