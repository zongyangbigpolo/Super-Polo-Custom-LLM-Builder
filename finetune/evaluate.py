"""Compare the base model with the LoRA-tuned model and export report artifacts.

Outputs (in --report-dir):
  metrics.json        eval loss / perplexity / reply length / distinct-n / char-F1, base vs LoRA
  loss_curve.png      training + held-out loss curve (from train_log.json)
  samples.md          side-by-side replies on held-out conversations

Example:
  python finetune/evaluate.py --adapter outputs/polo-style-lora --eval data/eval.jsonl --report-dir docs/results
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from train_lora import encode, evaluate, load_jsonl, pick_device  # noqa: E402


def distinct(texts, n):
    grams = [t[i:i + n] for t in texts for i in range(len(t) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def char_f1(pred, ref):
    common = sum((Counter(pred) & Counter(ref)).values())
    if not common:
        return 0.0
    p, r = common / len(pred), common / len(ref)
    return 2 * p * r / (p + r)


@torch.no_grad()
def generate(model, tok, messages, device, max_new_tokens):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.1, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def plot(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot([x["step"] for x in log["train"]], [x["loss"] for x in log["train"]], label="train loss", alpha=0.8)
    ax.plot([x["step"] for x in log["eval"]], [x["loss"] for x in log["eval"]], "o-", label="held-out loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (reply tokens)")
    c = log["config"]
    ax.set_title(f"LoRA r={c['r']} alpha={c['alpha']} on {c['model'].split('/')[-1]} "
                 f"({log['train_samples']} samples, {log['device']})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--report-dir", default="docs/results")
    p.add_argument("--num-samples", type=int, default=60, help="held-out conversations used for generation metrics")
    p.add_argument("--show", type=int, default=15, help="conversations written to samples.md")
    p.add_argument("--max-new-tokens", type=int, default=48)
    args = p.parse_args()

    with open(os.path.join(args.adapter, "train_log.json"), encoding="utf-8") as f:
        log = json.load(f)
    cfg = log["config"]
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(device)
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model.eval()

    data = load_jsonl(args.eval)
    enc = [encode(tok, s["messages"], cfg["max_len"]) for s in data]
    gen_set = data[:args.num_samples]
    results = {}
    replies = {"base": [], "lora": []}
    for name in ("base", "lora"):
        ctx = model.disable_adapter() if name == "base" else torch.no_grad()
        with ctx:
            loss = evaluate(model, enc, cfg["batch_size"], pad_id, device)
            preds = [generate(model, tok, s["messages"][:-1], device, args.max_new_tokens) for s in gen_set]
        refs = [s["messages"][-1]["content"] for s in gen_set]
        replies[name] = preds
        results[name] = {
            "eval_loss": round(loss, 4),
            "perplexity": round(math.exp(loss), 2),
            "avg_reply_chars": round(sum(map(len, preds)) / len(preds), 1),
            "distinct_1": round(distinct(preds, 1), 3),
            "distinct_2": round(distinct(preds, 2), 3),
            "char_f1_vs_reference": round(sum(char_f1(a, b) for a, b in zip(preds, refs)) / len(preds), 3),
        }
        print(name, results[name], flush=True)
    refs = [s["messages"][-1]["content"] for s in gen_set]
    results["reference"] = {"avg_reply_chars": round(sum(map(len, refs)) / len(refs), 1),
                            "distinct_1": round(distinct(refs, 1), 3), "distinct_2": round(distinct(refs, 2), 3)}
    results["meta"] = {"model": cfg["model"], "eval_samples": len(enc), "generation_samples": len(gen_set),
                       "train_samples": log["train_samples"], "trainable_params": log["trainable_params"],
                       "total_params": log["total_params"], "train_time_s": log["train_time_s"],
                       "device": log["device"], "decoding": "greedy, repetition_penalty=1.1"}

    os.makedirs(args.report_dir, exist_ok=True)
    with open(os.path.join(args.report_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    plot(log, os.path.join(args.report_dir, "loss_curve.png"))
    with open(os.path.join(args.report_dir, "samples.md"), "w", encoding="utf-8") as f:
        f.write("# 微调前后回复对比（留出集，贪心解码，未经挑选）\n\n")
        f.write("按留出集顺序取前 {} 条，未做筛选。\n\n".format(args.show))
        for i, s in enumerate(gen_set[:args.show]):
            ctx = "<br>".join(("**对方**：" if m["role"] == "user" else "**我**：") + m["content"].replace("\n", " ")
                              for m in s["messages"][1:-1])
            f.write(f"### #{i + 1}\n\n{ctx}\n\n| | 回复 |\n|---|---|\n")
            f.write(f"| 真实回复 | {refs[i]} |\n| 基座模型 | {replies['base'][i].replace(chr(10), ' ')} |\n")
            f.write(f"| LoRA 微调后 | {replies['lora'][i].replace(chr(10), ' ')} |\n\n")
    print(f"report -> {args.report_dir}")


if __name__ == "__main__":
    main()
