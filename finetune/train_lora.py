"""LoRA supervised fine-tuning for chat style (works on CUDA, Apple MPS and CPU).

Only the final assistant turn of each sample is supervised (prompt tokens are masked),
so the adapter learns *how I reply* given the conversation context.

Example:
  python finetune/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --train data/train.jsonl --eval data/eval.jsonl --out outputs/polo-style-lora
"""
import argparse
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(tokenizer, messages, max_len):
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    # strip a trailing newline after <|im_end|> so the target ends with the EOS token
    while len(f_ids) > len(p_ids) and tokenizer.decode(f_ids[-1:]).strip() == "":
        f_ids = f_ids[:-1]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    if len(f_ids) > max_len:  # keep the reply, drop the oldest context
        f_ids, labels = f_ids[-max_len:], labels[-max_len:]
    return {"input_ids": f_ids, "labels": labels}


def batches(data, bs, pad_id, shuffle, seed=0):
    idx = list(range(len(data)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i + bs]]
        L = max(len(x["input_ids"]) for x in chunk)
        ids = torch.full((len(chunk), L), pad_id)
        lab = torch.full((len(chunk), L), -100)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, x in enumerate(chunk):
            n = len(x["input_ids"])
            ids[k, :n] = torch.tensor(x["input_ids"])
            lab[k, :n] = torch.tensor(x["labels"])
            att[k, :n] = 1
        yield ids, lab, att


def reply_loss(model, ids, att, lab):
    """Summed cross-entropy over reply tokens only.
    Projects just the supervised positions through lm_head instead of the full sequence,
    which avoids materialising (batch x seq x 152k-vocab) logits and saves several GB."""
    inner = model.get_base_model()
    hidden = inner.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    mask = tgt != -100
    logits = inner.lm_head(hidden[mask]).float()
    return torch.nn.functional.cross_entropy(logits, tgt[mask], reduction="sum"), mask.sum()


@torch.no_grad()
def evaluate(model, data, bs, pad_id, device):
    model.eval()
    total, count = 0.0, 0
    for ids, lab, att in batches(data, bs, pad_id, shuffle=False):
        loss, n = reply_loss(model, ids.to(device), att.to(device), lab.to(device))
        total += loss.item()
        count += n.item()
    model.train()
    return total / max(count, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--out", default="outputs/polo-style-lora")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        target_modules=args.target_modules.split(","), task_type="CAUSAL_LM"))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"device={device} trainable={trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    train = [encode(tok, s["messages"], args.max_len) for s in load_jsonl(args.train)]
    evals = [encode(tok, s["messages"], args.max_len) for s in load_jsonl(args.eval)]
    steps_per_epoch = math.ceil(len(train) / args.batch_size / args.grad_accum)
    total_steps = int(steps_per_epoch * args.epochs)
    warmup = max(1, int(total_steps * args.warmup))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs(args.out, exist_ok=True)
    log = {"config": vars(args), "device": device, "trainable_params": trainable, "total_params": total,
           "train_samples": len(train), "eval_samples": len(evals), "train": [], "eval": []}
    base_eval = evaluate(model, evals, args.batch_size, pad_id, device)
    log["eval"].append({"step": 0, "loss": base_eval, "ppl": math.exp(base_eval)})
    print(f"step 0 eval_loss={base_eval:.4f} ppl={math.exp(base_eval):.2f}")

    step, t0, run_loss, micro = 0, time.time(), 0.0, 0
    model.train()
    epoch = 0
    while step < total_steps:
        for ids, lab, att in batches(train, args.batch_size, pad_id, shuffle=True, seed=args.seed + epoch):
            loss_sum, n = reply_loss(model, ids.to(device), att.to(device), lab.to(device))
            loss = loss_sum / n.clamp(min=1)
            (loss / args.grad_accum).backward()
            run_loss += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % args.log_every == 0:
                avg = run_loss / (args.log_every * args.grad_accum)
                log["train"].append({"step": step, "loss": avg, "lr": sched.get_last_lr()[0],
                                     "elapsed_s": round(time.time() - t0, 1)})
                print(f"step {step}/{total_steps} loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"{time.time() - t0:.0f}s", flush=True)
                run_loss = 0.0
            if step % args.eval_every == 0 or step == total_steps:
                ev = evaluate(model, evals, args.batch_size, pad_id, device)
                log["eval"].append({"step": step, "loss": ev, "ppl": math.exp(ev)})
                print(f"step {step} eval_loss={ev:.4f} ppl={math.exp(ev):.2f}", flush=True)
            if step >= total_steps:
                break
        epoch += 1

    log["train_time_s"] = round(time.time() - t0, 1)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"saved adapter -> {args.out} ({log['train_time_s']}s)")


if __name__ == "__main__":
    main()
