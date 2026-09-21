"""Quick fine-tune of a Laya checkpoint on teacher-labelled rows. Soft cross-entropy over the option logits.

    ../.venv/bin/python train_laya.py --out ../models/laya-era-distill-v0

Follows the author's notebook where it can: AdamW, encoder lr 2.5e-5, head lr 1e-4, weight decay 0.01, clip 1.0,
effective batch 64. The notebook adds a policy-gradient term (RLCD). This script uses only its cross-entropy term.
The saved directory loads with `laya.load(path)`.
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch

import laya
from laya.common import QTYPES, build_sequence, collate_items

HERE = Path(__file__).resolve().parent


def targets(q, ans):
    """Teacher answer -> distribution over the options, in option order."""
    t = q["type"]
    if t == "choice":
        keys = list(q["criteria"])
        if ans.get("choice") not in keys:
            return None
        eps = 0.05
        return [1 - eps if k == ans["choice"] else eps / (len(keys) - 1) for k in keys]
    if t == "score":
        k, c = len(q["criteria"]), int(ans["score"])
        out = [0.0] * k
        out[c] = 0.8
        near = [i for i in (c - 1, c + 1) if 0 <= i < k]
        for i in near:
            out[i] = 0.2 / len(near)
        return out
    p = min(0.98, max(0.02, ans["noul"]))
    return [1 - p, p]


def load_rows(path, questions_for):
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        # Tool rows: keep a row only when the teacher label equals the intent the request was written for.
        if r["qset"].startswith("tool") and r["answers"].get("tool", {}).get("choice") != r.get("intent"):
            continue
        for qid, q in questions_for(r).items():
            a = r["answers"].get(qid)
            tgt = targets(q, a) if a else None
            if tgt:
                rows.append({"state": {"request": r["request"]}, "q": q, "target": tgt, "src": r["source"]})
    return rows


def encode(agent, row, rng, shuffle):
    q = agent._to_internal(row["q"])
    k = len(row["target"])
    order = list(range(k))
    if shuffle and q["t"] == "choice":
        rng.shuffle(order)  # option order must not carry the answer
    seq, markers = build_sequence(agent.tok, row["state"], q, agent.cfg.get("max_len", 512),
                                  agent.cfg.get("head_max_len", 192), option_order=order)
    if len(markers) != k:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]], "target": [row["target"][i] for i in order]}


def run_batch(agent, items, device, pad_to_multiple=0):
    b = collate_items([items], agent.tok.pad_token_id)
    if pad_to_multiple:
        from moderation_padding import pad_batch
        b = pad_batch(b, agent.tok.pad_token_id, pad_to_multiple)
    logits, _ = agent.model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                            b["marker_mask"].to(device), b["qtype"].to(device))
    mask = b["marker_mask"].to(device)
    logp = torch.log_softmax(logits.float().masked_fill(~mask, -1e4), -1)
    target = b["target"].to(device)
    loss = -(target * logp).sum(-1).mean()
    agree = (logp.argmax(-1) == target.argmax(-1)).float().sum().item()
    return loss, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--data", default=str(HERE.parent / "train" / "distill.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from distill import ROUTE_Q, toolboxes
    boxes = toolboxes()
    rows = load_rows(args.data, lambda r: ROUTE_Q if r["qset"] == "route" else {"tool": boxes[r["qset"].split(":")[1]]})
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_dev = max(50, len(rows) // 10)
    dev, train = rows[:n_dev], rows[n_dev:]
    print("rows: train %d, dev %d" % (len(train), len(dev)), flush=True)

    agent = laya.load(args.base)
    device, model = agent.device, agent.model
    model.float().train()
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.01)
    steps = (len(train) // (args.micro_batch * args.accum)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2.5e-5, 1e-4], total_steps=max(steps, 1), pct_start=0.1)

    def evaluate():
        model.eval()
        tot = agree = loss_sum = 0
        with torch.no_grad():
            for i in range(0, len(dev), 32):
                items = [x for x in (encode(agent, r, rng, False) for r in dev[i:i + 32]) if x]
                loss, a = run_batch(agent, items, device)
                tot, agree, loss_sum = tot + len(items), agree + a, loss_sum + loss.item() * len(items)
        model.train()
        return loss_sum / tot, agree / tot

    print("dev before: loss %.3f  teacher agreement %.3f" % evaluate(), flush=True)
    t0, step = time.time(), 0
    for epoch in range(args.epochs):
        rng.shuffle(train)
        for i in range(0, len(train) - args.micro_batch + 1, args.micro_batch):
            items = [x for x in (encode(agent, r, rng, True) for r in train[i:i + args.micro_batch]) if x]
            loss, _ = run_batch(agent, items, device)
            (loss / args.accum).backward()
            if (i // args.micro_batch + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if step < steps:
                    sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print("epoch %d step %d/%d loss %.3f  %.0f s" % (epoch + 1, step, steps, loss.item(), time.time() - t0), flush=True)
        print("dev after epoch %d: loss %.3f  teacher agreement %.3f" % ((epoch + 1,) + evaluate()), flush=True)

    # Save as a loadable Laya directory: base snapshot files plus the new weights.
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    base_dir = Path(args.base) if Path(args.base).exists() else Path(snapshot_download(args.base, allow_patterns=[
        "rl_agent_config.json", "encoder/*", "tokenizer/*"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("encoder", "tokenizer"):
        shutil.copytree(base_dir / name, out / name, dirs_exist_ok=True)
    cfg = dict(agent.cfg)
    # The base calibration temperatures were fitted to the base logits. They do not fit the new logits.
    cfg["temperature"], cfg["temperature_by_options"] = [1.0, 1.0, 1.0], {}
    cfg["finetune"] = {"base": args.base, "data": args.data, "rows": len(train), "epochs": args.epochs,
                       "loss": "soft cross-entropy", "date": time.strftime("%Y-%m-%d")}
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=1))
    model.eval()
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
