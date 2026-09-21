"""Supervised public-data fine-tune with fixed request splits and complete batches.

Uses the established Laya soft-cross-entropy implementation, without importing
the private dataset builder. All outputs and input hashes are archived.
"""
import argparse
import hashlib
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

from public_workflow import DATA, read_jsonl, norm
from telemetry import Recorder


def load_split(path):
    from train_laya import targets

    cases = read_jsonl(path)
    rows = []
    for case in cases:
        for qid, q in case["questions"].items():
            rows.append({"state": case["state"], "q": q,
                         "target": targets(q, {"choice": case["expected"][qid]}), "id": case["id"]})
    return cases, rows


def validate_splits(train_cases, dev_cases, test_cases):
    sets = [{norm(c["state"]["request"]) for c in cases} for cases in (train_cases, dev_cases, test_cases)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("request text overlaps across splits")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--revision", default="1c5edc17a7acd8701df6fc341c0d179f1c62c982")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--pad-to-multiple",type=int,default=0)
    ap.add_argument("--mps-cache-clear-interval",type=int,default=1,
                    help="release unused Metal allocator cache every N optimizer steps; 0 disables")
    args = ap.parse_args(argv)
    if min(args.epochs, args.micro_batch, args.accum) < 1:
        ap.error("epochs and batch sizes must be positive")
    if args.pad_to_multiple<0:ap.error('padding multiple must be nonnegative')
    if args.mps_cache_clear_interval<0:ap.error('cache clearing interval must be nonnegative')
    return args


def run_identity(manifest):
    dataset = manifest["dataset"]
    # Retain the original Banking77 archive naming convention by default.
    default_slug = "banking77" if dataset == "PolyAI/banking77" else dataset
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", manifest.get("slug", default_slug)).strip("-").lower()
    if not slug:
        raise ValueError("dataset run slug must contain letters or numbers")
    return slug, "public/" + dataset.replace("/", "-")


def token_budgets(manifest):
    protocol = manifest.get("protocol", {})
    budgets = {key: protocol.get(key, default) for key, default in (("max_len", 1024), ("head_max_len", 768))}
    if any(type(value) is not int or value <= 0 for value in budgets.values()):
        raise ValueError("token budgets must be positive integers")
    return budgets


def main():
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    data = args.data.resolve()
    manifest = json.loads((data / "manifest.json").read_text())
    slug, source = run_identity(manifest)
    budgets = token_budgets(manifest)
    for split, expected in manifest["split_sha256"].items():
        if hashlib.sha256((data / (split + ".jsonl")).read_bytes()).hexdigest() != expected:
            raise ValueError("split hash mismatch: " + split)
    train_cases, train = load_split(data / "train.jsonl")
    dev_cases, dev = load_split(data / "development.jsonl")
    validate_splits(train_cases, dev_cases, read_jsonl(data / "heldout.jsonl"))
    if not train or not dev:
        raise ValueError("training and development splits must be nonempty")
    import torch
    import laya
    from train_laya import encode, run_batch
    from functools import partial
    run_batch=partial(run_batch,pad_to_multiple=args.pad_to_multiple)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    rec = Recorder(time.strftime("%Y%m%dT%H%M%S") + "_train_public_" + slug, source,
                   meta={"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                         "manifest": manifest, "seed": args.seed, "data": str(data), "loss": "label-smoothed cross entropy",
                         "calibration": "not fitted; confidence analysis must use development data"})
    rec.attach(__file__, Path(__file__).with_name("train_laya.py"), data / "manifest.json")
    outcome = "failed"
    history = []
    try:
        from huggingface_hub import snapshot_download
        # Resolve once, so loading and saving cannot straddle a repository update.
        base = Path(args.base) if Path(args.base).exists() else Path(snapshot_download(args.base, revision=args.revision,
                    allow_patterns=["rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*"]))
        rec.meta["base_snapshot"] = str(base)
        agent = laya.load(str(base))
        agent.cfg.update(**budgets)
        model, device = agent.model, agent.device
        model.float()
        encoder = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
        head = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
        opt = torch.optim.AdamW([{"params": encoder, "lr": 2.5e-5}, {"params": head, "lr": 1e-4}], weight_decay=.01)
        group_size = args.micro_batch * args.accum
        total = math.ceil(len(train) / group_size) * args.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2.5e-5, 1e-4], total_steps=total, pct_start=.1)

        def encoded(rows, shuffle):
            items = [encode(agent, row, rng, shuffle) for row in rows]
            if any(item is None for item in items):
                raise ValueError("option truncation: all choices must be retained")
            return items

        def evaluate():
            model.eval()
            loss_sum = correct = count = 0
            with torch.no_grad():
                for start in range(0, len(dev), args.micro_batch):
                    items = encoded(dev[start:start + args.micro_batch], False)
                    loss, agree = run_batch(agent, items, device)
                    loss_sum += loss.item() * len(items)
                    correct += agree
                    count += len(items)
            return {"loss": loss_sum/count, "accuracy": correct/count, "n": count}

        history.append({"epoch": 0, **evaluate()})
        print("development", history[-1], flush=True)
        started, step = time.time(), 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            rng.shuffle(train)
            for start in range(0, len(train), group_size):
                group = train[start:start + group_size]
                opt.zero_grad(set_to_none=True)
                for pos in range(0, len(group), args.micro_batch):
                    items = encoded(group[pos:pos + args.micro_batch], True)
                    loss, _ = run_batch(agent, items, device)
                    (loss * (len(items) / len(group))).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                sched.step()
                step += 1
                if device.type=='mps' and args.mps_cache_clear_interval and step%args.mps_cache_clear_interval==0:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                memory = f' MPS allocated {torch.mps.current_allocated_memory()/1e9:.1f}GB driver {torch.mps.driver_allocated_memory()/1e9:.1f}GB' if device.type=='mps' else ''
                print(f"epoch {epoch} step {step}/{total} loss {loss.item():.4f} {time.time()-started:.0f}s{memory}", flush=True)
            history.append({"epoch": epoch, **evaluate()})
            print("development", history[-1], flush=True)
        from safetensors.torch import save_file
        args.out.mkdir(parents=True)
        for name in ("encoder", "tokenizer"):
            shutil.copytree(base/name, args.out/name)
        cfg = dict(agent.cfg)
        cfg.update(temperature=[1., 1., 1.], temperature_by_options={})
        cfg["finetune"] = {"dataset": manifest["dataset"], "train_rows": len(train), "epochs": args.epochs,
                           "pad_to_multiple": args.pad_to_multiple,
                           "mps_cache_clear_interval": args.mps_cache_clear_interval,
                           "seed": args.seed, "data": str(data), "revision": args.revision,
                           "manifest": manifest, "development": history,
                           "loss": "label-smoothed cross entropy", "base": args.base,
                           "base_snapshot": str(base)}
        (args.out/"rl_agent_config.json").write_text(json.dumps(cfg, indent=1))
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(args.out/"model.safetensors"))
        rec.attach(args.out/"rl_agent_config.json")
        print("saved", args.out, flush=True)
        outcome = "completed"
    finally:
        rec.meta["development"] = history
        rec.close(outcome=outcome)


if __name__ == "__main__":
    main()
