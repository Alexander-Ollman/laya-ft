"""Pinned public Banking77 preparation and evaluation, independent of private corpora.

Run from bench/: python public_workflow.py prepare | evaluate --help
The official test is never sampled into training or development.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "train" / "public-banking77"
REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
HF_REVISION = "90d4e2ee5521c04fc1488f065b8b083658768c57"
SOURCE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/" + REVISION
INSTRUCTION = "Which banking customer-support intent best matches the customer's request?"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def norm(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def read_jsonl(path):
    # JSONL records are LF-delimited; Unicode line separators can occur inside
    # valid JSON strings and must not split an input record.
    return [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]


def split_rows(train, test, seed=7, train_per_class=13, dev_per_class=10):
    """Remove test overlap and ambiguous training texts before stratified splitting."""
    forbidden = {norm(row["text"]) for row in test}
    groups = defaultdict(list)
    for row in train:
        groups[norm(row["text"])].append(row)
    clean = defaultdict(list)
    excluded = Counter()
    for key, group in sorted(groups.items()):
        if key in forbidden:
            excluded["test_overlap"] += len(group)
        elif len({r["category"] for r in group}) > 1:
            excluded["conflicting_labels"] += len(group)
        else:
            clean[group[0]["category"]].append(group[0])
            excluded["duplicate_train"] += len(group) - 1
    rng = random.Random(seed)
    fit, dev = [], []
    for label, rows in sorted(clean.items()):
        rng.shuffle(rows)
        if len(rows) < train_per_class + dev_per_class:
            raise ValueError("not enough unique rows for " + label)
        dev.extend(rows[:dev_per_class])
        fit.extend(rows[dev_per_class:dev_per_class + train_per_class])
    return fit, dev, dict(excluded)


def make_case(row, split, labels):
    # Label strings alone are shared by every model; no examples or gold in input.
    question = {"type": "choice", "instructions": INSTRUCTION,
                "criteria": {label: None for label in labels}}
    return {"id": "banking77-" + split + "-" + str(row["index"]),
            "tier": 1, "category": "banking77", "split": split,
            "state": {"request": row["text"]}, "questions": {"intent": question},
            "expected": {"intent": row["category"]},
            "label_source": "PolyAI Banking77 human labels; CC BY 4.0; " + REVISION}


def prepare(args):
    if DATA.exists():
        raise FileExistsError("Frozen dataset already exists: " + str(DATA))
    raw = {}
    for name in ("train.csv", "test.csv"):
        raw[name] = urllib.request.urlopen(SOURCE + "/banking_data/" + name, timeout=60).read()
    raw["LICENSE"] = urllib.request.urlopen(SOURCE + "/LICENSE", timeout=60).read()
    rows = {}
    for split in ("train", "test"):
        rows[split] = [dict(r, index=i) for i, r in enumerate(csv.DictReader(io.StringIO(raw[split + ".csv"].decode())))]
    labels = sorted({r["category"] for r in rows["train"]})
    assert len(labels) == 77 and len(rows["train"]) == 10003 and len(rows["test"]) == 3080
    fit, dev, excluded = split_rows(rows["train"], rows["test"])
    DATA.mkdir(parents=True)
    for name, content in raw.items():
        (DATA / name).write_bytes(content)
    hashes = {}
    for split, examples in (("train", fit), ("development", dev), ("heldout", rows["test"])):
        blob = "".join(json.dumps(make_case(row, split, labels), ensure_ascii=False) + "\n" for row in examples).encode()
        (DATA / (split + ".jsonl")).write_bytes(blob)
        hashes[split] = digest(blob)
    manifest = {"dataset": "PolyAI/banking77", "license": "CC-BY-4.0",
                "huggingface_revision": HF_REVISION, "source_revision": REVISION,
                "source": SOURCE, "seed": 7, "train": len(fit), "development": len(dev),
                "test": len(rows["test"]), "classes": len(labels), "train_per_class": 13,
                "dev_per_class": 10, "exclusions": excluded,
                "raw_sha256": {k: digest(v) for k, v in raw.items()}, "split_sha256": hashes,
                "protocol": {"max_len": 1024, "head_max_len": 768, "epochs": 3,
                             "base": "convaiinnovations/laya", "selection": "fixed final epoch",
                             "calibration": "development only; no test-based tuning",
                             "primary_metric": "77-way accuracy on all 3080 official test rows",
                             "test_duplicate_groups": len({norm(r['text']) for r in rows['test']})},
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def validate_case_file(path, data=DATA):
    """Only frozen split bytes can be evaluated under this protocol."""
    manifest = json.loads((data / "manifest.json").read_text())
    for split, expected in manifest["split_sha256"].items():
        if digest((data / (split + ".jsonl")).read_bytes()) != expected:
            raise ValueError("frozen split hash mismatch: " + split)
    actual = digest(Path(path).read_bytes())
    if actual not in manifest["split_sha256"].values():
        raise ValueError("evaluation cases are not a frozen protocol split")
    return actual


def parse_answer(answers, labels):
    """Validate the decision; preserve raw response separately in telemetry.

    Complete distributions within 2% of unit mass are normalized to correct
    response rounding; other probability payloads are excluded from scoring.
    """
    answer = answers.get("intent") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("choice") not in labels:
        raise ValueError("invalid or missing choice")
    probs = answer.get("probabilities")
    if (not isinstance(probs, dict) or set(probs) != set(labels)
            or any(isinstance(p, bool) or not isinstance(p, (float, int))
                   or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values())
            or abs(sum(probs.values()) - 1) > .02):
        probs = None
    else:
        total = sum(probs.values())
        probs = {key: value / total for key, value in probs.items()}
    return answer["choice"], probs


def cost_summary(records):
    """Unknown API charges must not silently become zero-dollar charges."""
    costs = []
    for row in records:
        usage = row.get("usage")
        value = usage.get("cost_usd") if isinstance(usage, dict) else None
        if (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0):
            costs.append(value)
    return {"cost_usd": sum(costs) if len(costs) == len(records) else None,
            "reported_cost_usd": sum(costs), "cost_reported_rows": len(costs),
            "cost_unknown_rows": len(records) - len(costs)}


def summarize(records, labels):
    """All requests enter accuracy; probability metrics require a valid distribution."""
    n = len(records)
    if not n:
        raise ValueError("empty evaluation")
    correct = sum(r["prediction"] == r["expected"] and not r["error"] for r in records)
    p = correct / n
    z = 1.959963984540054
    half = z * math.sqrt(p * (1 - p) / n + z*z / (4*n*n)) / (1 + z*z/n)
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    f1s = []
    for label in labels:
        tp = sum(r["expected"] == label and r["prediction"] == label for r in records)
        fp = sum(r["expected"] != label and r["prediction"] == label for r in records)
        fn = sum(r["expected"] == label and r["prediction"] != label for r in records)
        f1s.append(2*tp / (2*tp + fp + fn) if 2*tp + fp + fn else 0)
    valid = [r for r in records if not r["error"] and r.get("probabilities")]
    brier = sum(sum((r["probabilities"][k] - (k == r["expected"]))**2 for k in labels) for r in valid)
    nll = sum(-math.log(max(r["probabilities"][r["expected"]], 1e-12)) for r in valid)
    ece = 0
    for bucket in range(10):
        group = [r for r in valid if min(9, int(r["probabilities"][r["prediction"]] * 10)) == bucket]
        if group:
            confidence = sum(r["probabilities"][r["prediction"]] for r in group) / len(group)
            accuracy = sum(r["prediction"] == r["expected"] for r in group) / len(group)
            ece += len(group) / len(valid) * abs(confidence - accuracy)
    latencies = sorted(r["latency_ms"] for r in records if not r["error"])
    return {"n": n, "correct": correct, "accuracy": p, "wilson_95": [center-half, center+half],
            "macro_f1": sum(f1s) / len(labels), "errors": sum(bool(r["error"]) for r in records),
            "probability_rows": len(valid), "multiclass_brier": brier / len(valid) if valid else None,
            "nll": nll / len(valid) if valid else None, "ece_10_equal_width": ece if valid else None,
            "p50_ms": latencies[len(latencies)//2] if latencies else None,
            "p95_ms": latencies[min(len(latencies)-1, int(.95*len(latencies)))] if latencies else None,
            **cost_summary(records)}


def evaluate(args):
    from run_bench import load_env, make_target
    from telemetry import Recorder
    load_env()
    cases_sha256 = validate_case_file(args.cases)
    cases = read_jsonl(args.cases)
    labels = list(cases[0]["questions"]["intent"]["criteria"])
    target = make_target(args.target)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    run_id = stamp + "_banking77_" + re.sub(r"[^A-Za-z0-9.-]", "_", args.target)
    rec = Recorder(run_id, "public/PolyAI-banking77", meta={"cases_sha256": cases_sha256,
                   "target": args.target, "protocol": "public-banking77-v1", "warmup": "one development request"})
    rec.attach(__file__)
    # Warm-up is archived, uses development only, and is excluded from metrics.
    warm = read_jsonl(DATA / "development.jsonl")[0]
    rows = []
    try:
        for i, case in enumerate([warm] + cases):
            t0 = time.perf_counter()
            answers, usage, resolved, error = {}, {}, None, None
            prediction, probs = None, None
            try:
                answers, usage, resolved = target.ask(case)
                prediction, probs = parse_answer(answers, labels)
            except Exception as exc:
                error = type(exc).__name__ + ": " + str(exc)[:250]
            ms = (time.perf_counter()-t0)*1000
            rec.record(args.target.split(":")[0], target.model, case["state"], case["questions"], answers,
                       model_resolved=resolved, case_id=case["id"], split=case["split"],
                       expected=case["expected"], label_source=case["label_source"], latency_ms=ms,
                       usage=usage, error=error)
            if i == 0:
                continue
            rows.append({"id": case["id"], "text_group": digest(norm(case["state"]["request"]).encode()),
                         "expected": case["expected"]["intent"], "prediction": prediction,
                         "probabilities": probs, "error": error, "latency_ms": ms, "usage": usage,
                         "model_resolved": resolved})
            if i % 100 == 0:
                print(f"{i}/{len(cases)}", flush=True)
        result = {"target": args.target, "cases": str(args.cases), "cases_sha256": cases_sha256,
                  "summary": summarize(rows, labels), "rows": rows}
        path = ROOT / "bench" / "results" / (run_id + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1) + "\n")
        rec.attach(path, DATA / "manifest.json")
        print(json.dumps(result["summary"], indent=2), flush=True)
        print("saved", path, flush=True)
    finally:
        rec.close(outcome=f"evaluated {len(rows)}/{len(cases)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--cases", type=Path, default=DATA / "heldout.jsonl")
    ev.add_argument("--target", required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args)
