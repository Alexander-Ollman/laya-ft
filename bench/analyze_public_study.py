"""CPU-only rescoring, grouped paired uncertainty and development calibration.

Reads completed results without altering them. Calibration approximates logits
from rounded stored probabilities, with a 1e-12 floor. No model calls occur.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from public_workflow import DATA, ROOT, digest, norm, parse_answer, read_jsonl, summarize


def normalized_rows(rows, labels):
    output = []
    for source in rows:
        row = dict(source)
        row.setdefault("error", None)
        if row["error"]:
            row["prediction"], row["probabilities"] = None, None
        else:
            try:
                row["prediction"], row["probabilities"] = parse_answer(
                    {"intent": {"choice": row.get("prediction"),
                                "probabilities": row.get("probabilities")}}, labels)
            except ValueError as exc:
                row.update(prediction=None, probabilities=None, error=str(exc))
        output.append(row)
    return output


def paired_rows(base, fine):
    def index(rows):
        result = {r["id"]: r for r in rows}
        if len(result) != len(rows) or not result:
            raise ValueError("empty results or duplicate case IDs")
        return result
    left, right = index(base), index(fine)
    if left.keys() != right.keys():
        raise ValueError("paired case IDs differ")
    pairs = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if any(a.get(field) != b.get(field) for field in ("expected", "text_group")):
            raise ValueError("paired labels or text groups differ: " + key)
        if not a.get("text_group"):
            raise ValueError("missing text group: " + key)
        pairs.append((a, b))
    return pairs


def grouped_bootstrap(base, fine, resamples=5000, seed=7):
    """Sample request-text clusters with replacement; retain all rows per cluster."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    groups = defaultdict(list)
    for a, b in paired_rows(base, fine):
        correct = lambda r: int(not r.get("error") and r["prediction"] == r["expected"])
        groups[a["text_group"]].append(correct(b) - correct(a))
    sums = np.array([sum(groups[k]) for k in sorted(groups)], dtype=float)
    sizes = np.array([len(groups[k]) for k in sorted(groups)], dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    # Bounded memory, even for the full official test.
    for start in range(0, resamples, 100):
        indices = rng.integers(0, len(groups), size=(min(100, resamples-start), len(groups)))
        draws.extend((sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)).tolist())
    return {"accuracy_delta_fine_minus_base": float(sums.sum()/sizes.sum()),
            "percentile_95": np.quantile(draws, [.025, .975]).tolist(),
            "n_rows": int(sizes.sum()), "n_text_groups": len(groups),
            "resamples": resamples, "seed": seed,
            "method": "paired request-text cluster bootstrap, row-weighted accuracy",
            "limitation": "Conditional on these checkpoints and test data; not training-seed robustness."}


def probability_matrix(rows, labels):
    valid = [r for r in rows if not r.get("error") and r.get("probabilities")]
    if not valid:
        raise ValueError("no valid probability rows")
    logp = np.log(np.maximum([[r["probabilities"][k] for k in labels] for r in valid], 1e-12))
    y = np.array([labels.index(r["expected"]) for r in valid])
    return valid, logp, y


def fit_temperature(development, labels):
    """Fit only development NLL; callers cannot pass test data into this optimizer."""
    if not development or any(not r["id"].startswith("banking77-development-") for r in development):
        raise ValueError("temperature fitting requires development rows only")
    valid, logp, y = probability_matrix(development, labels)
    def objective(log_temperature):
        logits = logp / math.exp(log_temperature)
        logits -= logits.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(logits).sum(axis=1)) - logits[np.arange(len(y)), y]))
    # Deterministic bounded golden-section minimization in log T, T in [.05, 20].
    low, high = math.log(.05), math.log(20)
    ratio = (math.sqrt(5)-1)/2
    c, d = high-ratio*(high-low), low+ratio*(high-low)
    fc, fd = objective(c), objective(d)
    for _ in range(80):
        if fc < fd:
            high, d, fd = d, c, fc
            c = high-ratio*(high-low)
            fc = objective(c)
        else:
            low, c, fc = c, d, fd
            d = low+ratio*(high-low)
            fd = objective(d)
    optimum = min([math.log(.05), math.log(20), 0., (low+high)/2], key=objective)
    return {"temperature": math.exp(optimum), "development_probability_rows": len(valid),
            "development_nll_before": objective(0.), "development_nll_after": objective(optimum),
            "temperature_bounds": [.05, 20], "fit_split": "development"}


def apply_temperature(rows, labels, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    output = []
    for source in rows:
        row = dict(source)
        if not row.get("error") and row.get("probabilities"):
            logits = np.log(np.maximum([row["probabilities"][k] for k in labels], 1e-12)) / temperature
            probs = np.exp(logits - logits.max())
            row["probabilities"] = dict(zip(labels, (probs/probs.sum()).tolist()))
        output.append(row)
    return output


def calibrate(development, heldout, labels):
    # Split disjointness is checked before fitting; test labels cannot choose T.
    if {r["text_group"] for r in development} & {r["text_group"] for r in heldout}:
        raise ValueError("development/test text overlap")
    fit = fit_temperature(development, labels)
    fields = ("probability_rows", "multiclass_brier", "nll", "ece_10_equal_width")
    before = summarize(heldout, labels)
    after = summarize(apply_temperature(heldout, labels, fit["temperature"]), labels)
    return {**fit, "heldout_before": {k: before[k] for k in fields},
            "heldout_after": {k: after[k] for k in fields},
            "approximation": "Temperature scales log rounded stored probabilities, floored at 1e-12; original logits are unavailable. Predictions stay unchanged."}


def analyze(results_dir, data, base_target, gold_target):
    manifest = json.loads((data / "manifest.json").read_text())
    canonical = {}
    for split in ("development", "heldout"):
        path = data / (split + ".jsonl")
        if digest(path.read_bytes()) != manifest["split_sha256"][split]:
            raise ValueError("frozen split hash mismatch: " + split)
        cases = read_jsonl(path)
        canonical[split] = [{"id": c["id"], "expected": c["expected"]["intent"],
                             "text_group": digest(norm(c["state"]["request"]).encode())} for c in cases]
    labels = list(cases[0]["questions"]["intent"]["criteria"])
    output = {"schema": "public-study-analysis-v1", "runs": [], "skipped": [],
              "paired_comparison": None, "calibration": [],
              "selection": "Lexicographically latest completed, validated run per target and split; no score-based selection."}
    selected = {}
    for path in sorted(results_dir.glob("*_banking77_*.json")):
        try:
            source = json.loads(path.read_text())
            split = Path(source["cases"]).stem
            if split not in canonical:
                raise ValueError("not a development or heldout evaluation")
            rows = normalized_rows(source["rows"], labels)
            paired_rows(canonical[split], rows)
            expected_hash = manifest["split_sha256"][split]
            if source.get("cases_sha256") and source["cases_sha256"] != expected_hash:
                raise ValueError("result cases hash differs from frozen split")
            caveats = []
            if not source.get("cases_sha256"):
                caveats.append("Legacy result lacks cases_sha256; IDs, labels and text-group hashes match frozen data, but input question bytes cannot be established from this file.")
            if any(not r.get("model_resolved") for r in rows):
                caveats.append("Result lacks resolved model identity for one or more rows; requested target is not a checkpoint hash.")
            entry = {"input": str(path), "input_sha256": digest(path.read_bytes()), "target": source["target"],
                     "split": split, "summary": summarize(rows, labels), "caveats": caveats}
            output["runs"].append(entry)
            selected[(source["target"], split)] = (entry, rows)
        except (ValueError, KeyError, TypeError) as exc:
            output["skipped"].append({"input": str(path), "reason": str(exc)})
    base = selected.get((base_target, "heldout"))
    fine = selected.get((gold_target, "heldout"))
    if base and fine:
        output["paired_comparison"] = {"base_input": base[0]["input"], "fine_input": fine[0]["input"],
                                       **grouped_bootstrap(base[1], fine[1])}
    else:
        output["paired_comparison_pending"] = "Completed matching basewide and gold heldout results are required."
    jev = selected.get(("jev", "heldout"))
    if jev and fine:
        output["jev_comparison"] = {"base_input": jev[0]["input"], "fine_input": fine[0]["input"],
                                    **grouped_bootstrap(jev[1], fine[1])}
    for target, split in sorted(selected):
        if split != "heldout" or not target.startswith("laya:"):
            continue
        dev = selected.get((target, "development"))
        test = selected[(target, split)]
        if not dev:
            output["skipped"].append({"target": target, "reason": "No completed development run; calibration skipped."})
            continue
        try:
            output["calibration"].append({"target": target, "development_input": dev[0]["input"],
                                          "heldout_input": test[0]["input"], **calibrate(dev[1], test[1], labels)})
        except ValueError as exc:
            output["skipped"].append({"target": target, "reason": str(exc)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "bench" / "results")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--base-target", default="laya:laya-english-wide")
    parser.add_argument("--gold-target", default="laya:laya-banking77-gold-1k")
    args = parser.parse_args()
    output = args.out or args.results / "public_study_analysis.json"
    if output.resolve() in {p.resolve() for p in args.results.glob("*_banking77_*.json")}:
        parser.error("analysis output must not overwrite input results")
    report = analyze(args.results, args.data, args.base_target, args.gold_target)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
