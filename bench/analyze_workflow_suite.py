"""Validate and rescore frozen workflow results; paired CPU-only uncertainty."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_public_study import normalized_rows, paired_rows
from evaluate_workflow_suite import detailed_metrics
from public_workflow import ROOT, digest, norm, read_jsonl

TASKS = ("injection", "sms", "emotion", "counterfactual", "massive")
TARGETS = ("laya:base", "laya:fine", "jev")


def validated_run(source, manifest, manifest_hash, cases):
    """Reject incomplete or mismatched evidence before looking at performance."""
    if source.get("dataset") != manifest["slug"] or source.get("split") != "heldout":
        raise ValueError("dataset or split mismatch")
    if source.get("target") not in TARGETS:
        raise ValueError("unexpected target")
    if source.get("manifest_sha256") != manifest_hash:
        raise ValueError("result manifest hash mismatch")
    if source.get("cases_sha256") != manifest["split_sha256"]["heldout"]:
        raise ValueError("result split hash mismatch")
    rows = source["rows"]
    if len(rows) != len(cases) or source.get("summary", {}).get("n") != len(cases):
        raise ValueError("incomplete request count")
    canonical = [{"id": c["id"], "expected": c["expected"]["intent"],
                  "text_group": digest(norm(c["state"]["request"]).encode())} for c in cases]
    paired_rows(canonical, rows)
    return normalized_rows(rows, manifest["labels"])


def paired_bootstrap(base, fine, labels, resamples=5000, seed=7, primary=True):
    """Resample complete text groups; always retain the fixed class label set."""
    if resamples < 1 or not labels or len(set(labels)) != len(labels):
        raise ValueError("positive resamples and unique labels required")
    pairs = paired_rows(normalized_rows(base, labels), normalized_rows(fine, labels))
    groups = defaultdict(list)
    truth, predictions = [], [[], []]
    mapping = {label: i for i, label in enumerate(labels)}
    for i, pair in enumerate(pairs):
        truth.append(mapping[pair[0]["expected"]])
        groups[pair[0]["text_group"]].append(i)
        for model, row in enumerate(pair):
            predictions[model].append(mapping.get(row["prediction"], -1) if not row["error"] else -1)
    truth = np.asarray(truth)
    predictions = [np.asarray(p) for p in predictions]
    clusters = [np.asarray(groups[key], dtype=int) for key in sorted(groups)]

    def metrics(indices, prediction):
        y, p = truth[indices], prediction[indices]
        support = np.bincount(y, minlength=len(labels))
        predicted = np.bincount(p[p >= 0], minlength=len(labels))
        true_positive = np.bincount(y[p == y], minlength=len(labels))
        denominator = support + predicted
        f1 = np.divide(2 * true_positive, denominator, out=np.zeros(len(labels)), where=denominator != 0)
        return np.array([f1.mean(), (p == y).mean()])

    def delta(indices):
        return metrics(indices, predictions[1]) - metrics(indices, predictions[0])

    observed = delta(np.arange(len(pairs)))
    rng = np.random.default_rng(seed)
    draws = np.empty((resamples, 2))
    singletons = all(len(cluster) == 1 for cluster in clusters)
    flat = np.concatenate(clusters)
    for i in range(resamples):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        indices = flat[chosen] if singletons else np.concatenate([clusters[k] for k in chosen])
        draws[i] = delta(indices)
    f1_quantiles = [.005, .995] if primary else [.025, .975]
    return {"macro_f1_delta_fine_minus_reference": float(observed[0]),
            "macro_f1_interval": np.quantile(draws[:, 0], f1_quantiles).tolist(),
            "macro_f1_confidence": .99 if primary else .95,
            "accuracy_delta_fine_minus_reference": float(observed[1]),
            "accuracy_interval_95": np.quantile(draws[:, 1], [.025, .975]).tolist(),
            "n_rows": len(pairs), "n_text_groups": len(clusters), "classes": len(labels),
            "resamples": resamples, "seed": seed,
            "method": "Paired normalized-text group percentile bootstrap; fixed class set, absent-class F1=0.",
            "multiplicity": "99% macro-F1 interval: Bonferroni family of five primary contrasts" if primary else "Descriptive secondary 95% intervals; no multiplicity correction",
            "limitation": "Conditional on these checkpoints and test examples; does not measure training-seed variation."}


def analyze(results, data, resamples=5000):
    output = {"schema": "workflow-suite-analysis-v1", "tasks": [], "skipped": [],
              "selection": "Lexicographically latest completed validated run per task and target; no score-based selection."}
    for slug in TASKS:
        folder = data / slug
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            output["tasks"].append({"slug": slug, "runs": {}, "basecomparison": None,
                                    "jevcomparison": None, "pending": ["Frozen dataset manifest unavailable"]})
            continue
        manifest = json.loads(manifest_path.read_text())
        manifest_hash = digest(manifest_path.read_bytes())
        if manifest["slug"] != slug:
            raise ValueError("manifest slug mismatch: " + slug)
        for split, expected in manifest["split_sha256"].items():
            if digest((folder / (split + ".jsonl")).read_bytes()) != expected:
                raise ValueError("frozen split hash mismatch: " + slug + "/" + split)
        cases = read_jsonl(folder / "heldout.jsonl")
        if len(cases) != manifest["test"] or not cases:
            raise ValueError("frozen test count mismatch: " + slug)
        if any(list(c["questions"]["intent"]["criteria"]) != manifest["labels"] or
               c["expected"]["intent"] not in manifest["labels"] for c in cases):
            raise ValueError("frozen labels mismatch: " + slug)
        task = {"slug": slug, "dataset": manifest["dataset"], "manifest": str(manifest_path),
                "manifest_sha256": manifest_hash, "cases": str(folder / "heldout.jsonl"),
                "cases_sha256": manifest["split_sha256"]["heldout"],
                "source": "https://huggingface.co/datasets/" + manifest["dataset"],
                "train": manifest["train"], "development": manifest["development"], "test": manifest["test"],
                "labels": manifest["labels"], "runs": {}, "basecomparison": None,
                "jevcomparison": None, "pending": []}
        selected = {}
        for path in sorted(results.glob("*_suite_" + slug + "_heldout_*.json")):
            try:
                blob = path.read_bytes()
                source = json.loads(blob)
                rows = validated_run(source, manifest, manifest_hash, cases)
                entry = {"input": str(path), "input_sha256": digest(blob), "target": source["target"],
                         "summary": detailed_metrics(rows, manifest["labels"], manifest["protocol"].get("positive_class")),
                         "model_resolved": sorted({r["model_resolved"] for r in rows if r.get("model_resolved")}),
                         "missing_model_identity_rows": sum(not r.get("model_resolved") for r in rows)}
                selected[source["target"]] = (entry, rows)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                output["skipped"].append({"input": str(path), "reason": str(exc)})
        task["runs"] = {target: selected[target][0] for target in TARGETS if target in selected}
        task["pending"] = [target for target in TARGETS if target not in selected]
        for reference, key, primary in (("laya:base", "basecomparison", True), ("jev", "jevcomparison", False)):
            if reference in selected and "laya:fine" in selected:
                task[key] = {"reference_target": reference, "reference_input": selected[reference][0]["input"],
                             "fine_input": selected["laya:fine"][0]["input"],
                             **paired_bootstrap(selected[reference][1], selected["laya:fine"][1],
                                                manifest["labels"], resamples=resamples, primary=primary)}
        output["tasks"].append(task)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "bench/results/workflow-suite")
    parser.add_argument("--data", type=Path, default=ROOT / "train/workflow-suite")
    args = parser.parse_args()
    output = analyze(args.results, args.data)
    args.results.mkdir(parents=True, exist_ok=True)
    path = args.results / "analysis.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
