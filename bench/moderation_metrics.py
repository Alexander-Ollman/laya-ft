"""CPU binary moderation scoring and paired request-cluster uncertainty.

All requests enter decision metrics. Failed unsafe requests are false negatives;
failed safe requests are reported separately, never counted as true negatives.
Average precision uses only successful rows with valid probability distributions.
"""
import math
from collections import defaultdict

import numpy as np

from analyze_public_study import normalized_rows, paired_rows
from evaluate_workflow_suite import detailed_metrics, wilson

LABELS = ["safe", "unsafe"]


def clean_rows(rows):
    if not rows:
        raise ValueError("empty moderation evaluation")
    if any(r.get("expected") not in LABELS for r in rows):
        raise ValueError("expected labels must be safe or unsafe")
    return normalized_rows(rows, LABELS)


def average_precision(rows):
    """Noninterpolated AP; equal scores enter together at one threshold.

    Undefined when no scored positive exists. This is average precision, not
    trapezoidal interpolation of the precision-recall curve.
    """
    rows = clean_rows(rows)
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["error"] or row.get("probabilities") is None:
            continue
        group = groups[row["probabilities"]["unsafe"]]
        group[0] += row["expected"] == "unsafe"
        group[1] += 1
    positives = sum(v[0] for v in groups.values())
    scored = sum(v[1] for v in groups.values())
    cumulative_tp = cumulative_n = weighted_precision = 0
    for score in sorted(groups, reverse=True):
        tp, count = groups[score]
        cumulative_tp += tp
        cumulative_n += count
        weighted_precision += tp * cumulative_tp / cumulative_n
    return {"value": weighted_precision / positives if positives else None,
            "scored_rows": scored, "excluded_rows": len(rows)-scored,
            "positive_support": positives,
            "method": "noninterpolated average precision; tied scores grouped; valid probability rows only"}


def moderation_metrics(rows):
    rows = clean_rows(rows)
    result = detailed_metrics(rows, LABELS, "unsafe")
    harmful = dict(result["safety_positive_class"])
    denominator = 2*harmful["tp"] + harmful["fp"] + harmful["fn"]
    harmful["f1"] = 2*harmful["tp"]/denominator if denominator else 0.
    harmful["negative_errors"] = sum(r["expected"] == "safe" and bool(r["error"]) for r in rows)
    result["harmful_class"] = harmful
    result["harmful_f1"] = harmful["f1"]
    result["harmful_f1_zero_division"] = 0
    result["average_precision"] = average_precision(rows)
    categories = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["expected"] != "unsafe":
            continue
        if row.get("categories") is not None and not isinstance(row["categories"], list):
            raise ValueError("categories must be a list when provided")
        for category in set(row.get("categories") or []):
            if not isinstance(category, str):
                raise ValueError("category names must be strings")
            categories[category][0] += 1
            categories[category][1] += bool(not row["error"] and row["prediction"] == "unsafe")
    result["per_category_positive_recall"] = {
        category: {"support": support, "tp": tp, "recall": tp/support,
                   "recall_wilson_95": wilson(tp, support)}
        for category, (support, tp) in sorted(categories.items())}
    result["category_metric_scope"] = "Positive unsafe rows only; categories do not define negatives or category false-positive rates."
    return result


def paired_bootstrap(reference, fine, resamples=5000, seed=7, confidence=.95):
    """Paired harmful F1 and accuracy differences, clustering normalized text."""
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    pairs = paired_rows(clean_rows(reference), clean_rows(fine))
    groups = defaultdict(lambda: np.zeros((2, 5), dtype=float))
    for left, right in pairs:
        for index, row in enumerate((left, right)):
            valid = not row["error"]
            positive = row["expected"] == "unsafe"
            predicted = valid and row["prediction"] == "unsafe"
            groups[left["text_group"]][index] += [1, valid and row["prediction"] == row["expected"],
                                                    positive and predicted, not positive and predicted,
                                                    positive and not predicted]
    counts = np.array([groups[k] for k in sorted(groups)])

    def effects(totals):
        accuracy = totals[..., 1] / totals[..., 0]
        den = 2*totals[..., 2] + totals[..., 3] + totals[..., 4]
        f1 = np.divide(2*totals[..., 2], den, out=np.zeros_like(den), where=den > 0)
        return accuracy[..., 1]-accuracy[..., 0], f1[..., 1]-f1[..., 0]

    observed_accuracy, observed_f1 = effects(counts.sum(axis=0))
    rng = np.random.default_rng(seed)
    accuracy_draws, f1_draws = [], []
    for start in range(0, resamples, 100):
        indices = rng.integers(0, len(groups), size=(min(100, resamples-start), len(groups)))
        accuracy, f1 = effects(counts[indices].sum(axis=1))
        accuracy_draws.extend(accuracy.tolist())
        f1_draws.extend(f1.tolist())
    tail = (1-confidence)/2
    return {"accuracy_delta_fine_minus_reference": float(observed_accuracy),
            "harmful_f1_delta_fine_minus_reference": float(observed_f1),
            "accuracy_interval": np.quantile(accuracy_draws, [tail, 1-tail]).tolist(),
            "harmful_f1_interval": np.quantile(f1_draws, [tail, 1-tail]).tolist(),
            "confidence": confidence, "n_rows": len(pairs), "n_text_groups": len(groups),
            "resamples": resamples, "seed": seed, "harmful_f1_zero_division": 0,
            "method": "paired text-cluster percentile bootstrap; retain all rows per sampled cluster",
            "limitation": "Conditional on these fitted checkpoints and test set; not training-seed robustness. Confidence level alone does not establish a prespecified multiple-comparison family."}
