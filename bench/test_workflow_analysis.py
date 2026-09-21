"""CPU checks for paired uncertainty, provenance rejection and failed responses."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_workflow_suite as analyzer
from analyze_workflow_suite import paired_bootstrap, validated_run
from evaluate_workflow_suite import detailed_metrics
from public_workflow import digest, norm


def row(identifier, expected, prediction, group="shared", error=None):
    return {"id": identifier, "expected": expected, "prediction": prediction,
            "text_group": group, "error": error, "probabilities": None, "latency_ms": 1, "usage": {}}


class WorkflowAnalysisTest(unittest.TestCase):
    def test_known_paired_cluster_difference(self):
        base = [row("a", "a", "a"), row("b", "b", "a")]
        fine = [row("a", "a", "a"), row("b", "b", "b")]
        result = paired_bootstrap(base, fine, ["a", "b"], resamples=50)
        self.assertAlmostEqual(result["macro_f1_delta_fine_minus_reference"], 2 / 3)
        self.assertEqual(result["accuracy_interval_95"], [.5, .5])
        self.assertEqual(result["macro_f1_confidence"], .99)
        for value in result["macro_f1_interval"]:
            self.assertAlmostEqual(value, 2 / 3)

    def test_permutation_invariance_and_determinism(self):
        base = [row(str(i), "a" if i % 2 else "b", "a", str(i // 2)) for i in range(20)]
        fine = [dict(r, prediction=r["expected"]) for r in base]
        left = paired_bootstrap(base, fine, ["a", "b"], resamples=80)
        right = paired_bootstrap(list(reversed(base)), fine[10:] + fine[:10], ["a", "b"], resamples=80)
        self.assertEqual(left, right)

    def test_fixed_class_set_and_failures(self):
        base = [row("a", "a", "a", error="timeout"), row("b", "b", "b", error="timeout")]
        fine = [row("a", "a", "a"), row("b", "b", "b")]
        result = paired_bootstrap(base, fine, ["a", "b", "absent"], resamples=20, primary=False)
        self.assertAlmostEqual(result["macro_f1_delta_fine_minus_reference"], 2 / 3)
        self.assertEqual(result["accuracy_delta_fine_minus_reference"], 1)
        self.assertEqual(result["macro_f1_confidence"], .95)

    def fixture(self):
        cases = [{"id": "one", "state": {"request": "Some text!"}, "expected": {"intent": "a"}}]
        manifest = {"slug": "sample", "labels": ["a", "b"], "split_sha256": {"heldout": "split-hash"}}
        source = {"dataset": "sample", "split": "heldout", "target": "jev", "manifest_sha256": "manifest-hash",
                  "cases_sha256": "split-hash", "summary": {"n": 1},
                  "rows": [row("one", "a", "a", digest(norm("Some text!").encode()))]}
        return source, manifest, cases

    def test_safety_false_positives_and_missing_prediction_denominators(self):
        rows = [row("tp", "attack", "attack"), row("miss", "attack", None, error="timeout"),
                row("fp", "benign", "attack"), row("tn", "benign", "benign"),
                row("unknown", "benign", None, error="invalid output")]
        result = detailed_metrics(rows, ["attack", "benign"], "attack")
        safety = result["safety_positive_class"]
        self.assertEqual((safety["support"], safety["tp"], safety["fp"], safety["fn"]), (2, 1, 1, 1))
        self.assertEqual(safety["precision"], .5)
        self.assertEqual(safety["recall"], .5)
        self.assertAlmostEqual(safety["false_positive_rate"], 1 / 3)
        self.assertEqual((result["n"], result["errors"], result["correct"]), (5, 2, 2))
        self.assertEqual(result["per_class"]["benign"]["fn"], 2)
        self.assertAlmostEqual(safety["recall_wilson_95"][0], .0945312057342307)
        self.assertAlmostEqual(safety["false_positive_wilson_95"][1], .7923403991979522)

    def test_safety_zero_support_has_null_rates_and_intervals(self):
        result = detailed_metrics([row("one", "benign", "benign")], ["attack", "benign"], "attack")
        safety = result["safety_positive_class"]
        self.assertEqual(safety["support"], 0)
        self.assertIsNone(safety["precision"])
        self.assertIsNone(safety["recall"])
        self.assertIsNone(safety["recall_wilson_95"])
        self.assertEqual(safety["false_positive_rate"], 0)
        self.assertAlmostEqual(safety["false_positive_wilson_95"][1], .7934506856227626)
        self.assertIsNone(result["per_class"]["benign"]["false_positive_rate"])
        self.assertIsNone(result["per_class"]["benign"]["false_positive_wilson_95"])

    def test_rejects_mismatched_evidence_and_counts(self):
        source, manifest, cases = self.fixture()
        self.assertEqual(len(validated_run(source, manifest, "manifest-hash", cases)), 1)
        for key, value in (("id", "wrong"), ("expected", "b"), ("text_group", "wrong")):
            bad = copy.deepcopy(source)
            bad["rows"][0][key] = value
            with self.assertRaises(ValueError):
                validated_run(bad, manifest, "manifest-hash", cases)
        for key in ("manifest_sha256", "cases_sha256"):
            bad = dict(source, **{key: "wrong"})
            with self.assertRaises(ValueError):
                validated_run(bad, manifest, "manifest-hash", cases)
        with self.assertRaises(ValueError):
            validated_run(dict(source, rows=[]), manifest, "manifest-hash", cases)
        with self.assertRaises(ValueError):
            validated_run(dict(source, summary={"n": 2}), manifest, "manifest-hash", cases)

    def test_latest_complete_selection_ignores_score(self):
        source, manifest, cases = self.fixture()
        cases[0]["questions"] = {"intent": {"criteria": {"a": None, "b": None}}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "data" / "sample"
            folder.mkdir(parents=True)
            results = root / "results"
            results.mkdir()
            blob = (json.dumps(cases[0]) + "\n").encode()
            (folder / "heldout.jsonl").write_bytes(blob)
            manifest.update(dataset="author/sample", train=10, development=2, test=1, protocol={},
                            split_sha256={"heldout": digest(blob)})
            manifest_blob = json.dumps(manifest).encode()
            (folder / "manifest.json").write_bytes(manifest_blob)
            source.update(manifest_sha256=digest(manifest_blob), cases_sha256=digest(blob))
            for stamp, prediction in (("20200101", "a"), ("20200102", "b")):
                source["rows"][0]["prediction"] = prediction
                (results / (stamp + "_suite_sample_heldout_jev.json")).write_text(json.dumps(source))
            bad = copy.deepcopy(source)
            bad["rows"] = []
            (results / "20200103_suite_sample_heldout_jev.json").write_text(json.dumps(bad))
            with patch.object(analyzer, "TASKS", ("sample",)):
                output = analyzer.analyze(results, root / "data", resamples=10)
            run = output["tasks"][0]["runs"]["jev"]
            self.assertEqual(run["summary"]["accuracy"], 0)
            self.assertIn("20200102", run["input"])
            self.assertEqual(len(output["skipped"]), 1)
            self.assertEqual(output["tasks"][0]["pending"], ["laya:base", "laya:fine"])

    def test_invalid_output_is_scored_as_failure_not_removed(self):
        source, manifest, cases = self.fixture()
        source["rows"][0]["prediction"] = "not a label"
        rows = validated_run(source, manifest, "manifest-hash", cases)
        result = detailed_metrics(rows, manifest["labels"])
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(result["macro_f1"], 0)
        self.assertEqual(result["per_class"]["a"]["fn"], 1)
        self.assertEqual(source["rows"][0]["prediction"], "not a label")


if __name__ == "__main__":
    unittest.main()
