"""CPU-only checks of pairing, cluster inference, and development-only fitting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from analyze_public_study import (analyze, calibrate, fit_temperature, grouped_bootstrap,
                                 normalized_rows, paired_rows)
from public_workflow import digest, make_case, norm


def row(i, prediction="a", expected="a", split="heldout", group=None, probability=.8):
    return {"id": f"banking77-{split}-{i}", "text_group": group or f"{split}-{i}",
            "prediction": prediction, "expected": expected, "error": None,
            "probabilities": {"a": probability, "b": 1-probability}, "latency_ms": 1,
            "usage": {"input_tokens": 100}}


class PublicAnalysisTests(unittest.TestCase):
    def test_pairing_rejects_ids_labels_groups_and_duplicates(self):
        base = [row(0), row(1)]
        for field, value in (("id", "different"), ("expected", "b"), ("text_group", "different")):
            changed = copy.deepcopy(base)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired_rows(base, changed)
        with self.assertRaises(ValueError):
            paired_rows(base, base + [base[0]])
        self.assertEqual(len(paired_rows(base, list(reversed(base)))), 2)

    def test_cluster_bootstrap_known_effect_and_duplicate_groups(self):
        base = [row(0, prediction="b", group="same"), row(1, prediction="b", group="same"),
                row(2, prediction="b", group="other")]
        fine = [dict(r, prediction="a") for r in base]
        result = grouped_bootstrap(base, fine)
        self.assertEqual(result["accuracy_delta_fine_minus_base"], 1)
        self.assertEqual(result["percentile_95"], [1, 1])
        self.assertEqual(result["n_rows"], 3)
        self.assertEqual(result["n_text_groups"], 2)
        self.assertEqual(result, grouped_bootstrap(base, fine))
        no_effect = grouped_bootstrap(base, base, resamples=100)
        self.assertEqual(no_effect["percentile_95"], [0, 0])

    def test_temperature_is_fit_from_development_only(self):
        dev = [row(i, expected="a" if i < 5 else "b", split="development", probability=.99)
               for i in range(10)]
        heldout = [row(i, probability=.99) for i in range(4)]
        first = calibrate(dev, heldout, ["a", "b"])
        changed_test_labels = [dict(r, expected="b") for r in heldout]
        second = calibrate(dev, changed_test_labels, ["a", "b"])
        self.assertEqual(first["temperature"], second["temperature"])
        self.assertGreater(first["temperature"], 1)
        self.assertLess(first["development_nll_after"], first["development_nll_before"])
        self.assertNotEqual(first["heldout_after"]["nll"], second["heldout_after"]["nll"])
        with self.assertRaises(ValueError):
            fit_temperature(heldout, ["a", "b"])
        with self.assertRaises(ValueError):
            calibrate(dev, [dict(heldout[0], text_group=dev[0]["text_group"])], ["a", "b"])

    def test_rescore_normalizes_legacy_maps_without_mutation(self):
        source = [row(0)]
        source[0]["probabilities"] = {"a": .8, "b": .19}
        before = copy.deepcopy(source)
        result = normalized_rows(source, ["a", "b"])
        self.assertEqual(source, before)
        self.assertAlmostEqual(sum(result[0]["probabilities"].values()), 1)

    def test_analysis_rescores_complete_runs_and_skips_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, results = root / "data", root / "results"
            data.mkdir()
            results.mkdir()
            hashes = {}
            for split in ("development", "heldout"):
                cases = [make_case({"text": f"{split} request {i}", "index": i, "category": "a"},
                                   split, ["a", "b"]) for i in range(2)]
                blob = "".join(json.dumps(c) + "\n" for c in cases).encode()
                (data / (split + ".jsonl")).write_bytes(blob)
                hashes[split] = digest(blob)
            (data / "manifest.json").write_text(json.dumps({"split_sha256": hashes}))
            records = [dict(row(i), text_group=digest(norm(f"heldout request {i}").encode())) for i in range(2)]
            source = {"target": "jev-direct", "cases": "/old/heldout.jsonl", "rows": records}
            path = results / "20260919_banking77_jev-direct.json"
            path.write_text(json.dumps(source))
            initial = path.read_bytes()
            (results / "20260920_banking77_partial.json").write_text(json.dumps(dict(source, rows=records[:1])))
            report = analyze(results, data, "base", "gold")
            self.assertEqual(path.read_bytes(), initial)
            self.assertEqual(len(report["runs"]), 1)
            self.assertEqual(len(report["skipped"]), 1)
            self.assertEqual(len(report["runs"][0]["caveats"]), 2)
            self.assertIsNone(report["runs"][0]["summary"]["cost_usd"])
            self.assertIsNone(report["paired_comparison"])


if __name__ == "__main__":
    unittest.main()
