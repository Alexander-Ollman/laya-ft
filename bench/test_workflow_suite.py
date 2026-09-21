"""CPU-only checks of frozen workflow sampling and label separation."""
import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import prepare_workflow_suite as suite


def row(text, category="a", index="0"):
    return {"text": text, "category": category, "index": index}


class WorkflowPreparationTest(unittest.TestCase):
    def test_proportional_sample_is_exact_and_repeatable(self):
        rows = [row(str(i), "a" if i < 80 else "b", str(i)) for i in range(100)]
        sample = suite.stratified(rows, 20)
        self.assertEqual(Counter(r["category"] for r in sample), {"a": 16, "b": 4})
        self.assertEqual(sample, suite.stratified(rows, 20))
        self.assertNotEqual(sample, suite.stratified(rows, 20, seed=8))
        self.assertEqual(len({r["index"] for r in sample}), 20)
        self.assertEqual(rows[0]["index"], "0")

    def test_rare_classes_survive_and_full_small_set_is_preserved(self):
        rows = [row(str(i), "a", str(i)) for i in range(98)] + [row("rare b", "b"), row("rare c", "c")]
        self.assertEqual(Counter(r["category"] for r in suite.stratified(rows, 3)), {"a": 1, "b": 1, "c": 1})
        with self.assertRaises(ValueError):
            suite.stratified(rows, 2)
        self.assertEqual(suite.stratified(rows, 1000), rows)

    def test_normalized_duplicates_conflicts_and_forbidden_text(self):
        rows = [row("Keep me!"), row("keep me"), row("Conflict", "a"), row("CONFLICT!", "b"),
                row("Official TEST!"), row("!!!"), row("different", "b")]
        cleaned, exclusions = suite.clean(rows, {suite.norm("official test")})
        self.assertEqual({suite.norm(r["text"]) for r in cleaned}, {"keep me", "different"})
        self.assertEqual(exclusions, {"duplicates": 1, "conflicting_labels": 2,
                                     "heldout_or_development_overlap": 1, "empty_normalized_text": 1})

    def test_model_input_does_not_change_with_gold_label(self):
        config = {"repo": "author/data", "revision": "pinned", "instruction": "Choose the matching class."}
        left = suite.make_case("demo", row("request", "a"), "heldout", config, ["a", "b"])
        right = suite.make_case("demo", row("request", "b"), "heldout", config, ["a", "b"])
        for key in ("id", "state", "questions"):
            self.assertEqual(left[key], right[key])
        self.assertNotEqual(left["expected"], right["expected"])
        self.assertEqual(list(left["questions"]["intent"]["criteria"]), ["a", "b"])

    def test_entire_official_holdouts_excluded_before_sampling(self):
        test_rows = [row("official test " + str(i), str(i % 2), "test-" + str(i)) for i in range(1100)]
        sampled = suite.stratified(test_rows, 1000)
        sampled_texts = {r["text"] for r in sampled}
        omitted_test = next(r for r in test_rows if r["text"] not in sampled_texts)
        valid_rows = [row("validation " + str(i), str(i % 2), "validation-" + str(i)) for i in range(220)]
        sampled_valid = {r["text"] for r in suite.stratified(valid_rows, 200, seed=8)}
        omitted_valid = next(r for r in valid_rows if r["text"] not in sampled_valid)
        training = [row("training " + str(i), str(i % 2)) for i in range(300)] + [omitted_test, omitted_valid]
        datasets = {"train": training, "validation": valid_rows, "test": test_rows}
        config = {"repo": "fixture/source", "revision": "pinned", "labels": ["a", "b"], "license": "fixture",
                  "instruction": "Classify.", "text": "text", "label": "label",
                  "files": {s: s + ".tsv" for s in datasets}}

        class Response:
            content = b"Fixture card"

            def raise_for_status(self):
                pass

        class Client:
            def get(self, url):
                return Response()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "fixture" / "raw"
            raw.mkdir(parents=True)
            for split, rows in datasets.items():
                buffer = io.StringIO()
                writer = csv.writer(buffer, delimiter="\t")
                writer.writerow(["text", "label"])
                writer.writerows((r["text"], r["category"]) for r in rows)
                (raw / (split + ".tsv")).write_text(buffer.getvalue())
            with patch.object(suite, "SUITE", root), patch.object(suite, "prior_holdouts", return_value=set()), \
                    patch.object(suite.httpx, "Client", return_value=Client()):
                suite.prepare_one("fixture", config)
            manifest = json.loads((root / "fixture" / "manifest.json").read_text())
            self.assertEqual(manifest["exclusions"]["train"]["heldout_or_development_overlap"], 2)
            self.assertEqual(manifest["test"], 1000)
            fit = suite.read_jsonl(root / "fixture" / "train.jsonl")
            self.assertEqual(len(fit), 300)
            self.assertFalse({c["state"]["request"] for c in fit} & {omitted_test["text"], omitted_valid["text"]})


if __name__ == "__main__":
    unittest.main()
