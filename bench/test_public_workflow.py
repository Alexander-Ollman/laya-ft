"""Publication-critical split and scoring checks; no GPU or network."""
import unittest
import json
import tempfile
from pathlib import Path
from public_workflow import split_rows, norm, summarize, make_case, parse_answer, cost_summary, validate_case_file, digest


class PublicProtocolTests(unittest.TestCase):
    def test_malformed_answers_and_probability_payloads(self):
        for answers in (None, [], {"intent": None}, {"intent": "a"}, {"intent": {"choice": "c"}}):
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                parse_answer(answers, ["a", "b"])
        for probs in ({"a": 1}, {"a": float("nan"), "b": 0},
                      {"a": True, "b": False}, {"a": .3, "b": .3}, None):
            prediction, valid = parse_answer({"intent": {"choice": "a", "probabilities": probs}}, ["a", "b"])
            self.assertEqual(prediction, "a")
            self.assertIsNone(valid)
        _, probs = parse_answer({"intent": {"choice": "a", "probabilities": {"a": .8, "b": .19}}}, ["a", "b"])
        self.assertAlmostEqual(sum(probs.values()), 1)

    def test_unknown_charges_are_not_zero(self):
        rows = [{"usage": {"cost_usd": .01}}, {"usage": {"input_tokens": 1000}}, {"usage": None}]
        cost = cost_summary(rows)
        self.assertIsNone(cost["cost_usd"])
        self.assertEqual(cost["reported_cost_usd"], .01)
        self.assertEqual(cost["cost_unknown_rows"], 2)
        self.assertEqual(cost_summary([{"usage": {"cost_usd": 0}}])["cost_usd"], 0)
        self.assertIsNone(cost_summary([{"usage": {"cost_usd": float("nan")}}])["cost_usd"])

    def test_modified_evaluation_or_frozen_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            frozen = data / "heldout.jsonl"
            frozen.write_bytes(b"original\n")
            (data / "manifest.json").write_text(json.dumps({"split_sha256": {"heldout": digest(frozen.read_bytes())}}))
            self.assertEqual(validate_case_file(frozen, data), digest(b"original\n"))
            foreign = data / "foreign.jsonl"
            foreign.write_bytes(b"other\n")
            with self.assertRaises(ValueError):
                validate_case_file(foreign, data)
            frozen.write_bytes(b"changed\n")
            with self.assertRaises(ValueError):
                validate_case_file(frozen, data)

    def test_ece_uses_confidence_of_returned_decision(self):
        row = {"expected": "b", "prediction": "b", "error": None,
               "probabilities": {"a": .8, "b": .2}, "latency_ms": 10}
        self.assertAlmostEqual(summarize([row], ["a", "b"])["ece_10_equal_width"], .8)

    def test_split_blocks_normalized_overlap_and_conflicting_labels(self):
        train = [{"text": f"request {label} {i}", "category": label, "index": i}
                 for label in ("a", "b") for i in range(8)]
        train += [{"text": "TEST!!!", "category": "a", "index": 99},
                  {"text": "ambiguous", "category": "a", "index": 100},
                  {"text": "Ambiguous!", "category": "b", "index": 101}]
        test = [{"text": "test", "category": "a"}]
        fit, dev, excluded = split_rows(train, test, train_per_class=3, dev_per_class=2)
        self.assertEqual((len(fit), len(dev)), (6, 4))
        self.assertFalse({norm(r['text']) for r in fit} & {norm(r['text']) for r in dev})
        self.assertEqual(excluded['test_overlap'], 1)
        self.assertEqual(excluded['conflicting_labels'], 2)
        self.assertEqual((fit, dev, excluded), split_rows(train, test, train_per_class=3, dev_per_class=2))

    def test_full_multiclass_brier_and_errors_in_denominator(self):
        rows = [{"expected": "a", "prediction": "a", "error": None,
                 "probabilities": {"a": .8, "b": .2}, "latency_ms": 10},
                {"expected": "b", "prediction": None, "error": "timeout",
                 "probabilities": None, "latency_ms": 100}]
        result = summarize(rows, ["a", "b"])
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['multiclass_brier'], .08)
        self.assertEqual(result['errors'], 1)
        self.assertLess(result['wilson_95'][0], .5)
        self.assertGreater(result['wilson_95'][1], .5)

    def test_gold_does_not_change_model_input(self):
        row = {"text": "help me", "index": 1, "category": "a"}
        first = make_case(row, "heldout", ["a", "b"])
        second = make_case(dict(row, category="b"), "heldout", ["a", "b"])
        self.assertEqual(first['state'], second['state'])
        self.assertEqual(first['questions'], second['questions'])
        self.assertNotEqual(first['expected'], second['expected'])


if __name__ == '__main__':
    unittest.main()
