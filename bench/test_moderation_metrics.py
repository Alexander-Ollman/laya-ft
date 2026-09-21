"""CPU tests of safety failures, tied AP, categories and paired inference."""
import copy
import unittest

from moderation_metrics import average_precision, moderation_metrics, paired_bootstrap


def row(identifier, expected, prediction, score=None, error=None, group=None, categories=None):
    return {"id": identifier, "expected": expected, "prediction": prediction,
            "probabilities": None if score is None else {"safe": 1-score, "unsafe": score},
            "error": error, "text_group": group or identifier, "latency_ms": 1, "usage": {},
            "categories": categories}


class ModerationMetricsTest(unittest.TestCase):
    def test_failures_remain_in_denominators_and_safe_errors_not_tn(self):
        rows = [row("tp", "unsafe", "unsafe", .9, categories=["violence", "violence"]),
                row("miss", "unsafe", "unsafe", .9, error="timeout", categories=["violence"]),
                row("fp", "safe", "unsafe", .8, categories=["violence"]),
                row("tn", "safe", "safe", .1),
                row("failed", "safe", "safe", .1, error="timeout")]
        original = copy.deepcopy(rows)
        result = moderation_metrics(rows)
        self.assertEqual(rows, original)
        harmful = result["harmful_class"]
        self.assertEqual([harmful[k] for k in ("tp", "fp", "fn", "tn", "negative_errors")], [1, 1, 1, 1, 1])
        self.assertEqual(harmful["f1"], .5)
        self.assertEqual(harmful["precision"], .5)
        self.assertEqual(harmful["recall"], .5)
        self.assertAlmostEqual(harmful["false_positive_rate"], 1/3)
        self.assertEqual(result["accuracy"], .4)
        self.assertEqual(result["errors"], 2)
        self.assertEqual(result["average_precision"]["excluded_rows"], 2)
        category = result["per_category_positive_recall"]["violence"]
        self.assertEqual((category["support"], category["tp"], category["recall"]), (2, 1, .5))
        self.assertNotIn("false_positive_rate", category)

    def test_average_precision_ties_are_one_threshold(self):
        rows = [row("one", "unsafe", "unsafe", .9), row("two", "safe", "unsafe", .9),
                row("three", "unsafe", "safe", .1)]
        # AP = (1/2 positive mass)*precision(2-row tie) + (1/2)*precision(all).
        expected = .5*.5 + .5*(2/3)
        self.assertAlmostEqual(average_precision(rows)["value"], expected)
        self.assertEqual(average_precision(rows), average_precision(list(reversed(rows))))
        same_score = [dict(r, probabilities={"safe": .5, "unsafe": .5}) for r in rows]
        self.assertAlmostEqual(average_precision(same_score)["value"], 2/3)

    def test_no_positive_ap_is_undefined_and_invalid_probability_excluded(self):
        self.assertIsNone(average_precision([row("safe", "safe", "safe", .1)])["value"])
        bad = row("bad", "unsafe", "unsafe", .8)
        bad["probabilities"] = {"unsafe": .8}
        result = moderation_metrics([bad])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["average_precision"]["scored_rows"], 0)
        self.assertIsNone(result["average_precision"]["value"])
        self.assertEqual(moderation_metrics([row("safe", "safe", "safe")])["harmful_f1"], 0)

    def test_grouped_bootstrap_known_effect_and_family_confidence(self):
        base = [row("a", "unsafe", "safe", group="same"), row("b", "safe", "safe", group="same")]
        fine = [dict(r, prediction=r["expected"]) for r in base]
        for confidence in (.95, .9875):
            result = paired_bootstrap(base, fine, confidence=confidence, resamples=100)
            self.assertEqual(result["accuracy_delta_fine_minus_reference"], .5)
            self.assertEqual(result["accuracy_interval"], [.5, .5])
            self.assertEqual(result["harmful_f1_interval"], [1, 1])
            self.assertEqual(result["confidence"], confidence)
            self.assertEqual(result["n_text_groups"], 1)

    def test_bootstrap_pair_validation_and_permutation_invariance(self):
        base = [row(str(i), "unsafe" if i%2 else "safe", "safe", group=str(i//2)) for i in range(12)]
        fine = [dict(r, prediction=r["expected"]) for r in base]
        result = paired_bootstrap(base, fine, resamples=80)
        self.assertEqual(result, paired_bootstrap(list(reversed(base)), fine[5:]+fine[:5], resamples=80))
        for key, value in (("id", "bad"), ("text_group", "bad"), ("expected", "unsafe")):
            bad = copy.deepcopy(fine)
            bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                paired_bootstrap(base, bad, resamples=10)
        with self.assertRaises(ValueError):
            paired_bootstrap(base, fine+[fine[0]], resamples=10)
        with self.assertRaises(ValueError):
            paired_bootstrap(base, fine, confidence=1)

    def test_error_predictions_cannot_score_as_correct(self):
        base = [row("a", "unsafe", "unsafe", error="timeout", group="same"),
                row("b", "safe", "safe", error="timeout", group="same")]
        fine = [dict(r, error=None) for r in base]
        result = paired_bootstrap(base, fine, resamples=20)
        self.assertEqual(result["accuracy_interval"], [1, 1])
        self.assertEqual(result["harmful_f1_interval"], [1, 1])


if __name__ == "__main__":
    unittest.main()
