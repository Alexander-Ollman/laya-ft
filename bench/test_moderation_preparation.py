"""CPU-only source-label, split-leakage and nested-budget regression tests."""
import csv
import gzip
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import prepare_moderation_study as prep
from public_workflow import read_jsonl


def source(i, prompt=None, response=None, prompt_label="unsafe", response_label="safe"):
    return {"id": str(i), "prompt": prompt or f"unique user request {i}",
            "response": response or f"unique assistant reply {i}",
            "prompt_label": prompt_label, "response_label": response_label,
            "violated_categories": "Violence, Hate", "prompt_label_source": "human",
            "response_label_source": "human"}


class ModerationPreparationTests(unittest.TestCase):
    def test_jsonl_preserves_unicode_separators_inside_strings(self):
        records = [{"text": "before" + separator + "after"} for separator in ("\u2028", "\u0085", "\u2029")]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unicode.jsonl"
            blob = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")
            path.write_bytes(blob)
            self.assertEqual(read_jsonl(path), records)
            self.assertEqual(path.read_bytes(), blob)

    def test_unknown_openai_labels_never_default_to_safe(self):
        rows = [{"prompt": "unknown"}, {"prompt": "partial zero", "S": 0},
                {"prompt": "known positive", "H": 1},
                {"prompt": "fully known negative", **dict.fromkeys(prep.OAI_POLICIES, 0)}]
        binary, categories, unknown = prep.openai_cases(rows)
        self.assertEqual(unknown, 2)
        self.assertEqual([c["expected"]["intent"] for c in binary], ["unsafe", "safe"])
        self.assertEqual(len(categories), 10)
        self.assertEqual(categories[0]["policy_category"], "S")
        self.assertEqual(categories[0]["expected"]["intent"], "safe")
        with self.assertRaises(ValueError):
            prep.openai_cases([{"prompt": "bad label", "S": -1}])

    def test_prompt_and_response_supervision_stay_separate(self):
        rows, excluded = prep.aegis([source(1, prompt="harmful request", response="I cannot help.")], "train")
        self.assertEqual(excluded, {})
        prompt, response = rows
        self.assertEqual(prompt["expected"]["intent"], "unsafe")
        self.assertEqual(response["expected"]["intent"], "safe")
        self.assertNotIn("[ASSISTANT]", prompt["state"]["request"])
        self.assertIn("[ASSISTANT]", response["state"]["request"])
        self.assertEqual(prompt["questions"]["intent"]["instructions"], prep.PROMPT)
        self.assertEqual(response["questions"]["intent"]["instructions"], prep.RESPONSE)
        self.assertEqual(prompt["categories"], ["Violence", "Hate"])
        self.assertEqual(response["categories"], [])
        self.assertEqual(prompt["text_group"], response["text_group"])

    def test_leakage_checks_each_component_not_only_full_transcript(self):
        heldout = prep.case("benchmark", 1, "held user", "unsafe", response="held response")
        cases = [prep.case("aegis_prompt", 1, "HELD USER!!!", "unsafe"),
                 prep.case("aegis_response", 2, "new user", "safe", response="Held response!"),
                 prep.case("aegis_prompt", 3, "Held response!", "safe"),
                 prep.case("aegis_response", 4, "novel user", "safe", response="novel response")]
        clean, excluded = prep.clean_training(cases, prep.component_hashes([heldout]))
        self.assertEqual([c["id"] for c in clean], ["aegis_response-4"])
        self.assertEqual(excluded["heldout_or_development_component_overlap"], 3)

    def test_conflicts_removed_within_task_and_duplicates_deduplicated(self):
        cases = [prep.case("aegis_prompt", 1, "ambiguous", "safe"),
                 prep.case("aegis_prompt", 2, "AMBIGUOUS!", "unsafe"),
                 prep.case("aegis_prompt", 3, "ordinary", "safe"),
                 prep.case("aegis_prompt", 4, "Ordinary!", "safe"),
                 prep.case("aegis_response", 5, "ambiguous", "safe", response="refusal")]
        clean, excluded = prep.clean_training(cases, set())
        self.assertEqual(excluded["conflicting_labels"], 2)
        self.assertEqual(excluded["duplicates"], 1)
        self.assertEqual({c["id"] for c in clean}, {"aegis_prompt-3", "aegis_response-5"})

    def test_nested_budgets_retain_tasks_and_source_class_prevalence(self):
        cases = []
        for task in ("aegis_prompt", "aegis_response"):
            for i in range(200):
                cases.append(prep.case(task, i, f"request {i}", "safe" if i < 150 else "unsafe",
                                       response="answer" if task.endswith("response") else None))
        large = prep.sample(cases, 160, 9)
        small = prep.sample(large, 40, 10)
        self.assertEqual(len(large), 160)
        self.assertEqual(len(small), 40)
        self.assertTrue({c["id"] for c in small} <= {c["id"] for c in large})
        self.assertEqual(Counter(c["category"] for c in small), {"aegis_prompt": 20, "aegis_response": 20})
        self.assertEqual(Counter(c["expected"]["intent"] for c in small), {"safe": 30, "unsafe": 10})
        self.assertEqual(small, prep.sample(large, 40, 10))
        with self.assertRaises(ValueError):
            prep.sample(cases, 1000, 7)
        for invalid_size in (0, -2, 41):
            with self.subTest(size=invalid_size), self.assertRaises(ValueError):
                prep.sample(cases, invalid_size, 7)

    def test_complete_source_exclusions_include_unscored_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "train/moderation-study"
            raw = data / "raw"
            for name in ("aegis", "toxicchat", "beavertails", "wildguard", "openai", "xstest"):
                (raw / name).mkdir(parents=True)
            suite = root / "train/workflow-suite"
            suite.mkdir()
            (suite / "prior_holdout_hashes.json").write_text("[]")
            forbidden = ["unscored aegis prompt", "unscored aegis response", "unscored wild prompt",
                         "unscored wild response", "unannotated toxic input", "unknown openai input",
                         "unscored development prompt", "unscored development response"]
            test = [source("test"), source("unscored", forbidden[0], forbidden[1], None, None)]
            dev = [source(f"dev{i}") for i in range(6)] + [source("unscoreddev", forbidden[6], forbidden[7], None, None)]
            train = [source(f"train{i}") for i in range(12)] + [source(f"leak{i}", text) for i, text in enumerate(forbidden)]
            for filename, rows in (("test.json", test), ("validation.json", dev), ("train.json", train),
                                   ("refusals_validation.json", []), ("refusals_train.json", [])):
                (raw / "aegis" / filename).write_text(json.dumps(rows))
            with (raw / "toxicchat/toxic-chat_annotation_test.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=["user_input", "toxicity", "human_annotation"])
                writer.writeheader()
                writer.writerows([{"user_input": forbidden[4], "toxicity": "0", "human_annotation": "false"},
                                  {"user_input": "scored toxic test", "toxicity": "1", "human_annotation": "true"}])
            beaver = {"prompt": "beaver test request", "response": "beaver test response", "is_safe": True, "category": {}}
            (raw / "beavertails/test.jsonl.gz").write_bytes(gzip.compress((json.dumps(beaver)+"\n").encode()))
            wild = [{"prompt": forbidden[2], "response": forbidden[3], "prompt_harm_label": None, "response_harm_label": None},
                    {"prompt": "scored wild prompt", "response": "scored wild answer", "prompt_harm_label": "harmful", "response_harm_label": "unharmful"}]
            (raw / "wildguard/test.parquet").touch()
            oai = [{"prompt": forbidden[5]}, {"prompt": "scored openai", **dict.fromkeys(prep.OAI_POLICIES, 0)}]
            (raw / "openai/samples-1680.jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r)+"\n" for r in oai).encode()))
            (raw / "xstest/xstest_prompts.csv").write_text("id,prompt,type\n1,xstest unique,benign\n")
            actual_sample = prep.sample
            captured = []
            def scaled_sample(cases, n, seed):
                if n == 5000:
                    captured.extend(cases)
                return actual_sample(cases, {400: 4, 5000: 8, 1000: 4}[n], seed)
            with patch.object(prep, "ROOT", root), patch.object(prep, "DATA", data), patch.object(prep, "RAW", raw), \
                    patch.object(prep, "sample", side_effect=scaled_sample), patch.object(prep.pq, "read_table") as parquet, \
                    redirect_stdout(io.StringIO()):
                parquet.return_value.to_pylist.return_value = wild
                prep.prepare()
            leaked = {component for c in captured for component in c["components"]} & set(forbidden)
            self.assertEqual(leaked, set(), "unscored source content leaked into the training candidate pool")


if __name__ == "__main__":
    unittest.main()
