"""CPU evidence-integrity and category scoring tests for moderation analysis."""
import copy
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

from analyze_moderation import analyze, validated_run, summarize_run
from public_workflow import digest


def fixture(root, slug="aegis_prompt"):
    data, results = root/"data", root/"results"
    folder = data/slug
    folder.mkdir(parents=True)
    results.mkdir()
    cases = [{"id": str(i), "text_group": "group"+str(i), "state": {"request": "request"+str(i)},
              "expected": {"intent": label}, "categories": ["violence"] if label == "unsafe" else [],
              "policy_category": "S", "questions": {"intent": {"criteria": {"safe": None, "unsafe": None}}}}
             for i, label in enumerate(("safe", "unsafe"))]
    blob = "".join(json.dumps(c)+"\n" for c in cases).encode()
    (folder/"heldout.jsonl").write_bytes(blob)
    manifest = {"slug": slug, "labels": ["safe", "unsafe"], "test": 2,
                "split_sha256": {"heldout": digest(blob)}}
    manifest_blob = json.dumps(manifest).encode()
    (folder/"manifest.json").write_bytes(manifest_blob)
    study_blob = json.dumps({"test_counts": {slug: 2}, "raw_sha256": {"raw/unbundled.json": "0"*64}}).encode()
    (data/"study_manifest.json").write_bytes(study_blob)
    (results/"preflight.json").write_text(json.dumps({"train/moderation-study/"+slug+"/heldout.jsonl":
        {"n": 2, "truncated_ids": ["1"], "input_truncated": 1}}))
    identity = "sha256:"+"a"*64
    rows = [{"id": c["id"], "text_group": c["text_group"], "expected": c["expected"]["intent"],
             "prediction": "safe", "probabilities": {"safe": .8, "unsafe": .2}, "error": None,
             "categories": c["categories"], "policy_category": c["policy_category"],
             "input_truncated": c["id"] == "1", "latency_ms": 1, "usage": {}, "model_resolved": identity}
            for c in cases]
    source = {"dataset": slug, "target": "base", "summary": {"n": 2}, "rows": rows,
              "cases_sha256": digest(blob), "manifest_sha256": digest(manifest_blob),
              "study_sha256": digest(study_blob), "checkpoint": str(root/"unbundled-checkpoint"),
              "weights_identity": identity, "config_sha256": "b"*64, "pad_to_multiple": 128}
    return data, results, cases, manifest, source


class ModerationAnalysisTests(unittest.TestCase):
    def test_standalone_rescorer_import_needs_no_model_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ("analyze_moderation", "moderation_metrics", "analyze_public_study",
                         "evaluate_workflow_suite", "public_workflow", "telemetry"):
                (folder/(name+".py")).write_bytes((Path(__file__).parent/(name+".py")).read_bytes())
            subprocess.run([sys.executable, "-c", "import sys; import analyze_moderation; "
                            "assert 'torch' not in sys.modules; assert 'laya' not in sys.modules"],
                           cwd=folder, check=True, capture_output=True)

    def test_available_finetune_config_must_record_training_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, cases, manifest, source = fixture(Path(temporary))
            checkpoint = Path(source["checkpoint"])
            checkpoint.mkdir()
            weights = b"fixture weights"
            (checkpoint/"model.safetensors").write_bytes(weights)
            source.update(target="fine1000", weights_identity="sha256:"+digest(weights))
            for row in source["rows"]:
                row["model_resolved"] = source["weights_identity"]
            for padding in (0, 128):
                config = json.dumps({"max_len": 2048, "head_max_len": 256,
                                     "finetune": {"pad_to_multiple": padding}}).encode()
                (checkpoint/"rl_agent_config.json").write_bytes(config)
                source["config_sha256"] = digest(config)
                def validate():
                    return validated_run(source, "fine1000", manifest, source["manifest_sha256"], source["study_sha256"], cases, {"1"}, {})
                if padding == 0:
                    with self.assertRaisesRegex(ValueError, "fine-tune padding"):
                        validate()
                else:
                    self.assertEqual(len(validate()), 2)

    def test_missing_or_different_padding_cannot_enter_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, cases, manifest, source = fixture(Path(temporary))
            for value in (None, 0, 64, 256, "128", 128., True):
                bad = dict(source, pad_to_multiple=value)
                with self.subTest(padding=value), self.assertRaisesRegex(ValueError, "padding"):
                    validated_run(bad, "base", manifest, source["manifest_sha256"], source["study_sha256"], cases, {"1"}, {})

    def test_tampered_rows_metadata_counts_and_identities_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, cases, manifest, source = fixture(Path(temporary))
            def validate(value):
                return validated_run(value, "base", manifest, source["manifest_sha256"], source["study_sha256"], cases, {"1"}, {})
            self.assertEqual(len(validate(source)), 2)
            for key, value in (("id", "wrong"), ("expected", "unsafe"), ("text_group", "wrong"),
                               ("categories", ["wrong"]), ("policy_category", "H"),
                               ("model_resolved", None), ("input_truncated", True)):
                bad = copy.deepcopy(source)
                bad["rows"][0][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate(bad)
            for key in ("cases_sha256", "manifest_sha256", "study_sha256", "weights_identity", "config_sha256"):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate(dict(source, **{key: "wrong"}))
            with self.assertRaises(ValueError):
                validate(dict(source, rows=source["rows"][:1]))
            with self.assertRaises(ValueError):
                validate(dict(source, summary={"n": 1}))

    def test_known_policy_negatives_enter_category_scores_and_sensitivity(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, _, source = fixture(Path(temporary), "openai_categories")
            original = copy.deepcopy(source["rows"])
            summary = summarize_run(source["rows"])
            category = summary["policy_categories"]["S"]
            self.assertEqual(category["n"], 2)
            self.assertEqual(category["harmful_class"]["negative_support"], 1)
            self.assertEqual(category["harmful_class"]["support"], 1)
            self.assertEqual(summary["untruncated_sensitivity"]["n_excluded"], 1)
            self.assertEqual(summary["untruncated_sensitivity"]["summary"]["accuracy"], 1)
            self.assertEqual(source["rows"], original)

    def test_portable_analysis_fixed_files_and_primary_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, results, _, _, source = fixture(Path(temporary))
            for target in ("base", "fine1000", "fine5000"):
                run = copy.deepcopy(source)
                run["target"] = target
                if target != "base":
                    for row in run["rows"]:
                        row["prediction"] = row["expected"]
                (results/('aegis_prompt_'+target+'.json')).write_text(json.dumps(run))
            untouched = {p: p.read_bytes() for p in results.glob("*.json")}
            output = analyze(results, data, resamples=20)
            task = output["tasks"][0]
            self.assertEqual(task["pending"], [])
            self.assertEqual(task["comparisons"]["fine1000_vs_base"]["confidence"], .9875)
            self.assertEqual(task["comparisons"]["fine5000_vs_base"]["confidence"], .9875)
            self.assertEqual(task["comparisons"]["fine5000_vs_fine1000"]["confidence"], .95)
            self.assertIn("identities only", task["runs"]["base"]["checkpoint_verification"])
            self.assertEqual(output["unavailable_raw_sources"], ["raw/unbundled.json"])
            self.assertTrue(all(p.read_bytes() == blob for p, blob in untouched.items()))

    def test_incomplete_results_pending_and_frozen_tampering_fatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, results, _, _, source = fixture(Path(temporary), "wildguard_response")
            source["rows"] = source["rows"][:1]
            (results/"wildguard_response_base.json").write_text(json.dumps(source))
            output = analyze(results, data, resamples=20)
            self.assertEqual(len(output["skipped"]), 1)
            self.assertEqual(output["tasks"][0]["pending"], ["base", "fine1000", "fine5000"])
            (data/"wildguard_response/heldout.jsonl").write_text("tampered\n")
            with self.assertRaises(ValueError):
                analyze(results, data, resamples=20)

    def test_present_checkpoint_bytes_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, results, _, _, source = fixture(Path(temporary))
            checkpoint = Path(source["checkpoint"])
            checkpoint.mkdir()
            (checkpoint/"model.safetensors").write_bytes(b"different checkpoint")
            (checkpoint/"rl_agent_config.json").write_text(json.dumps({"max_len": 2048, "head_max_len": 256}))
            (results/"aegis_prompt_base.json").write_text(json.dumps(source))
            output = analyze(results, data, resamples=20)
            self.assertIn("checkpoint bytes", output["skipped"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
