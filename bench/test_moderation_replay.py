"""CPU replay verification; all writes remain inside temporary fixture directories."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from public_workflow import digest
from restore_moderation_evidence import verify_and_restore


def fixture(root):
    data, evidence = root/"data", root/"evidence"
    for folder in (data/"raw", data/"task", evidence/"task"):
        folder.mkdir(parents=True)
    raw = b"raw source bytes\n"
    split = b'{"id": "one"}\n'
    (data/"raw/source.json").write_bytes(raw)
    (data/"task/heldout.jsonl").write_bytes(split)
    manifest = {"slug": "task", "split_sha256": {"heldout": digest(split)}}
    for folder in (data, evidence):
        (folder/"task/manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    study = {"schema": "moderation-study-v1", "test_counts": {"task": 1},
             "training_sizes": [], "raw_sha256": {"raw/source.json": digest(raw)}}
    (evidence/"study_manifest.json").write_text(json.dumps(study, indent=2)+"\n")
    (data/"study_manifest.json").write_text(json.dumps(dict(reversed(list(study.items())))))
    return data, evidence, study


class ModerationReplayTests(unittest.TestCase):
    def test_semantic_reordering_restores_exact_recorded_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, evidence, _ = fixture(Path(temporary))
            frozen_split = (data/"task/heldout.jsonl").read_bytes()
            output = verify_and_restore(data, evidence)
            self.assertTrue(output["study_manifest_restored"])
            self.assertEqual(output["raw_files_verified"], 1)
            self.assertEqual(output["split_files_verified"], 1)
            self.assertEqual((data/"study_manifest.json").read_bytes(), (evidence/"study_manifest.json").read_bytes())
            self.assertEqual((data/"task/heldout.jsonl").read_bytes(), frozen_split)
            self.assertFalse(verify_and_restore(data, evidence)["study_manifest_restored"])

    def test_metadata_or_referenced_bytes_changes_rejected_without_restore(self):
        for kind in ("study", "raw", "split", "split_hash"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                data, evidence, study = fixture(Path(temporary))
                if kind == "study":
                    altered = copy.deepcopy(study)
                    altered["test_counts"]["task"] = 2
                    (data/"study_manifest.json").write_text(json.dumps(altered))
                elif kind == "raw":
                    (data/"raw/source.json").write_bytes(b"changed raw")
                elif kind == "split":
                    (data/"task/heldout.jsonl").write_bytes(b"changed split")
                else:
                    path = data/"task/manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["split_sha256"]["heldout"] = "0"*64
                    path.write_text(json.dumps(manifest))
                before = (data/"study_manifest.json").read_bytes()
                with self.assertRaises(ValueError):
                    verify_and_restore(data, evidence)
                self.assertEqual((data/"study_manifest.json").read_bytes(), before)

    def test_missing_evidence_dataset_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, evidence, _ = fixture(Path(temporary))
            (evidence/"task/manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "study datasets"):
                verify_and_restore(data, evidence)


if __name__ == "__main__":
    unittest.main()
