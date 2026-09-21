"""Verify a regenerated moderation dataset, then restore recorded study JSON bytes.

Filesystem traversal can reorder raw_sha256 object keys without changing any
data. This helper accepts only semantic equality and verified source/split bytes;
it restores study_manifest.json formatting/order so archived result hashes match.
"""
import argparse
import json
from pathlib import Path

from public_workflow import ROOT, digest


def referenced_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("evidence reference escapes dataset directory")
    return path


def verify_and_restore(data, evidence):
    recorded_path = evidence / "study_manifest.json"
    recorded_bytes = recorded_path.read_bytes()
    regenerated_path = data / "study_manifest.json"
    regenerated_bytes = regenerated_path.read_bytes()
    recorded = json.loads(recorded_bytes)
    regenerated = json.loads(regenerated_bytes)
    if recorded != regenerated:
        raise ValueError("regenerated study metadata differs from recorded evidence")
    raw_count = 0
    for relative, expected in recorded["raw_sha256"].items():
        if digest(referenced_path(data, relative).read_bytes()) != expected:
            raise ValueError("raw source hash differs from evidence: " + relative)
        raw_count += 1
    manifests = sorted(evidence.glob("*/manifest.json"))
    required = set(recorded["test_counts"]) | {"aegis_"+str(n) for n in recorded.get("training_sizes", [])}
    if {path.parent.name for path in manifests} != required:
        raise ValueError("evidence dataset manifests do not match study datasets")
    split_count = 0
    for path in manifests:
        slug = path.parent.name
        expected_bytes = path.read_bytes()
        expected = json.loads(expected_bytes)
        actual_bytes = (data / slug / "manifest.json").read_bytes()
        if json.loads(actual_bytes) != expected:
            raise ValueError("regenerated dataset manifest differs from evidence: " + slug)
        # These files are independently hashed by each result; this helper only
        # restores the filesystem-order-sensitive top-level study manifest.
        if actual_bytes != expected_bytes:
            raise ValueError("dataset manifest bytes differ from evidence: " + slug)
        if expected["slug"] != slug:
            raise ValueError("evidence manifest slug mismatch")
        for split, sha in expected["split_sha256"].items():
            relative = slug + "/" + split + ".jsonl"
            if digest(referenced_path(data, relative).read_bytes()) != sha:
                raise ValueError("frozen split hash differs from evidence: " + relative)
            split_count += 1
    # No writes occur until every semantic comparison and byte hash passes.
    restored = regenerated_bytes != recorded_bytes
    if restored:
        regenerated_path.write_bytes(recorded_bytes)
    return {"raw_files_verified": raw_count, "split_files_verified": split_count,
            "study_manifest_restored": restored, "study_sha256": digest(recorded_bytes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"train/moderation-study")
    parser.add_argument("--evidence", type=Path, default=ROOT/"evidence")
    args = parser.parse_args()
    print(json.dumps(verify_and_restore(args.data, args.evidence), indent=2))


if __name__ == "__main__":
    main()
