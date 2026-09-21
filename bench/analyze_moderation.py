"""Validate frozen moderation evidence and compute CPU-only safety comparisons."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from analyze_public_study import paired_rows
from moderation_metrics import clean_rows, moderation_metrics, paired_bootstrap
from public_workflow import ROOT, digest, read_jsonl

TARGETS = ("base", "fine1000", "fine5000")
PRIMARY = ("aegis_prompt", "aegis_response")


def checkpoint_identity(path, cache):
    path = Path(path).resolve()
    if path not in cache:
        if not (path / "model.safetensors").is_file() or not (path / "rl_agent_config.json").is_file():
            return None
        with (path / "model.safetensors").open("rb") as handle:
            weights = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
        config = (path / "rl_agent_config.json").read_bytes()
        cfg = json.loads(config)
        if cfg.get("max_len") != 2048 or cfg.get("head_max_len") != 256:
            raise ValueError("checkpoint sequence budgets differ from frozen protocol")
        cache[path] = (weights, digest(config))
    return cache[path]


def validated_run(source, target, manifest, manifest_hash, study_hash, cases, truncated, identities):
    if source.get("dataset") != manifest["slug"] or source.get("target") != target:
        raise ValueError("dataset/target differs from fixed result filename")
    if type(source.get("pad_to_multiple")) is not int or source["pad_to_multiple"] != 128:
        raise ValueError("evaluation padding must match the 128-token study protocol")
    for key, expected in (("manifest_sha256", manifest_hash), ("study_sha256", study_hash),
                          ("cases_sha256", manifest["split_sha256"]["heldout"])):
        if source.get(key) != expected:
            raise ValueError("result provenance mismatch: " + key)
    rows = source["rows"]
    if len(rows) != len(cases) or source.get("summary", {}).get("n") != len(cases):
        raise ValueError("incomplete result count")
    canonical = [{"id": c["id"], "expected": c["expected"]["intent"], "text_group": c["text_group"]} for c in cases]
    paired_rows(canonical, rows)
    by_id = {c["id"]: c for c in cases}
    weights, config = source.get("weights_identity"), source.get("config_sha256")
    if not isinstance(weights, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", weights):
        raise ValueError("invalid checkpoint weight identity")
    if not isinstance(config, str) or not re.fullmatch(r"[0-9a-f]{64}", config):
        raise ValueError("invalid checkpoint config identity")
    local_identity = checkpoint_identity(source["checkpoint"], identities)
    if local_identity is not None and local_identity != (weights, config):
        raise ValueError("result checkpoint identity differs from checkpoint bytes")
    if local_identity is not None and target != "base":
        cfg = json.loads((Path(source["checkpoint"])/"rl_agent_config.json").read_text())
        if cfg.get("finetune", {}).get("pad_to_multiple") != 128:
            raise ValueError("fine-tune padding differs from the 128-token study protocol")
    for row in rows:
        case = by_id[row["id"]]
        if row.get("model_resolved") != weights:
            raise ValueError("row checkpoint identity mismatch")
        if row.get("categories") != case.get("categories", []):
            raise ValueError("row categories mismatch")
        if row.get("policy_category") != case.get("policy_category"):
            raise ValueError("row policy_category mismatch")
        if type(row.get("input_truncated")) is not bool or row["input_truncated"] != (row["id"] in truncated):
            raise ValueError("row truncation flag differs from preflight")
    return clean_rows(rows)


def summarize_run(rows):
    policies = sorted({r["policy_category"] for r in rows if r.get("policy_category")})
    output = {"summary": moderation_metrics(rows), "policy_categories": {
        category: moderation_metrics([r for r in rows if r.get("policy_category") == category])
        for category in policies}}
    complete = [r for r in rows if not r["input_truncated"]]
    output["untruncated_sensitivity"] = {
        "n_excluded": len(rows)-len(complete), "n_retained": len(complete),
        "summary": moderation_metrics(complete) if complete else None,
        "scope": "Descriptive scores on untruncated inputs only; no paired confidence interval computed for this subset."}
    return output


def analyze(results, data, resamples=5000):
    output = {"schema": "moderation-analysis-v1", "tasks": [], "skipped": [], "complete": False,
              "pad_to_multiple": 128,
              "selection": "Fixed slug_target.json files only; no selection by score.",
              "primary_family": "Four harmful-F1 contrasts: Aegis prompt/response by 1000/5000 decisions versus base; 98.75% Bonferroni intervals. Accuracy and transfer comparisons are secondary."}
    study_path = data / "study_manifest.json"
    if not study_path.exists():
        output["pending"] = ["Frozen study manifest unavailable"]
        return output
    study_blob = study_path.read_bytes()
    study = json.loads(study_blob)
    study_hash = digest(study_blob)
    output["study_sha256"] = study_hash
    output["unavailable_raw_sources"] = []
    for relative, expected in study["raw_sha256"].items():
        if not (data / relative).is_file():
            output["unavailable_raw_sources"].append(relative)
            continue
        if digest((data / relative).read_bytes()) != expected:
            raise ValueError("frozen raw source hash mismatch: " + relative)
    manifests = {}
    for path in sorted(data.glob("*/manifest.json")):
        blob = path.read_bytes()
        manifest = json.loads(blob)
        if manifest["slug"] != path.parent.name or manifest["labels"] != ["safe", "unsafe"]:
            raise ValueError("frozen manifest slug/labels mismatch")
        for split, expected in manifest["split_sha256"].items():
            splitpath = path.parent / (split + ".jsonl")
            if digest(splitpath.read_bytes()) != expected:
                raise ValueError("frozen split hash mismatch: " + str(splitpath))
        manifests[path.parent.name] = (manifest, digest(blob))
    for size in study.get("training_sizes", []):
        slug = "aegis_" + str(size)
        if slug not in manifests or manifests[slug][0].get("train") != size:
            raise ValueError("frozen training manifest missing or budget mismatch: " + slug)
    preflight_path = results / "preflight.json"
    if not preflight_path.exists():
        output["pending"] = ["Tokenizer preflight unavailable"]
        return output
    preflight_blob = preflight_path.read_bytes()
    preflight = json.loads(preflight_blob)
    output["preflight_sha256"] = digest(preflight_blob)
    identities, target_identities = {}, {}
    for slug, count in study["test_counts"].items():
        if slug not in manifests:
            raise ValueError("frozen test manifest missing: " + slug)
        manifest, manifest_hash = manifests[slug]
        path = data / slug / "heldout.jsonl"
        cases = read_jsonl(path)
        if not cases or len(cases) != count or manifest["test"] != count:
            raise ValueError("frozen test count mismatch: " + slug)
        if any(c["expected"]["intent"] not in manifest["labels"] or
               list(c["questions"]["intent"]["criteria"]) != manifest["labels"] for c in cases):
            raise ValueError("frozen test labels/options mismatch")
        matches = [v for k, v in preflight.items() if k.endswith("/"+slug+"/heldout.jsonl")]
        if len(matches) != 1 or matches[0]["n"] != count:
            raise ValueError("preflight missing or count mismatch: " + slug)
        truncated = set(matches[0]["truncated_ids"])
        if not truncated <= {c["id"] for c in cases} or len(truncated) != matches[0]["input_truncated"]:
            raise ValueError("preflight truncation IDs/count mismatch")
        task = {"slug": slug, "n": count, "runs": {}, "comparisons": {}, "pending": []}
        selected = {}
        for target in TARGETS:
            resultpath = results / (slug + "_" + target + ".json")
            if not resultpath.exists():
                task["pending"].append(target)
                continue
            try:
                blob = resultpath.read_bytes()
                source = json.loads(blob)
                rows = validated_run(source, target, manifest, manifest_hash, study_hash, cases, truncated, identities)
                identity = (source["weights_identity"], source["config_sha256"])
                if target in target_identities and target_identities[target] != identity:
                    raise ValueError("same target uses different checkpoints across datasets")
                target_identities[target] = identity
                task["runs"][target] = {"input": str(resultpath), "input_sha256": digest(blob),
                                        "checkpoint": source["checkpoint"], "weights_identity": source["weights_identity"],
                                        "config_sha256": source["config_sha256"],
                                        "pad_to_multiple": source["pad_to_multiple"],
                                        "checkpoint_verification": "local weight/config bytes verified" if Path(source["checkpoint"]).resolve() in identities else "consistent recorded identities only; checkpoint files unavailable",
                                        **summarize_run(rows)}
                selected[target] = rows
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
                output["skipped"].append({"input": str(resultpath), "reason": str(exc)})
                task["pending"].append(target)
        for reference, fine in (("base", "fine1000"), ("base", "fine5000"), ("fine1000", "fine5000")):
            if reference not in selected or fine not in selected:
                continue
            primary = slug in PRIMARY and reference == "base"
            task["comparisons"][fine+"_vs_"+reference] = {
                "primary": primary, "reference": reference, "fine": fine,
                **paired_bootstrap(selected[reference], selected[fine], resamples=resamples,
                                   confidence=.9875 if primary else .95)}
        task["complete"] = not task["pending"]
        output["tasks"].append(task)
    output["complete"] = bool(output["tasks"]) and all(task["complete"] for task in output["tasks"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "bench/results/moderation-study")
    parser.add_argument("--data", type=Path, default=ROOT / "train/moderation-study")
    args = parser.parse_args()
    output = analyze(args.results, args.data)
    args.results.mkdir(parents=True, exist_ok=True)
    path = args.results / "analysis.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(path)


if __name__ == "__main__":
    main()
