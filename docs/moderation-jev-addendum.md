---
title: Jev moderation comparison
order: 8
---

# Jev moderation comparison

This secondary comparison was added after the Laya study was complete. It does not change the training data, checkpoints, original evaluation results, or planned comparisons. Jev outputs must never be used for training.

## What is compared

The evaluator sends the exact same frozen `state` and `questions` to Jev through OpenRouter's Decisions endpoint. It evaluates all 22,630 labelled decisions across nine existing views: Aegis prompts and responses, ToxicChat, BeaverTails, WildGuard prompts and responses, OpenAI known binary labels and individual categories, and XSTest. It uses existing source labels, the same safe/unsafe choice rule and the same harmful-content F1 scoring code as Laya.

This matches requests and evaluation labels, but not inference implementation. Laya has a documented 2,048-token budget; Jev receives the complete request and its server-side input handling is not observable. Eight WildGuard response cases were truncated by Laya. Jev is hosted, and Laya is local. Network timings and local timings are not hardware-controlled speed comparisons.

OpenAI and Mistral published scores remain separately attributed reference results. They are not measurements from this matched protocol. The original OpenAI known-label subset and XSTest prompt-task caveats still apply.

## Run and resume

The following is the original command for a fresh or resumable evaluator run. Readers of the published clone should use the offline scoring or fresh-run instructions at the end of this page; existing published predictions are protected against overwrite. From `bench/`, with `OPENROUTER_API_KEY` in the existing environment or project `.env`:

```bash
../.venv/bin/python -m unittest test_moderation_jev.py
../.venv/bin/python evaluate_moderation_jev.py --workers 4
../.venv/bin/python analyze_moderation_jev.py
```

Four bounded concurrent requests use separate HTTP clients. This concurrency is recorded in every result. One Aegis development example warms up each dataset; it is excluded from metrics. The requested model is `typesafe/jev-1.13`; the resolved model name and actual API-reported charges are retained. The evaluator refuses a changed resolved identity, changed input hashes, or incompatible resume metadata.

Each completed batch is appended in source order to `bench/results/moderation-study/*_jev.progress.jsonl`. The same command resumes after persisted rows without selecting or replacing failed predictions. Interrupted in-flight requests can be repeated because their responses were not saved. Complete results are written as `*_jev.json`, using the common moderation metrics implementation. The evaluator stops when the latest five predictions have failed. HTTP 429 retries follow the existing shared client behavior.

## Evidence and limits

Local telemetry archives full requests and raw answers under the existing telemetry root, with every Jev record marked ineligible for training. Public result files contain IDs, labels, predictions, probabilities, usage, resolved model identity, and timings; they do not contain source conversation text or credentials. The evaluation uses one hosted model version and no tuning against these results. Treat it as a post-hoc benchmark comparison, not a preregistered new primary experiment.

The user's explicit no-Git exception remains in effect for this workspace. This addition does not initialize or repair Git.

The independent analysis checks saved scores, case metadata and input hashes before computing paired 95% bootstrap intervals for Jev versus each Laya checkpoint. Resampling groups duplicate texts by the existing `text_group` field. These intervals are descriptive secondary results without a multiple-comparison adjustment. The comparison key `harmful_f1_delta_fine_minus_reference` comes from the shared helper and means Jev minus the named Laya reference in this addendum.

For a resumed evaluation, a result's `run_id` names only the final process segment, not every request. Locate all local telemetry manifests whose `source` is `public/moderation-study/jev-posthoc` and whose metadata match the result's dataset, requested model, case SHA-256, study SHA-256 and concurrency. Collect their held-out records and match each published row by case ID, resolved model, prediction and probabilities. Interrupted requests can appear more than once in telemetry; the ordered persisted progress rows are authoritative for scoring. Preserve all matching manifests rather than treating the last segment as the complete request archive. The initial comparison completed all 22,630 decisions in one uninterrupted evaluator process on September 21, 2026. The resolved model was `typesafe/jev-1.13-20260917`. One OpenAI category request received an HTTP 520 provider error and remains a failed prediction in the results; it was not selectively retried. Its gold label was safe, so the scorer records a negative-label error rather than a true negative. Reported held-out request charges total $0.48288874; the failed request has no reported charge, so this is not a claim that the complete billed amount is known. Development warm-ups are excluded from these counts and charges.

## Reproduce scores or make fresh hosted calls

To reproduce the published scores, run `analyze_moderation_jev.py` after preparing the frozen datasets. This needs no API calls or progress metadata. It verifies the recorded local results against the original complete Laya analysis and, when available, verifies the checkpoint bytes again.

Fresh hosted calls must use a separate clone of the published repository, with no existing `*_jev.progress.*` files. Preserve its published Jev results before running the evaluator:

```bash
# Run from bench/ in the separate clone, after preparing the datasets.
mkdir results/moderation-study/published-jev
mv results/moderation-study/*_jev.json results/moderation-study/published-jev/
mv results/moderation-study/jev_analysis.json results/moderation-study/published-jev/
../.venv/bin/python evaluate_moderation_jev.py --workers 4
../.venv/bin/python analyze_moderation_jev.py
```

Keep the original Laya results, `analysis.json`, and `preflight.json` in place. The published Jev files remain preserved in `published-jev/`; the new API run is a separate measurement. A hosted model alias can resolve to a newer version, so a future run need not reproduce identical predictions or costs.
