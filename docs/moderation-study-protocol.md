---
title: Local chat moderation study
order: 13
---

# Can a small local model moderate chat reliably?

Approved by the owner on 21 September 2026. Train two independent Laya checkpoints on 1,000 and 5,000 labelled Aegis decision examples; measure input and response safety plus cross-dataset transfer. Each budget is half prompt decisions and half response decisions. The smaller set is nested inside the larger set. No Jev outputs, new teacher labels, or provider model calls are used. Existing no-Git exception continues. The owner subsequently authorized publication on 21 September 2026. This protocol describes the original Laya experiment; the later hosted Jev comparison has a separate [addendum](moderation-jev-addendum.md).

## Frozen design

- Base: `convaiinnovations/laya` revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
- Both fine-tunes start from base, use seed 7, three fixed epochs, final checkpoint, and the existing supervised trainer. Learning rates: encoder 2.5e-5, head 1e-4; effective batch 64; label smoothing 0.05.
- Matched base and both fine-tunes use 2,048 sequence tokens and a 256-token question/option budget. Preflight records shortening of inputs or instructions. No hidden removal of long test cases.
- Training source: Aegis 2.0 training plus synthetic refusal augmentation. Labels retain their original human or model-generated provenance. A separate 400-example development sample comes only from official validation data; no test-based checkpoint or threshold selection.
- Global exclusion uses normalized component hashes: a matching prompt or response removes a training example. All source test content is excluded, including rows omitted from scoring because labels are unknown. Complete official development content and prior benchmark hashes are excluded too. Conflicting training labels and duplicates are removed. Near-duplicate/pretraining contamination remains possible.
- A preparation-only draft was archived before fixing exclusion of unscorable source rows. No inference used that draft.

## Runtime correction

Two incomplete training attempts were stopped before any checkpoint or held-out evaluation. Variable-length batches first showed growing memory use; masked padding reduced shape variation but retained allocator cache still reached 106 GB with only 6.8 GB of live tensors. A training-only diagnostic confirmed that releasing unused cache reduced driver memory from 59 GB to 16 GB after a longest-input batch without removing live tensors.

Both final runs start from base with batches padded to multiples of 128 tokens and unused Metal allocator cache released after every optimizer step (`--pad-to-multiple 128 --mps-cache-clear-interval 1`). Padding is masked and does not replace input text. Data, token limits, seed, epochs and optimizer settings are unchanged. A development-only check on 20 examples produced identical choices and probabilities with and without padding. Original attempts, restart records and diagnostics are preserved under `bench/results/moderation-study/`.

## Evaluation sets

| Set | Scored task | Handling |
|---|---|---|
| Aegis 2.0 | Prompt safety and response safety, separately | Official test; omit unavailable/redacted text and missing labels. Response decisions retain user context. |
| ToxicChat 0124 | Toxic user input | Human-annotated official test subset, matching the subset size in Mistral's report. |
| WildGuardTest | Prompt harm and response harm | Official gated data accessed after owner enabled access. Unknown harm labels omitted separately per task. |
| BeaverTails | Response safety in context | Entire original 30k release test split; do not mix with its 330k release. |
| OpenAI 2022 | Known binary subset and individual category decisions | Binary subset: any positive category means unsafe; safe requires all eight explicit negatives. Otherwise unknown. Category evaluation scores only supplied labels. |
| XSTest | Benign-request false alarms and unsafe contrasts | Original 450 prompts: 250 safe, 200 unsafe. This is not the published response-refusal classification task. |

All generic prompt and response decisions share fixed broad Aegis-oriented policy wording. OpenAI category queries use the source's category definitions, and its known-binary query uses their union. Category recall reports coverage of unsafe examples, not clinical reliability or downstream attack prevention. Aegis category metadata is used for prompt-positive slices only, because it does not provide independent response category labels.

## Metrics and comparisons

Primary measure: harmful-class F1 for Aegis prompt and response tasks, with base versus each training budget: four prespecified contrasts. Use paired bootstrap by normalized source prompt, 5,000 draws, seed 7, 98.75% intervals (Bonferroni family of four). Other paired comparisons use descriptive 95% intervals. One training seed does not establish robustness across training randomness.

Also report harmful precision/recall, benign false-positive rate, confusion counts, accuracy, macro-F1, noninterpolated average precision with tied scores grouped, per-category positive recall, and local median/95th-percentile response time. Fixed model choice is the decision; no test threshold tuning. Errors count against accuracy and harmful recall. Five consecutive errors stop a run.

Published OpenAI and Mistral scores are external references, not paired measurements. See [reference notes](./moderation-reference-notes.md) and [the exact score records](../bench/moderation_references.json). Their model versions, policy wording, subsets, thresholds and F1 definitions may differ. OpenAI's 859 known-binary subset and masked category evaluation must not be presented as the same task as its advertised full-set aggregate. Mistral's model card and OpenAI's report evaluate safeguard independently; keep their numbers attributed separately.

## Reproduction and preservation

Pinned source URLs and raw hashes: `train/moderation-study/raw/sources.json` and `study_manifest.json`. Frozen inputs and hashes: per-task directories under `train/moderation-study/`. Training, inference and report evidence are archived through the existing telemetry recorder. Source content remains local; no restricted or gated text is placed in the review evidence ZIP. Run one GPU workload at a time and preserve existing services and checkpoints.

Preflight found no shortened test input except eight WildGuard responses; their descriptive scores with those rows omitted are reported alongside the full test. One training input in the smaller set and three in the larger set are shortened. All questions, answer options and development inputs fit intact. Study jobs run sequentially on a shared workstation; other local services are preserved, so latency is an observed measurement under this workload, not an uncontended hardware benchmark.

`fetch_moderation_sources.py` retrieves data from pinned URLs and verifies hashes. `prepare_moderation_study.py` recreates the frozen splits using `prior_exclusion_hashes.json`, which contains no source text. All 27 split/manifest files reproduced byte-for-byte in an isolated directory before scoring.

For evidence-bundle replay, run `restore_moderation_evidence.py` after preparation and before analysis. It verifies raw and split hashes and metadata equality before restoring recorded study-manifest formatting. This handles filesystem-dependent JSON key order without accepting changed data.
