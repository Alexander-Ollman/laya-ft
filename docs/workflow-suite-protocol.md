---
title: Five additional workflow datasets
order: 12
---

# Testing whether the fine-tuning result holds across tasks

Owner request, 2026-09-20: download five further Hugging Face datasets, measure Jev and Laya, and include malicious prompt classification. This extends the independent report; the owner subsequently authorized publication on 21 September 2026. The explicit no-Git exception continues.

The hypothesis is that a small task-specific fine-tune improves base Laya across practical classification tasks and can narrow the gap to Jev. This is a test of that hypothesis, including null results or regressions. Each new Laya model starts independently from the same pinned base, not from the earlier banking or private workflow models.

## Frozen datasets

| Dataset | Task | Train / development / test | Source and terms |
|---|---|---|---|
| Deepset Prompt Injections | Classify attempted instruction overrides | 446 / 100 / 116 | [HF](https://huggingface.co/datasets/deepset/prompt-injections); current Apache-2.0 metadata, historical nested CC-BY-4.0 metadata retained |
| SMS Spam | Filter unsolicited messages | 1,000 / 200 / 1,000 | [HF](https://huggingface.co/datasets/ucirvine/sms_spam); [original UCI](https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection) specifies CC-BY-4.0 |
| DAIR Emotion | Recognize emotional tone | 1,000 / 200 / 1,000 | [HF](https://huggingface.co/datasets/dair-ai/emotion); research and educational use only; distant labels, not a fully human-labelled corpus |
| Amazon Counterfactual, English | Find statements imagining a different outcome in reviews | 1,000 / 200 / 670 | [HF](https://huggingface.co/datasets/SetFit/amazon_counterfactual); original author [LICENSE](https://github.com/amazon-science/amazon-multilingual-counterfactual-dataset/blob/main/LICENSE) says CC-BY-NC-4.0; mirror descriptions differ |
| MASSIVE, US English | Route requests among 60 assistant intents | 1,000 / 200 / 1,000 | [HF](https://huggingface.co/datasets/AmazonScience/massive); CC-BY-4.0; pinned official English Parquet conversion |

Exact revisions, raw file hashes, split hashes, source rows, labels and class counts are in `train/workflow-suite/<task>/manifest.json`. Sources are downloaded as data files; repository dataset scripts are not executed. The Parquet reader added for this expansion is PyArrow 25.0.1 (Apache-2.0, version checked against PyPI).

## Selection before scoring

All splits and prompts are frozen before model inference. Sampling preserves approximate class proportions and retains at least one example per class. Use at most 1,000 training examples and 200 development examples (100 for the smaller injection dataset). Official test sets with at most 1,000 rows are retained in full; larger ones use a fixed proportional sample of 1,000. Sampling seeds are test 7, development 8, training 9. Training seed is 7.

Remove normalized duplicates, conflicting labels and every official test/validation text from training, including official examples not sampled into this experiment. Also exclude previously benchmarked request text from the prior public/private benchmark archives. SMS provides one source split, so its new test is a custom split after deduplication and previous-benchmark exclusion, not an official benchmark split. Its exclusions include 322 previously benchmarked rows, 421 duplicate rows and two rows empty after normalization.

A preparation-only draft was archived at `train/workflow-suite-preflight-v1` before adding the older SMS benchmark to the exclusion list. No model saw those draft splits. Original source labels are retained: no Jev output or new teacher labelling is used.

## Models and measurement

Compare freshly measured Jev, base Laya with the same input limits as the fine-tune, and an independent task-specific Laya checkpoint. Base revision: `convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`. All Laya conditions use maximum sequence length 1,024 and option budget 768, matching the earlier public experiment. A tokenizer-only preflight verifies complete labels/instructions and records any truncated input text in `bench/results/workflow_suite_preflight.json`.

Train three fixed epochs with the established public trainer: label-smoothed cross entropy, encoder learning rate 2.5e-5, head learning rate 1e-4, AdamW, effective batch 64, final checkpoint regardless of outcome. No test-based prompt, epoch or hyperparameter selection. One GPU job runs at a time. Hosted Jev evaluation can run concurrently with local training; this incurs API charges and its latency reflects that execution setting.

Primary metric per dataset is macro-F1, giving each class equal weight. Also report accuracy, majority baseline, per-class recall, full Brier/NLL/ECE where complete probabilities exist, and client median/95th-percentile latency. Invalid responses count against accuracy and recall. Five consecutive failed requests stop a run instead of continuing through an outage. Any retry or partial run must be disclosed.

For each task, compare fine-tuned Laya against base Laya using paired resampling of normalized-text groups: 5,000 bootstrap draws, seed 7. Report 99% intervals for the five primary macro-F1 contrasts (a conservative Bonferroni family of five) and descriptive 95% intervals for accuracy and secondary fine-tuned Laya versus Jev comparisons. No pooled accuracy headline: each dataset remains visible. Intervals are conditional on these trained checkpoints, not evidence of training-seed robustness.

Injection and spam additionally report positive-class precision, recall, false-positive rate, confusion counts, and Wilson intervals for recall/false positives. Injection's 116 test examples support a small classification experiment, not proof that a downstream LLM cannot be successfully attacked. The dataset does not supply full system/task context for every attempted instruction override. Dataset labels can be ambiguous.

This suite does not establish absence of pretraining contamination. Familiar public datasets may already be part of either base model's training; Emotion in particular overlaps an already-tested task. Results and source attribution may enter the report. Research-restricted source text and derived weights are not automatically approved for public redistribution.

## Run and audit

From `bench/`, use the project virtual environment:

```bash
../.venv/bin/python prepare_workflow_suite.py
../.venv/bin/python run_workflow_suite.py --lane local
../.venv/bin/python run_workflow_suite.py --lane jev
../.venv/bin/python analyze_workflow_suite.py
```

Preparation refuses an existing frozen manifest. Training refuses an existing checkpoint. The two execution lanes record separate status files and logs under `bench/results/workflow-suite/`. Every request is archived through the existing telemetry recorder; no external messages or public uploads are part of this work.

The exclusion list is portable: `train/workflow-suite/prior_holdout_hashes.json` contains 8,249 normalized-text hashes, not private benchmark text. All five datasets were independently reconstructed in a temporary directory using that list, and all split files and manifests matched byte-for-byte. The tokenizer preflight found no shortened test input or answer label; one injection development request exceeds the sequence limit. Local suite timing measures direct model calls, while the earlier Banking77 table used the local HTTP server.

## Additional similarity audit

After the first four task comparisons were available, a post-hoc lexical audit checked normalized character-five-gram Jaccard similarity of at least 0.85 and exact normalized matches after replacing digit sequences. It flagged four SMS, one counterfactual and one MASSIVE test example; zero injection/emotion examples. `audit_workflow_similarity.py` records paired train/test IDs and descriptive scores with flagged rows excluded in `results/workflow-suite/similarity_audit.json`. No primary split, prompt, checkpoint or primary score changes. This audit is additional evidence, not a preregistered comparison or proof of semantic independence.
