---
title: Moderation benchmark reference notes
order: 14
---

# How to read the moderation comparisons

The new study measures what Laya learns from 1,000 or 5,000 Aegis examples, then tests whether that learning transfers to other safety datasets. Published model scores provide context. They are not new measurements, and sharing a dataset name does not make two experiments equivalent.

Exact reference values, source versions, access checks and dataset mappings are recorded in [moderation_references.json](../bench/moderation_references.json). Checked on 21 September 2026. No models were run during this reference review.

## Published scores

The [Shieldstral model card](https://huggingface.co/mistralai/Shieldstral-1.0-3B/blob/003ec7e2b0bab5f0e6307edbaf186fa5822b76f5/README.md) separates prompt safety, response safety and refusal detection. Its Shieldstral scores use a 0.5 threshold; its safeguard comparison uses high reasoning effort. Keep these tasks separate in tables. In particular, its XSTest refusal score is not a false-positive rate on benign prompts.

The [Shieldstral paper, Table 4](https://arxiv.org/html/2607.25857v1#S7) reports benchmark sizes that differ from several raw sources or proposed subsets. A complete list of selected rows and benchmark-specific prompts was not established by this review. These references cannot establish a statistically paired win over Shieldstral.

The [OpenAI technical report, Table 2](https://cdn.openai.com/pdf/08b7dee4-8bc6-4955-a219-7793fb69090c/Technical_report__Research_Preview_of_gpt_oss_safeguard.pdf) evaluates moderation with internal policies, including an adjusted policy, and ToxicChat with a short adapted policy prompt. It does not specify Table 2's reasoning effort or averaging convention. Its safeguard results come from a different experiment than Mistral's safeguard column. Preserve publisher attribution rather than combining these columns into one leaderboard.

## What each dataset actually labels

| Dataset | Appropriate target | Important boundary |
|---|---|---|
| [Aegis v2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | Response safety when a response is present; prompt safety separately | A harmful question followed by a safe refusal must not become an unsafe response merely because the question is harmful. |
| [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) | Toxicity of the user prompt | The supplied model response is not the labelled target. Declare whether evaluation uses all test rows or only human-annotated rows. |
| [WildGuardTest](https://huggingface.co/datasets/allenai/wildguardmix) | Separate prompt harm, response harm and refusal tasks | Access is gated. Missing annotations must not silently become safe. |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Response safety in its question context | Keep the 30k test and 330k test distinct. Its category definitions are broader than some other moderation policies. |
| [OpenAI moderation](https://github.com/openai/moderation-api-release) | Category labels with missing values preserved | Absence of a category annotation means unknown, not negative. |
| [Original XSTest](https://github.com/paul-rottger/xstest) | Prompt safety, including 250 benign prompts | A false-positive check here differs from published response/refusal tasks. |

Aegis training includes a separate refusal file. Its response labels mix human annotations, a model jury, and refusal augmentation. Record which files are included, exclude redacted content, and define treatment of missing or empty responses before selection. Do not describe every label as human-authored.

## OpenAI's incomplete labels

Inspection of the pinned 1,680-row source found 522 rows with at least one positive category, 337 with all eight categories explicitly negative, and 821 with no positive but incomplete annotation. A conservative binary projection retains the first two groups: 859 examples. Report that denominator and the exclusion rule.

An [existing AllenAI loader](https://github.com/allenai/safety-eval/blob/060cc903d64703214c549b5c3a30ea8ceef2e588/evaluation/tasks/classification/openai_mod/__init__.py) instead labels every row without an observed positive as unharmful. That is a benchmark convention, not evidence that the missing labels are negative. The conservative subset and that convention answer different questions.

## Access and reuse

Aegis and original XSTest specify CC BY 4.0; ToxicChat and BeaverTails specify CC BY-NC 4.0. The OpenAI source repository uses MIT. Keep attribution and source-specific restrictions with any future release.

WildGuardMix's metadata reports ODC-BY and automatic gating. The owner enabled access, and the authenticated local download of the pinned test file succeeded on 21 September 2026. The agent accepted no access terms. The gated WalledAI XSTest mirror is unnecessary because the author's original GitHub source is available.

## Interpretation

Use prompt-only transfer tests to measure prompt classification and prompt-plus-response tests to measure the response in context. Do not pool their scores into a single safety claim. For every binary test, retain unsafe-class precision and recall alongside F1, benign false positives and uncertainty. A classifier benchmark does not establish downstream protection against successful attacks.
