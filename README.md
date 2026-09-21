# Jev vs. fine-tuned Laya

How much can a small amount of good training data improve a local AI model? This report compares Jev with Laya before and after task-specific fine-tuning, then examines what happens when the same approach is applied to chat safety.

**[Read the full report](https://alexander-ollman.github.io/laya-ft/)** · **[Reproduce the public experiments](docs/reproduce-full.md)** · **[Moderation report](https://alexander-ollman.github.io/laya-ft/moderation.html)**

The study has four parts:

- **Customer-support routing:** Banking77, with 1,001 training examples and all 3,080 official test questions. Matched-budget Laya improved from **51.3% to 79.4% accuracy**; Jev scored **80.0%**. That small final gap does not establish a winner.
- **Five more public workflows:** prompt-injection detection, SMS spam, emotion, product-review counterfactuals and assistant request routing. Each task has its own fine-tuned Laya model and a measured Jev comparison.
- **An exploratory curated pilot:** routing, tool selection and safety decisions. Only aggregate results are published; the private examples are not an independently reproducible public benchmark.
- **Chat moderation:** two independent Aegis fine-tunes, using 1,000 or 5,000 labelled decisions, tested across six sources. The report separates locally measured results from OpenAI and Mistral advertised scores.

## Five public workflows

Macro-F1 gives each class equal weight. It is a different metric from the Banking77 accuracy above and the harmful-class F1 used for moderation.

| Task | Laya before | Laya fine-tuned | Jev |
|---|---:|---:|---:|
| Prompt-injection detection | 69.8% | 94.8% | 78.7% |
| SMS spam | 76.0% | 98.1% | 90.8% |
| Emotion | 47.1% | 65.9% | 49.7% |
| Product-review counterfactuals | 57.8% | 90.0% | 86.5% |
| Assistant request routing | 45.7% | 75.0% | 79.9% |

Fine-tuning improved Laya on all five tests. These are results for particular checkpoints and frozen examples, not a claim that one model is always better. The report includes paired uncertainty intervals, class-level results and dataset limitations.

## Safety needs more than a detection score

The local moderation study completed **67,890 decisions across 27 model–task runs**, without request failures. Aegis prompt F1 rose from **60.6%** to **81.5%** and **83.7%** after training. However, false alarms on XSTest's 250 harmless prompts rose from **24.8%** to **65.2%** and **47.2%**. Useful learning did not establish readiness for unattended moderation.

The matched Jev comparison adds **22,630 decisions**, with one provider error retained in the results. On the same XSTest prompts, Jev caught **94.0%** of harmful requests and flagged **7.6%** of harmless ones, versus **91.5%** and **47.2%** for the larger Laya fine-tune. The [Jev moderation addendum](docs/moderation-jev-addendum.md) records the model version, results and secondary-analysis limits. OpenAI and Mistral scores are attributed published references, not models we ran locally; their policies, subsets and scoring can differ.

## What can be reproduced?

The repository includes preparation, training, evaluation, scoring and report code; public-test predictions; source and split manifests; and checkpoint identities. Saved-prediction scoring needs no paid API or GPU inference. Fresh hosted Jev runs require your own credentials and incur API charges.

Banking77 data is included with its license. Other source data is downloaded from pinned revisions; restricted and gated text is not redistributed here. Model weights, credentials, private pilot examples and private pilot Jev outputs are excluded. Public Jev predictions are evaluation evidence only: **none of these fine-tunes trained on Jev output**.

The public workflows use their source datasets' original labels, including distant supervision where documented. The curated pilot used labels supplied by a local Qwen model. Moderation uses existing Aegis labels with their original human and model-generated provenance. These sources of supervision remain distinct in the report.

Start with [the full reproduction guide](docs/reproduce-full.md), then [moderation reproduction](docs/reproduce.md) for that extension. The publication branch is `research/full-report`; [`PUBLIC_SHA256SUMS.json`](PUBLIC_SHA256SUMS.json) records the public snapshot. `index.html` is the combined report, `moderation.html` is the standalone moderation view, and `assets/` contains downloadable charts.

Source datasets and models retain their own licenses and conditions, including research-only and noncommercial restrictions for some datasets. See the [workflow protocol](docs/workflow-suite-protocol.md), [Banking77 protocol](docs/public-benchmark-protocol.md) and [moderation source notes](docs/moderation-reference-notes.md). Exact-text exclusions reduce leakage but cannot rule out near duplicates or pretraining contamination. One training seed and shared-workstation timings also limit generalization.
