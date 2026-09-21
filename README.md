# Can a small local model moderate chat?

Fine-tuning Laya improved harmful-content detection across several public tests. It also increased false alarms on harmless requests. This repository shares the measurements, code and instructions so that both sides of that result can be checked.

**[Read the report](https://alexander-ollman.github.io/laya-ft/)** · **[Reproduce the study](docs/reproduce.md)** · **[Study protocol](docs/moderation-study-protocol.md)**

We compared the same Laya base model before training and after fine-tuning on 1,000 or 5,000 existing Aegis safety labels. Each training set contains equal numbers of user-input and assistant-response decisions; the smaller set is included in the larger one. Both fine-tunes start independently from the base model.

The completed study covers six data sources, nine evaluation tasks and 27 model–task runs: **67,890 decisions, with no request failures**. It uses public source labels, including their documented human and model-generated labels. No Jev outputs or new teacher-generated labels were used for this moderation experiment.

| Harmful-content F1 | Before training | 1,000 decisions | 5,000 decisions |
|---|---:|---:|---:|
| Aegis prompts | 60.6% | 81.5% | 83.7% |
| Aegis responses | 37.4% | 69.5% | 77.0% |
| ToxicChat | 36.4% | 50.5% | 53.3% |
| WildGuard prompts | 48.7% | 59.4% | 65.7% |
| WildGuard responses | 20.8% | 53.3% | 51.6% |
| BeaverTails | 6.0% | 34.1% | 59.4% |

On XSTest's 250 harmless prompts, false alarms rose from **24.8%** before training to **65.2%** with 1,000 training decisions and **47.2%** with 5,000. More training improved several scores but did not make this a dependable general-purpose safety filter. The full report includes OpenAI moderation categories, confidence intervals, false-positive rates and observed response times.

OpenAI and Mistral advertised scores appear as separately attributed references. We did not run those models here, and differences in policy, examples and scoring prevent a direct leaderboard claim. See the [reference notes](docs/moderation-reference-notes.md).

## What is included

- `bench/`: preparation, training, evaluation, scoring, report generation and tests.
- `bench/results/moderation-study/`: saved predictions without prompt/response text, aggregate results, runtime records and checkpoint identities.
- `evidence/`: recorded source/split manifests used to verify regenerated data.
- `train/moderation-study/prior_exclusion_hashes.json`: text-free hashes used to exclude earlier benchmark examples.
- `index.html` and `assets/`: the public report and downloadable bar charts.
- `report/`: the detailed report generated from the saved measurements.
- `docs/`: method, source notes and reproduction instructions.

Dataset text, model weights, credentials and private Jev outputs are not distributed here. You can read all results immediately. Rebuilding and independently validating the splits requires downloading the pinned sources, including authorized access to WildGuard. Saved-prediction replay does not require GPU inference or a paid model API.

The publication branch is `research/moderation-publication`. [`PUBLIC_SHA256SUMS.json`](PUBLIC_SHA256SUMS.json) records the public snapshot files.

## Scope and rights

This is an independent Laya fine-tuning study extending earlier Jev/Laya workflow research. There is no newly measured Jev moderation baseline in this repository. Results come from one training seed, fixed final checkpoints and a shared Mac workstation; response times are observational. Exact-text exclusions reduce test leakage but cannot establish absence of near duplicates or contamination in the base model's pretraining.

The original sources retain their own terms: Aegis 2.0 and XSTest use CC BY 4.0; ToxicChat and BeaverTails use CC BY-NC 4.0; WildGuardMix uses ODC-BY with access conditions; the OpenAI moderation release uses MIT. Follow the source links and pinned revisions in [`bench/moderation_sources.json`](bench/moderation_sources.json) before using or redistributing data. Laya's model and package terms also apply. Publication of this research does not grant additional rights over any source dataset or model.
