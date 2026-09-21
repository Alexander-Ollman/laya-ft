---
title: Reproduce the full Jev and Laya report
order: 1
---

# Reproduce the full report

This publication combines Banking77, five public workflow tasks, an exploratory curated pilot and the moderation extension. Public predictions can be rescored without a GPU or paid model API. Fresh training and inference are separate operations. Private pilot data is not included, so its aggregate results cannot be independently replayed from this repository.

## Set up the recorded environment

```sh
git clone --branch research/full-report https://github.com/Alexander-Ollman/laya-ft.git
cd laya-ft
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy==2.5.3' 'pyarrow==25.0.1' 'httpx==0.28.1' 'huggingface-hub==1.32.0' 'matplotlib==3.11.2'
cd bench
```

These are recorded experiment/publication versions, not recommendations to use the latest releases. The model runs used Python 3.12.12 on an Apple M4 Max with macOS 15.6. Runtime records are in [`public_runtime.json`](../bench/results/public_runtime.json), [`workflow_suite_runtime.json`](../bench/results/workflow_suite_runtime.json) and the [moderation runtime](../bench/results/moderation-study/runtime.json).

Run the remaining commands from `bench/`. Scripts resolve data paths from their own repository root. Historical absolute paths in prediction records identify the original experiment; they do not require that machine or directory to exist for rescoring.

## Rebuild the presentation without downloading data

```sh
python build_full_report.py
```

This rebuilds `../index.html` and the workflow/Banking77 charts from saved analyses and the preserved historical report at `../report/jev-vs-laya.html`. It performs no model calls and does not independently verify the source data. The original report preserves aggregate pilot results without exposing private examples.

The standalone moderation page is `../moderation.html`. Follow [the moderation guide](reproduce.md) for its source downloads, independent scoring and chart generation. Check the output path of the publication builder before running it: the combined report belongs at `index.html`, and the moderation view belongs at `moderation.html`.

## Independently rescore Banking77

Banking77's frozen public data is already included under `train/public-banking77/`, along with the original license. Skip preparation and run:

```sh
python analyze_public_study.py --results results --data ../train/public-banking77 --out results/public_study_analysis.json
```

The analyzer verifies held-out IDs, labels, text groups and input hashes; recomputes accuracy, macro-F1 and probability metrics; and repeats the grouped paired bootstrap. Development-only temperature fitting is kept separate from test scoring.

To reconstruct Banking77 from upstream, use a separate working copy and first move its included data aside:

```sh
mv ../train/public-banking77 ../train/public-banking77-recorded
python public_workflow.py prepare
```

Compare the regenerated manifest and split hashes with `public-banking77-recorded/`. Preparation downloads the original PolyAI CSV files at pinned commit `57ec275d8078af65b7731c2a98be812d844a6d6b` and refuses an existing dataset directory. It recreates 1,001 training, 770 development and 3,080 test examples. The 77 category labels are presented to every model.

## Reconstruct and rescore the five workflow tasks

```sh
python prepare_workflow_suite.py
python analyze_workflow_suite.py --results results/workflow-suite --data ../train/workflow-suite
python audit_workflow_similarity.py
```

Preparation downloads pinned data files and creates the frozen splits. It refuses an existing per-task manifest. To fetch only one task, use `--dataset injection`, `sms`, `emotion`, `counterfactual` or `massive`; the flag can be repeated. Keep the included `train/workflow-suite/prior_holdout_hashes.json`: these text-free hashes preserve exclusions for previously benchmarked examples, including private examples that are not distributed.

| Task | Source | Training / development / test |
|---|---|---:|
| Prompt-injection detection | deepset/prompt-injections | 446 / 100 / 116 |
| SMS spam | ucirvine/sms_spam | 1,000 / 200 / 1,000 |
| Emotion | dair-ai/emotion | 1,000 / 200 / 1,000 |
| Product-review counterfactuals | SetFit/amazon_counterfactual, English | 1,000 / 200 / 670 |
| Assistant request routing | AmazonScience/massive, en-US | 1,000 / 200 / 1,000 |

The analyzer checks reconstructed split and manifest hashes against the saved predictions. It then recomputes per-class metrics and paired confidence intervals. It selects the lexicographically latest complete, validated run for each task/target; it does not select by score. The additional similarity audit reports a post-hoc sensitivity analysis without replacing primary test results.

SMS uses a custom split from its single source pool. Emotion has distant labels; it is not wholly human-labelled. Research/noncommercial restrictions and conflicting mirror metadata are documented in the [workflow protocol](workflow-suite-protocol.md). Dataset scripts are not executed.

## Rebuild the report after rescoring

```sh
python build_full_report.py
```

The publication builder preserves the historical narrative and curated-pilot aggregates while rebuilding charts and moderation content from analyses. This is suitable for replaying the published experiment. For a new experiment, review and update its narrative and preserved tables before publishing: rebuilding alone does not rewrite every conclusion for new scores. Do not use the older `build_public_report.py` as a complete public replay command; its pilot section expects private raw results that are deliberately absent here.

## Install the model runtime and run the included tests

```sh
python -m pip install 'torch==2.14.0' 'transformers==5.17.0' 'safetensors==0.8.0' 'fastapi==0.141.1' 'uvicorn==0.53.0'
python -m pip install 'laya @ git+https://github.com/NandhaKishorM/laya.git@6a5819129eb220570792e417e49723d697efd76f'
python -m unittest test_public_workflow test_public_analysis test_workflow_suite test_workflow_analysis test_suite_training
```

The source checkout reports Laya 0.3.3. Recorded module hashes in the moderation runtime match that pinned upstream revision. A matching seed does not ensure identical GPU arithmetic or byte-identical weights on another device. The moderation guide lists its additional tests.

## Fresh Banking77 training and evaluation

Use a separate clone for new runs. After preparing its data, archive its copied results so new and published runs do not mix:

```sh
mv results ../saved-results
mkdir results
export JEVTEST_TELEMETRY_ROOT="$(pwd)/../local-telemetry"
python train_public_laya.py --data ../train/public-banking77 --out ../models/laya-banking77-gold-1k --seed 7 --epochs 3 --micro-batch 4 --accum 16
```

The trainer starts from the pinned `convaiinnovations/laya` revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, verifies data hashes, uses three fixed epochs and saves the final checkpoint. It refuses to overwrite a checkpoint. Default padding is disabled for these original public workflows. The current trainer also releases unused Metal cache each optimizer step; that runtime setting is recorded in the new checkpoint rather than treated as a byte-identical replay of an older training run.

Create the matched-budget base alias from the same downloaded snapshot:

```sh
python - <<'PY'
import json
from pathlib import Path
fine = Path('../models/laya-banking77-gold-1k')
snapshot = Path(json.loads((fine/'rl_agent_config.json').read_text())['finetune']['base_snapshot'])
base = Path('../models/laya-english-wide')
base.mkdir(parents=True, exist_ok=False)
for name in ('encoder', 'tokenizer', 'model.safetensors'):
    (base/name).symlink_to((snapshot/name).resolve())
cfg = json.loads((snapshot/'rl_agent_config.json').read_text())
cfg.update(max_len=1024, head_max_len=768)
(base/'rl_agent_config.json').write_text(json.dumps(cfg, indent=1))
PY
```

Start the local server on an available port, without stopping another service:

```sh
LAYA_PRELOAD='' LAYA_EXTRA='laya-english-wide=../models/laya-english-wide,laya-banking77-gold-1k=../models/laya-banking77-gold-1k' uvicorn serve_laya:app --host 127.0.0.1 --port 8790
```

In a second terminal with the environment activated, also from `bench/`:

```sh
export LAYA_BASE_URL=http://127.0.0.1:8790
export JEVTEST_TELEMETRY_ROOT="$(pwd)/../local-telemetry"
python public_workflow.py evaluate --target laya:laya-english-wide
python public_workflow.py evaluate --target laya:laya-banking77-gold-1k
python public_workflow.py evaluate --target laya:laya-english-wide --cases ../train/public-banking77/development.jsonl
python public_workflow.py evaluate --target laya:laya-banking77-gold-1k --cases ../train/public-banking77/development.jsonl
```

Use the same chosen port in the server command and `LAYA_BASE_URL`. Stop your own server before starting the next local GPU experiment. These commands reproduce the two matched-budget conditions; the historical stock-budget Laya control remains separately identified in the report.

For a fresh Jev comparison, set `OPENROUTER_API_KEY` securely in your shell or local ignored environment file, then run:

```sh
python public_workflow.py evaluate --target jev
python analyze_public_study.py
```

This is a paid hosted call, using the historical client's OpenRouter Decisions endpoint and requested model `typesafe/jev-1.13`. Availability and provider behavior can change; saved results retain resolved model identities. Do not substitute a newer model and describe it as the original run. Keep local telemetry private: it contains complete source requests and responses.

## Fresh workflow training and evaluation

After constructing the same matched-budget base alias and preparing the five datasets, the local lane runs base inference, independent fine-tuning and fine-tuned inference sequentially for each task:

```sh
python run_workflow_suite.py --lane local
```

The hosted lane requires your credentials and incurs charges:

```sh
python run_workflow_suite.py --lane jev
python analyze_workflow_suite.py
python audit_workflow_similarity.py
```

The lanes refuse an existing status file under `results/workflow-suite/`; use the separate working copy with its original results archived. Each checkpoint starts independently from the pinned base, rather than continuing from Banking77 or another task. The lane records commands and logs. Run one local GPU workload at a time.

For one task, the equivalent explicit commands are:

```sh
python evaluate_workflow_suite.py --data ../train/workflow-suite/injection --target laya:base --checkpoint ../models/laya-english-wide
python train_public_laya.py --data ../train/workflow-suite/injection --out ../models/laya-suite-injection-s7 --seed 7
python evaluate_workflow_suite.py --data ../train/workflow-suite/injection --target laya:fine --checkpoint ../models/laya-suite-injection-s7
python evaluate_workflow_suite.py --data ../train/workflow-suite/injection --target jev
```

Replace `injection` with another task slug as needed. Unlike the moderation evaluator's fixed filenames, these evaluators create timestamped result files. Keep distinct experiments in separate result directories or working copies so the analyzer's latest-run rule does not silently combine different experiments.

## Moderation and the curated pilot

Follow [moderation reproduction](reproduce.md) for Aegis training, six-source evaluation, WildGuard access conditions, and harmful-class metrics. The [Jev moderation addendum](moderation-jev-addendum.md) records the matched hosted extension and its completion state. Advertised OpenAI/Mistral scores are separately attributed references; the repository does not run those models to recreate their advertised numbers.

The curated pilot combines existing workflow data and generated examples, with labels supplied by a local Qwen model. Its fixed 145 decision questions across 100 cases yielded 71.7% base accuracy, 82.8% after the smaller fine-tune and 91.7% after the larger one; the earlier Jev measurement was 97.2%. The larger pool held 2,646 unique requests, with 13 unusable answer sets skipped. Its old development split separated questions rather than complete requests, so it is not an independent development estimate. These are exploratory aggregate observations, not the public test protocol. No raw pilot data, private Jev predictions or pilot training command is distributed.

No Laya model in this study was trained on Jev answers. Public workflow labels retain source provenance; moderation labels retain Aegis human/model provenance. Match the metric to the task: Banking77 accuracy, workflow macro-F1, moderation harmful-class F1 and harmless-request false-positive rate. None alone establishes overall model superiority or deployment readiness.
