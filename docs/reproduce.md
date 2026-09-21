---
title: Reproduce the moderation study
order: 1
---

# Reproduce the moderation study

There are two useful levels of reproduction: check the published scores using saved predictions, or download the base model and repeat training and inference. The first is much faster and needs no GPU. Both require the original data to verify example IDs, labels and split hashes; moderation dataset text is deliberately excluded from this repository.

The publication branch is `research/full-report`. [`PUBLIC_SHA256SUMS.json`](../PUBLIC_SHA256SUMS.json) records the files in the public snapshot; historical evidence retains its own recorded hashes.

## 1. Set up a local copy

```sh
git clone --branch research/full-report https://github.com/Alexander-Ollman/laya-ft.git
cd laya-ft
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy==2.5.3' 'pyarrow==25.0.1' 'httpx==0.28.1' 'huggingface-hub==1.32.0'
```

These are versions recorded for this experiment, not a claim about the latest releases. The original runtime was Python 3.12.12 on macOS 15.6, ARM64. See [`runtime.json`](../bench/results/moderation-study/runtime.json) for the complete record. Platform-specific installation or availability may differ.

Run the commands below from `bench/`. The scripts locate data and results relative to their own repository root.

```sh
cd bench
```

## 2. Download and verify the source data

Accept AI2's access conditions for [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) using your own Hugging Face account, then authenticate locally:

```sh
hf auth login
python fetch_moderation_sources.py
python prepare_moderation_study.py
python restore_moderation_evidence.py --data ../train/moderation-study --evidence ../evidence
```

The fetcher downloads pinned files and checks their SHA-256 hashes. It does not execute dataset-provided Python code. The preparer recreates the nested training samples, development sample and all nine test views. The restore step verifies source bytes, split bytes and metadata before restoring the recorded top-level manifest's JSON ordering. It does not accept changed examples or labels.

Leave the downloaded data local. In particular, access to a gated dataset is not permission to redistribute its text. The sources and their recorded revisions are listed in [`moderation_sources.json`](../bench/moderation_sources.json).

## 3. Recompute the published scores and report

```sh
python analyze_moderation.py
python build_moderation_report.py
```

Open `../report/laya-moderation-study.html`. The analyzer joins the saved predictions to the verified examples, checks identities and hashes, and recomputes scores and paired bootstrap intervals. It overwrites the derived `analysis.json`; it does not change the saved predictions or make model calls.

To regenerate the public layout and its bar charts, install the plotting version used for publication and run:

```sh
python -m pip install 'matplotlib==3.11.2'
python build_public_site.py
```

This writes `../moderation.html` and downloadable charts under `../assets/`. You can rebuild that presentation from the included analysis and evidence manifests without downloading data; independent rescoring still requires steps 1–3.

Historical filesystem paths in the evidence identify the original run; they are not portable checkpoint downloads. The analysis records whether checkpoint bytes are locally available. Without the original weights, it verifies consistent recorded model identities, rather than independently rehashing unavailable weights. The recorded checkpoint configurations and hashes are in [`checkpoint_manifest.json`](../bench/results/moderation-study/checkpoint_manifest.json).

## 4. Install the training runtime and run tests

For local model execution and the complete included test suite, the experiment recorded these additional versions:

```sh
python -m pip install 'torch==2.14.0' 'transformers==5.17.0' 'safetensors==0.8.0'
python -m pip install 'laya @ git+https://github.com/NandhaKishorM/laya.git@6a5819129eb220570792e417e49723d697efd76f'
python -m unittest test_moderation_metrics test_moderation_preparation test_moderation_analysis test_moderation_padding test_moderation_replay
```

The recorded Laya installation came from the source checkout above, reporting version 0.3.3. `runtime.json` records hashes for the Laya modules used, which let you check the installed source independently of the version label. Saved-prediction scoring does not depend on loading Laya.

The original model runs used Apple's Metal backend. Other hardware, package builds or kernels can produce different floating-point results even with the same seed. Repeating the experiment means following the same data and training method; it does not promise byte-identical new weights.

## 5. Repeat training in a separate working copy

Keep the published predictions intact. Use a second clone, repeat setup and source preparation there, and move its copied results aside before any new inference. From that second clone's `bench/` directory:

```sh
mv results/moderation-study results/published-moderation-study
mkdir -p results/moderation-study
export JEVTEST_TELEMETRY_ROOT="$(pwd)/../local-telemetry"
```

The telemetry variable is a legacy code name; it only selects a local archive directory. New inference telemetry includes source text, so keep that directory private.

Download the pinned model and construct a base configuration with the same token budgets. The `laya-english-wide` directory name is retained because the tokenizer preflight script expects it.

```sh
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path.cwd().parent
source = root / 'models/laya-english-wide'
snapshot_download(
    'convaiinnovations/laya',
    revision='1c5edc17a7acd8701df6fc341c0d179f1c62c982',
    allow_patterns=['rl_agent_config.json', 'model.safetensors', 'encoder/*', 'tokenizer/*'],
    local_dir=source,
)
base = root / 'models/laya-moderation-base-2048'
base.mkdir(parents=True, exist_ok=False)
for name in ('encoder', 'tokenizer', 'model.safetensors'):
    (base / name).symlink_to((source / name).resolve())
cfg = json.loads((source / 'rl_agent_config.json').read_text())
cfg.update(max_len=2048, head_max_len=256)
(base / 'rl_agent_config.json').write_text(json.dumps(cfg, indent=1) + '\n')
PY
python preflight_moderation.py
```

Train both budgets independently from the pinned base, one GPU workload at a time:

```sh
python train_public_laya.py --data ../train/moderation-study/aegis_1000 --out ../models/laya-moderation-aegis-1000-s7 --base convaiinnovations/laya --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 --seed 7 --epochs 3 --micro-batch 4 --accum 16 --pad-to-multiple 128 --mps-cache-clear-interval 1
python train_public_laya.py --data ../train/moderation-study/aegis_5000 --out ../models/laya-moderation-aegis-5000-s7 --base convaiinnovations/laya --revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982 --seed 7 --epochs 3 --micro-batch 4 --accum 16 --pad-to-multiple 128 --mps-cache-clear-interval 1
```

The trainer refuses an existing output directory. It uses three fixed epochs, an effective batch of 64, encoder/head learning rates of 2.5e-5/1e-4, and the final checkpoint. Neither test-set results nor development scores select a different checkpoint. The study protocol explains masked padding and Metal cache release, which resolved excessive memory retention during the original attempts.

## 6. Evaluate the new checkpoints

```sh
python evaluate_moderation.py --name base --checkpoint ../models/laya-moderation-base-2048
python evaluate_moderation.py --name fine1000 --checkpoint ../models/laya-moderation-aegis-1000-s7
python evaluate_moderation.py --name fine5000 --checkpoint ../models/laya-moderation-aegis-5000-s7
python analyze_moderation.py
python build_moderation_report.py
```

Each evaluation command runs all nine tasks serially. There is no results-directory CLI flag: it always writes to `bench/results/moderation-study/` in that working copy, using fixed `<task>_<name>.json` filenames. Existing files cause an error. Use a fresh working copy or archive the complete previous results directory and regenerate preflight before starting a new run; do not mix predictions from different runs. The historical `run_moderation_study.py` also refuses an existing status file. The explicit commands above avoid restarting its completed historical lane.

All three checkpoints must use a 2,048-token sequence budget and 256-token question/option budget. Preflight identifies truncated inputs; the original study shortened eight WildGuard response test inputs, plus one or three training inputs. No question or answer option was truncated.

## Interpret the results

The headline metric is F1 for the harmful class, not macro-F1. Always read it alongside harmless-request false alarms. Aegis is the training source; the other datasets test transfer under different definitions of harmful content. OpenAI's known-binary subset has 859 examples because missing category labels are treated as unknown, not safe. Its 9,298 known category decisions are a separate task. XSTest here judges prompts, which differs from published response/refusal benchmarks.

Four prespecified Aegis comparisons use paired bootstrap intervals at 98.75%; secondary comparisons use descriptive 95% intervals. These intervals describe sampling uncertainty, not variation across training seeds. We used only one seed. Published OpenAI and Mistral numbers remain external references, with their source and policy differences documented in the [reference notes](moderation-reference-notes.md).
