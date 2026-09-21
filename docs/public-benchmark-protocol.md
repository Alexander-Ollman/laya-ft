---
title: Public decision-model benchmark protocol
order: 10
---

# Public decision-model benchmark protocol

Scope agreed 2026-09-19: evaluate Jev and Laya independently of Era, and report reproducible fine-tuning results for common AI workflows. Public benchmarks are the primary evidence. The previous private routing experiment remains a provenance-labelled continuation, not an independently reproducible public dataset.

## Governance exception

The project-local governance files are absent. The inherited constitution and directives were read at `~/.era/memory/`. The owner explicitly authorized continuing without Git on 2026-09-19. No initialization, parent Git repair, branch creation, or commits are performed. Existing telemetry continues to use its configured archive directory; that historical directory name does not make the public experiments depend on Era data or services.

For independent installations, set `JEVTEST_TELEMETRY_ROOT` to any archive directory. It takes precedence over the older `ERA_TELEMETRY_ROOT` alias. If neither is set, existing local runs retain `~/era-telemetry` for continuity. The archive is ordinary local JSONL and has no dependency on an Era service.

## Frozen Banking77 experiment

Use [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77), a customer-support intent classification dataset with 77 labels. Its Hugging Face loader references the author's original CSV files. Download those files at commit `57ec275d8078af65b7731c2a98be812d844a6d6b`, matching the original dataset rather than a processed mirror. Attribute Casanueva et al., [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807), under CC BY 4.0.

Split before modelling. Remove normalized training/test overlap, duplicate training requests, and conflicting-label duplicates. Using seed 7, select 13 unique training requests and 10 development requests per class: 1,001 training and 770 development requests. Preserve all 3,080 official test requests. Test labels never select options, prompts, hyperparameters, epochs, or calibration. Public pretraining contamination cannot be ruled out for any model.

Every request presents all 77 labels with null descriptions, so each label is rendered once. A tokenizer-only preflight found redundant label descriptions needed 894 option tokens, exceeding the proposed 768-token budget. Before any public inference or training, that draft dataset was archived and regenerated with null descriptions. Primary comparison: base Laya and supervised Laya with identical sequence limit 1,024 and option budget 768, three fixed epochs, label-smoothed cross entropy, encoder learning rate 2.5e-5 and head rate 1e-4. Both get the same input wording. Report stock-budget Laya separately as an architectural control. No claims of reproducing RLCD training. The public fine-tune starts from base Laya, not the private distilled checkpoint. Original human dataset labels provide the supervision; Qwen is allowed for a separately identified teacher experiment, never Jev output.

The final epoch is evaluated regardless of its result. Primary metric: accuracy over every official test request, counting API failures as errors. Also report macro-F1, full multiclass Brier score, negative log likelihood, ECE, p50/p95 client latency, and API cost. Local API cost of zero excludes hardware, power, and training cost. Wilson intervals assume independent rows; duplicates require a grouped paired bootstrap for between-model claims. Development data alone may fit a temperature; no calibrated-confidence claim is permitted until that step is complete.

The scorer validates complete finite probability maps. Distributions within 2% of unit mass are normalized to correct response rounding; other maps are excluded from probability metrics, not from accuracy. ECE uses the probability of the returned choice. Missing API charges are unknown, with the reported subtotal and missing-row count preserved. Rescore any earlier evaluation with these same rules before comparison. The trainer resolves one base snapshot before loading and records its path/revision in the saved configuration and archive.

The public trainer defaults to base revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`; `--revision` overrides it for a deliberately separate experiment. This cached ModernBERT encoder supports 8,192 positions, so the experiment's 1,024-token limit is within its configured capacity.

## Commands

Run Python from `bench/` to avoid the sibling clone shadowing the installed Laya package.

```bash
../.venv/bin/python public_workflow.py prepare
../.venv/bin/python -m unittest test_public_workflow
../.venv/bin/python train_public_laya.py --out ../models/laya-banking77-gold-1k
../.venv/bin/python public_workflow.py evaluate --target laya:laya-banking77-gold-1k
```

Preparation refuses to replace an existing frozen dataset. Training verifies split hashes and cross-split request separation, retains final partial batches, and refuses to overwrite a checkpoint. Dataset files and the experiment manifest live in `train/public-banking77/`. Serving requires registering the trained checkpoint via `LAYA_EXTRA`. GPU training, Qwen labelling, and GPU evaluation run sequentially. Port 8791 remains reserved for the owner's process.

`continue_public_study.py` sequences the already-authorized continuation and public runs; `results/public_study_status.json` records each stage. It is a session-specific runner and refuses a second start over existing status. `build_public_report.py` generates `report/jev-vs-laya.html` from saved results and preserves the earlier report in `report/archive/`. `run_bench.py --source <protocol>` gives external controls their own telemetry identity and respects explicit request split metadata. After public evaluation, `replay_public_controls.py` replays the archived 100 news and 100 emotion requests to check for loss of general classification ability. Their wording and labels are unchanged and their text must not overlap public training requests.

`analyze_public_study.py` rescales valid rounded probability maps consistently, validates all saved test IDs/labels/text groups, and generates `results/public_study_analysis.json`. Its paired confidence interval resamples normalized-text groups 5,000 times with seed 7. Temperature fitting uses development NLL only; the stored probabilities are rounded, so log-probability scaling is an approximation to exact-logit temperature scaling. The analysis never changes the underlying predictions or checkpoints.

Primary contrast: matched-budget base Laya versus public fine-tuned Laya. Secondary contrast: the same fine-tuned checkpoint versus our freshly measured Jev on identical public requests. Both use the grouped paired bootstrap. The secondary contrast was specified before the public fine-tune's held-out result was available; it is not selected from that resulting score.

### Reproduce from the review package

The ZIP already contains the frozen dataset: skip `prepare`. Install the versions recorded in `bench/results/public_runtime.json`, then run the trainer from `bench/`. After training, create the matched-budget base alias with this Python snippet (it writes a new directory and refuses an existing one):

```python
import json
from pathlib import Path
gold = Path('../models/laya-banking77-gold-1k')
base = Path(json.loads((gold/'rl_agent_config.json').read_text())['finetune']['base_snapshot'])
wide = Path('../models/laya-english-wide')
wide.mkdir()
for name in ('encoder', 'tokenizer', 'model.safetensors'):
    (wide/name).symlink_to(base/name, target_is_directory=(base/name).is_dir())
cfg = json.loads((base/'rl_agent_config.json').read_text())
cfg.update(max_len=1024, head_max_len=768)
(wide/'rl_agent_config.json').write_text(json.dumps(cfg, indent=1))
```

Serve the two public conditions from `bench/`, on an available port. These commands assume 8790 is available and do not stop an existing service:

```bash
LAYA_EXTRA='laya-english-wide=../models/laya-english-wide,laya-banking77-gold-1k=../models/laya-banking77-gold-1k' \
  ../.venv/bin/uvicorn serve_laya:app --host 127.0.0.1 --port 8790
```

In a second terminal, also from `bench/`, evaluate each condition on both splits and replay the bundled controls. The same commands work with the stock alias `laya:laya-english` for the input-budget control; development calibration is only required for the two primary Laya conditions.

```bash
../.venv/bin/python public_workflow.py evaluate --target laya:laya-english-wide
../.venv/bin/python public_workflow.py evaluate --target laya:laya-banking77-gold-1k
../.venv/bin/python public_workflow.py evaluate --target laya:laya-english-wide --cases ../train/public-banking77/development.jsonl
../.venv/bin/python public_workflow.py evaluate --target laya:laya-banking77-gold-1k --cases ../train/public-banking77/development.jsonl
../.venv/bin/python run_bench.py --cases results/public_transfer_controls.jsonl --source public/classification-pilot-controls --warm --target laya:laya-english --target laya:laya-english-wide --target laya:laya-banking77-gold-1k
../.venv/bin/python build_public_report.py
```

Saved Jev outputs can be independently rescored without credentials or new API calls. A fresh hosted run additionally requires the existing client's OpenRouter credentials. The session-specific continuation/replay helpers rely on the original local archive and are not required to reproduce the public experiment from the ZIP.

After all measurements, `package_public_study.py` creates a local ZIP of the report, public data, code, saved predictions and checksums. It refuses to package an incomplete comparison or overwrite an existing ZIP. It does not publish or include credentials, private training requests, or model weights. A checkpoint manifest records weight hashes. A separate telemetry archive captures the package and training logs.

## Benchmark sources and comparability

Discovery source: [awesome-jev](https://github.com/kraayenjon/awesome-jev#benchmarks-and-evaluations). Follow each benchmark's source, protocol, pinned revision and raw results. Published results and newly measured results must be labelled separately; matching task names alone do not establish a paired comparison.

| Source | Workflow | Use in the report |
|---|---|---|
| [ASSAY-001](https://github.com/jourdanlabs/assay-001) | Banking77 / CLINC150 routing and calibration | Public reference with frozen inputs and raw responses; our instruction and state formatting differ, so published scores are contextual references |
| [Jev Spam Eval](https://github.com/bitnovus/jev-spam-eval) | Inbox filtering | Existing 500-email local replication is historical; current upstream experiment has changed and must not be merged with that row |
| [Jev Rerank Bench](https://github.com/anessbelbati/jev-rerank-bench) | RAG relevance | Query-level ranking metrics and paired uncertainty; context and candidate protocol must match for a replication |
| [Agent Failure Benchmark](https://github.com/TokenTrim/jev-agent-failure-benchmark) | Agent failure attribution | Existing 300-trace local replication is historical; current upstream full-corpus results use a different scope |
| [Jev Security Bench](https://github.com/Gaurav-Gosain/jev-sec-bench) | Prompt injection and vulnerable-code screening | Context-sensitive labels and acknowledged synthetic-label noise; do not treat all failures as model defects |

The existing handover records unresolved TypeSafe benchmark-publication terms. Prepare the report and open-model results locally; resolve inclusion of proprietary Jev measurements before public release. Never train on Jev outputs. The [current Master Customer Agreement](https://typesafe.ai/legal/mca), rechecked 2026-09-19, retains the restriction in section 2.3(f). Applicability to this OpenRouter access remains unresolved in the handover. This is a recorded project constraint, not a new legal conclusion.
