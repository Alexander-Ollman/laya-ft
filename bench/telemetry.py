"""Structured telemetry for System One runs. One record for each request, full answers included.

Layout (owner rule 2026-09-11: every run is archived with a manifest and an index line):

    ~/era-telemetry/<date>/<run_id>/records.jsonl   one line per request, schema below
    ~/era-telemetry/<date>/<run_id>/manifest.json   run summary, written by close()
    ~/era-telemetry/<date>/<run_id>/artifacts/      result files of a third-party harness
    ~/era-telemetry/INDEX.jsonl                     one line per run

Record schema `sysone.telemetry.v1`:

    ts, run_id, source, recorder        when, which run, which benchmark, client or server side
    case_id, split                      benchmark row id. split: development | heldout | unsplit | null
    provider, model_requested, model_resolved, endpoint
    state, questions                    the exact request content
    answers                             the exact response answers, with every probability
    expected, label_source              gold answers when the recorder knows them
    latency_ms, usage, error
    state_sha256, request_sha256        join keys between client records, server records, and harness files
    train_ok, train_note                may the ANSWERS of this record train a model?

`train_ok` is about the model outputs only. Gold labels have their own license (see label_source).
TypeSafe's Master Customer Agreement, clause (b), forbids use of Jev output for model distillation,
so every Jev record is train_ok=false. Hosted LLM outputs are false until someone reads that provider's terms.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

SCHEMA = "sysone.telemetry.v1"
ROOT = Path(os.environ.get("JEVTEST_TELEMETRY_ROOT",
                          os.environ.get("ERA_TELEMETRY_ROOT", "~/era-telemetry"))).expanduser()

TRAIN_POLICY = {
    "laya": (True, "Apache 2.0 open weights, local inference"),
    "jev": (False, "TypeSafe MCA clause (b): no model distillation from Output"),
    "jev-direct": (False, "TypeSafe MCA clause (b): no model distillation from Output"),
    "local": (True, "open-weight teacher under Apache 2.0, local inference. Confirm the license of each model"),
    "openrouter": (False, "hosted LLM output: read the provider terms before any training use"),
}


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class Recorder:
    def __init__(self, run_id, source, recorder="client", meta=None):
        self.run_id, self.source, self.recorder = run_id, source, recorder
        self.date = time.strftime("%Y-%m-%d")
        self.dir = ROOT / self.date / run_id
        (self.dir / "artifacts").mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "records.jsonl"
        self.meta = dict(meta or {})
        self.started = time.time()
        self.n = self.errors = 0
        self.models = set()
        self.lock = threading.Lock()

    def record(self, provider, model_requested, state, questions, answers, *, model_resolved=None, endpoint=None,
               case_id=None, split=None, expected=None, label_source=None, latency_ms=None, usage=None, error=None):
        ok, note = TRAIN_POLICY.get(provider, (False, "unknown provider"))
        row = {
            "schema": SCHEMA, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run_id": self.run_id,
            "source": self.source, "recorder": self.recorder, "case_id": case_id, "split": split,
            "provider": provider, "model_requested": model_requested, "model_resolved": model_resolved,
            "endpoint": endpoint, "state": state, "questions": questions, "answers": answers,
            "expected": expected, "label_source": label_source,
            "latency_ms": None if latency_ms is None else round(latency_ms, 1), "usage": usage or {}, "error": error,
            "state_sha256": sha(state), "request_sha256": sha({"state": state, "questions": questions}),
            "train_ok": bool(ok and not error), "train_note": note,
        }
        with self.lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.n += 1
            self.errors += bool(error)
            self.models.add(model_resolved or model_requested)

    def attach(self, *paths):
        """Copy harness result files next to the records."""
        import shutil
        for p in map(Path, paths):
            if p.is_file():
                shutil.copy2(p, self.dir / "artifacts" / p.name)
            elif p.is_dir():
                shutil.copytree(p, self.dir / "artifacts" / p.name, dirs_exist_ok=True)

    def close(self, outcome=None):
        size = sum(f.stat().st_size for f in self.dir.rglob("*") if f.is_file())
        manifest = {
            "schema": SCHEMA, "date": self.date, "run_id": self.run_id, "kind": "sysone-bench",
            "source": self.source, "recorder": self.recorder, "models": sorted(m for m in self.models if m),
            "outcome": outcome, "prompt_records": self.n, "errors": self.errors,
            "seconds": round(time.time() - self.started, 1), "meta": self.meta,
            # keys below keep the line compatible with the era-computer-agent index
            "seats": None, "executor_turns": None, "bytes": size, "path": "%s/%s" % (self.date, self.run_id),
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        with (ROOT / "INDEX.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: manifest[k] for k in ("date", "run_id", "kind", "source", "seats", "models", "outcome",
                                                         "executor_turns", "prompt_records", "bytes", "path")}) + "\n")
        return manifest
