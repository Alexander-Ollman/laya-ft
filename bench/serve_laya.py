"""Serve Laya locally behind the TypeSafe System One HTTP shape.

POST /v1/systemone  {"state": ..., "model": ..., "questions": {...}}
GET  /v1/models
GET  /healthz

The request and the response match https://docs.typesafe.ai/api, so one client can call
Laya and Jev and change only the base URL and the model name.

Run from this directory (the repo clone at ../laya shadows the installed package otherwise):

    ../.venv/bin/uvicorn serve_laya:app --host 127.0.0.1 --port 8790
"""
import os
import threading
import time
from typing import Any, Dict, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import laya
from laya import Router

from telemetry import Recorder

# model name in the request -> Router argument. "laya-auto" lets the Router pick by language.
MODELS = {
    "laya-auto": None,
    "laya-english": "english",
    "laya-multilingual": "multilingual",
    "laya-typed-decisions": "typed-decisions",
}
PRELOAD = [n for n in os.environ.get("LAYA_PRELOAD", "english").split(",") if n]
# Local fine-tuned checkpoints: LAYA_EXTRA="laya-era-distill=/path/to/dir,other=/path". They bypass the Router.
EXTRA_PATHS = dict(item.split("=", 1) for item in os.environ.get("LAYA_EXTRA", "").split(",") if "=" in item)
EXTRA = {}

app = FastAPI(title="laya-local", version=laya.__version__)
router = Router(max_loaded=3, device=os.environ.get("LAYA_DEVICE"))
# One forward pass at a time: the checkpoints share one device and the Router LRU is not thread safe.
lock = threading.Lock()
# Server-side telemetry: captures calls from third-party harnesses that we cannot instrument.
recorder = {"active": None}


class SystemOneRequest(BaseModel):
    state: Union[str, Dict[str, Any], list]
    model: str = "laya-auto"
    questions: Dict[str, Dict[str, Any]]


@app.on_event("startup")
def _preload():
    if PRELOAD:
        router.preload(PRELOAD)
    for name, path in EXTRA_PATHS.items():
        EXTRA[name] = laya.load(path, device=os.environ.get("LAYA_DEVICE"))
    # Laya's README fix for 50+ options: raise the option budget (head_max_len) and the sequence limit.
    # Applies to preloaded checkpoints only.
    for key in ("head_max_len", "max_len"):
        if os.environ.get("LAYA_" + key.upper()):
            for agent in router._agents.values():
                agent.cfg[key] = int(os.environ["LAYA_" + key.upper()])


@app.get("/healthz")
def healthz():
    return {"ok": True, "loaded": router.loaded, "version": laya.__version__}


@app.get("/v1/models")
def models():
    return {"models": [{"name": n, "description": "Laya checkpoint: %s" % (v or "routed by language"),
                        "release_date": ""} for n, v in MODELS.items()]}


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    if req.model in EXTRA:
        t0 = time.perf_counter()
        with lock:
            out = EXTRA[req.model].system_one(req.state, req.questions)
        out["model"], out["routing"] = req.model, {"model": req.model, "repo": EXTRA_PATHS[req.model], "reason": "explicit"}
        out["server_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if recorder["active"]:
            recorder["active"].record("laya", req.model, req.state, req.questions, out["answers"],
                                      model_resolved=EXTRA_PATHS[req.model], endpoint="local:/v1/systemone",
                                      latency_ms=out["server_ms"], usage=out["usage"])
        return out
    if req.model not in MODELS:
        # A third-party harness may hard-code a Jev model name. LAYA_MODEL_ALIAS maps every unknown name to one checkpoint.
        if os.environ.get("LAYA_MODEL_ALIAS") not in MODELS:
            raise HTTPException(404, "unknown model %r; use one of %s" % (req.model, sorted(MODELS)))
    name = req.model if req.model in MODELS else os.environ["LAYA_MODEL_ALIAS"]
    if not req.questions:
        raise HTTPException(422, "questions must not be empty")
    t0 = time.perf_counter()
    try:
        with lock:
            out = router.predict(req.state, req.questions, model=MODELS[name])
    except (KeyError, ValueError) as e:
        if recorder["active"]:
            recorder["active"].record("laya", req.model, req.state, req.questions, {}, endpoint="local:/v1/systemone",
                                      error=str(e)[:300])
        raise HTTPException(422, str(e))
    out["model"] = name
    out["server_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if recorder["active"]:
        recorder["active"].record("laya", req.model, req.state, req.questions, out["answers"],
                                  model_resolved=out["routing"]["repo"], endpoint="local:/v1/systemone",
                                  latency_ms=out["server_ms"], usage=out["usage"])
    return out


class TelemetryStart(BaseModel):
    run_id: str
    source: str
    meta: Dict[str, Any] = {}


@app.post("/telemetry/start")
def telemetry_start(req: TelemetryStart):
    if recorder["active"]:
        recorder["active"].close(outcome="superseded by %s" % req.run_id)
    recorder["active"] = Recorder(req.run_id, req.source, recorder="server", meta=req.meta)
    return {"dir": str(recorder["active"].dir)}


class TelemetryStop(BaseModel):
    outcome: str = ""
    attach: list = []


@app.post("/telemetry/stop")
def telemetry_stop(req: TelemetryStop):
    rec, recorder["active"] = recorder["active"], None
    if not rec:
        raise HTTPException(409, "no active telemetry run")
    rec.attach(*req.attach)
    return rec.close(outcome=req.outcome)
