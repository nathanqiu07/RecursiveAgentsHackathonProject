import os
import re
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
import httpx

from . import GPU, MODEL
from .config import digest, load_config
from .evaluate import hardware, identity
from .queue import Queue
from .pool import baseline_state, inspect_pool


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repository: str = Field(pattern=r"^https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+/?$", max_length=2000)
    commit: str = Field(pattern=r"^(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})$")
    deployment: str = Field(pattern=r"^[A-Za-z0-9_-]+$", max_length=100)
    model: str = Field(max_length=256)
    gpu: str = Field(max_length=100)
    precision: str = Field(max_length=30)
    notes: str = Field(default="", max_length=6000)


class Evaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    submission: Payload
    submission_id: str = Field(min_length=1, max_length=100)
    benchmark_id: str = Field(min_length=1, max_length=100)
    base_checkpoint_url: str = Field(pattern=r"^https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+/(?:commit|tree)/(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})/?$", max_length=2000)


def auth(authorization: str = Header(default="")):
    key = os.environ.get("NEMOTRON_VERIFIER_API_KEY")
    if not key or not secrets.compare_digest(authorization.encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "Verifier API key required")


@asynccontextmanager
async def lifespan(app):
    app.state.queue = Queue(os.environ.get("NEMOTRON_VERIFIER_DB", "data/verifier.sqlite"))
    yield
    app.state.queue.db.close()


app = FastAPI(title="Nemotron RTX A5000 mini verifier", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
def invalid(_request, _error):
    return JSONResponse({"error": "Invalid evaluation request"}, status_code=422)


def setup():
    try:
        config = load_config()
        if config.get("nodes"):
            fingerprint, _nodes = inspect_pool(config)
            return config, fingerprint
        return config, identity(config, hardware(config))
    except (OSError, ValueError, KeyError, TypeError, httpx.HTTPError):
        raise HTTPException(503, "Configure the pinned suite, deployments, and remote RTX A5000 hardware first") from None


@app.get("/health")
def health(request: Request):
    try:
        config, fingerprint = setup()
        if config.get("nodes"):
            _, states = inspect_pool(config)
            _entries, complete, baseline_status = baseline_state(request.app.state.queue, states)
            available = sum(node["status"] != "offline" for node in states)
            return {"status": "ok", "ready": available > 0, "baseline_ready": complete,
                    "baseline_status": baseline_status, "suite": "mini", "configured_nodes": len(states),
                    "online_nodes": available, "message": "Local GPU job coordinator",
                    "runtime_target_seconds": [600, 1200], "runtime_target_is_advisory": True}
        ready, message = True, "Hardware and evaluation suite configured"
        baseline_job = request.app.state.queue.baseline_job(fingerprint)
        baseline_status = baseline_job["status"] if baseline_job else "unmeasured"
    except Exception:
        ready, message = False, "RTX A5000 deployment and evaluation suite setup required"
        baseline_status = "unavailable"
    return {"status": "ok", "ready": ready, "baseline_ready": baseline_status == "completed",
            "baseline_status": baseline_status, "suite": "mini", "message": message}


@app.post("/baseline", status_code=202, dependencies=[Depends(auth)])
def baseline(request: Request):
    config, fingerprint = setup()
    queue = request.app.state.queue
    if config.get("nodes"):
        fingerprint, states = inspect_pool(config)
        # Baselines must be measured independently on every physical GPU in the pool.
        if any(node["identity"] is None for node in states):
            raise HTTPException(503, "Connect every configured GPU node before measuring the pool baseline")
        jobs = [queue.submit_baseline(node["identity"], config["reference"], node["id"], fingerprint)
                for node in states]
        return {**jobs[0], "jobs": jobs, "pool_identity": fingerprint}
    # One immutable reference per suite/hardware/runtime identity.
    result = queue.submit_baseline(fingerprint, config["reference"])
    return result


@app.get("/baseline", dependencies=[Depends(auth)])
def get_baseline(request: Request):
    config, fingerprint = setup()
    if config.get("nodes"):
        _, states = inspect_pool(config)
        entries, complete, status = baseline_state(request.app.state.queue, states)
        # UI summary is one actual node's measurement; full per-node records stay available.
        return {"metrics": entries[0]["metrics"] if complete else None, "node_baselines": entries,
                "pool_identity": fingerprint, "status": status}
    job = request.app.state.queue.baseline_job(fingerprint)
    return {"metrics": request.app.state.queue.baseline(fingerprint),
            "evaluation": {"id": job["id"], "status": job["status"]} if job else None}


@app.post("/evaluate", status_code=202, dependencies=[Depends(auth)])
def evaluate(body: Evaluation, request: Request, idempotency_key: str = Header(default="")):
    config, fingerprint = setup()
    payload = body.submission
    if payload.model != MODEL or payload.gpu != GPU or payload.precision != "NVFP4":
        raise HTTPException(422, "Wrong model, GPU, or precision")
    deployment = config["deployments"].get(payload.deployment)
    if not deployment or any(getattr(payload, field) != deployment[field] for field in ["repository", "commit"]):
        raise HTTPException(422, "Deploy the pinned source and register it with the verifier first")
    if config.get("nodes"):
        fingerprint, states = inspect_pool(config)
        _entries, complete, _status = baseline_state(request.app.state.queue, states)
        if not complete:
            raise HTTPException(409, "Measure a reference baseline on each GPU node before submitting candidates")
    elif not request.app.state.queue.baseline(fingerprint):
        raise HTTPException(409, "Measure the reference baseline before submitting candidates")
    if idempotency_key != body.submission_id:
        raise HTTPException(400, "Use the submission ID as Idempotency-Key")
    try:
        request.app.state.queue.bind_benchmark(body.benchmark_id, fingerprint)
        return request.app.state.queue.submit(body.submission_id, "candidate", {
            **body.model_dump(), "deployment": payload.deployment, "identity": fingerprint,
            "deployment_identity": digest(deployment),
            **({"pool_identity": fingerprint} if config.get("nodes") else {}),
        })
    except ValueError:
        raise HTTPException(409, "Submission ID is reused or benchmark hardware/suite/reference changed; use a new benchmark") from None


@app.get("/evaluations/{job_id}", dependencies=[Depends(auth)])
def get_evaluation(job_id: str, request: Request):
    if not re.fullmatch(r"[a-f0-9-]{36}", job_id):
        raise HTTPException(404, "Evaluation not found")
    result = request.app.state.queue.get(job_id)
    if result is None:
        raise HTTPException(404, "Evaluation not found")
    return result


@app.get("/nodes", dependencies=[Depends(auth)])
def get_nodes():
    config, fingerprint = setup()
    if not config.get("nodes"):
        return {"pool_identity": fingerprint, "nodes": []}
    fingerprint, states = inspect_pool(config)
    return {"pool_identity": fingerprint, "nodes": [
        {key: node[key] for key in ("id", "status", "gpu_uuid")} for node in states]}


@app.get("/evaluations", dependencies=[Depends(auth)])
def list_evaluations(request: Request):
    return {"evaluations": request.app.state.queue.list()}


@app.post("/evaluations/{job_id}/cancel", status_code=202, dependencies=[Depends(auth)])
def cancel_evaluation(job_id: str, request: Request):
    result = request.app.state.queue.cancel(job_id)
    if result is None:
        raise HTTPException(404, "Evaluation not found")
    return result
