import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jobs import (
    ClaimRequest,
    FinishRequest,
    JobSubmission,
    LeaseRequest,
    NodeInfo,
    NodeList,
    NodeRegistration,
    Queue,
    QueueConflict,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.api_key = os.environ.get("VERIFIER_API_KEY")
    app.state.queue = Queue(os.environ.get("VERIFIER_DB_PATH", "data/jobs.sqlite3"))
    yield


app = FastAPI(
    title="Verification API",
    version="0.1.0",
    description="Queue Git checkpoints for execution through their required runner scripts.",
    lifespan=lifespan,
)
bearer = HTTPBearer(auto_error=False)


async def authenticate(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)
):
    key = request.app.state.api_key
    if key and (
        credentials is None
        or not secrets.compare_digest(credentials.credentials.encode(), key.encode())
    ):
        raise HTTPException(401, "Invalid verifier API key", headers={"WWW-Authenticate": "Bearer"})


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Avoid echoing private repository configuration in validation errors.
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
                for err in exc.errors()
            ]
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.exception_handler(QueueConflict)
async def queue_conflict(request: Request, exc: QueueConflict):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.post("/jobs", status_code=202, dependencies=[Depends(authenticate)])
def submit_job(body: JobSubmission, request: Request):
    return request.app.state.queue.submit(body)


@app.get("/jobs/{job_id}", dependencies=[Depends(authenticate)])
def get_job(job_id: str, request: Request):
    job = request.app.state.queue.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(authenticate)])
def cancel_job(job_id: str, request: Request):
    job = request.app.state.queue.cancel(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/worker/claim", dependencies=[Depends(authenticate)])
def claim_job(body: ClaimRequest, request: Request):
    return {"job": request.app.state.queue.claim(body.node_id)}


@app.post("/nodes/register", response_model=NodeInfo, dependencies=[Depends(authenticate)])
def register_node(body: NodeRegistration, request: Request):
    return request.app.state.queue.register_node(body)


@app.get("/nodes", response_model=NodeList, dependencies=[Depends(authenticate)])
def list_nodes(request: Request):
    return {"nodes": request.app.state.queue.list_nodes()}


@app.get("/nodes/{node_id}", response_model=NodeInfo, dependencies=[Depends(authenticate)])
def get_node(node_id: str, request: Request):
    node = request.app.state.queue.get_node(node_id)
    if node is None:
        raise HTTPException(404, "Node not found")
    return node


@app.post(
    "/nodes/{node_id}/heartbeat", response_model=NodeInfo, dependencies=[Depends(authenticate)]
)
def node_heartbeat(node_id: str, request: Request):
    node = request.app.state.queue.node_heartbeat(node_id)
    if node is None:
        raise HTTPException(404, "Node not found")
    return node


@app.post("/worker/jobs/{job_id}/heartbeat", dependencies=[Depends(authenticate)])
def heartbeat_job(job_id: str, body: LeaseRequest, request: Request):
    return request.app.state.queue.heartbeat(job_id, body.lease_token)


@app.post("/worker/jobs/{job_id}/finish", dependencies=[Depends(authenticate)])
def finish_job(job_id: str, body: FinishRequest, request: Request):
    return request.app.state.queue.finish(job_id, body)
