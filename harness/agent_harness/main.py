import os
import secrets
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .clients import ServiceError
from .config import database_path, services_from_env
from .models import RunRequest
from .prompts import tools_for
from .store import RunConflict, RunStore

bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app):
    app.state.db_path = database_path()
    app.state.api_key = os.environ.get("HARNESS_API_KEY")
    RunStore(app.state.db_path).close()
    yield


app = FastAPI(
    title="Agent Harness",
    version="0.1.0",
    lifespan=lifespan,
    description="Researcher and benchmark attempt agents, each in its own sandbox.",
)


def authenticate(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)
):
    key = request.app.state.api_key
    if key and (
        credentials is None
        or not secrets.compare_digest(
            key.encode(),
            credentials.credentials.encode(),
        )
    ):
        raise HTTPException(401, "Invalid harness API key", headers={"WWW-Authenticate": "Bearer"})


def database(request: Request):
    store = RunStore(request.app.state.db_path)
    try:
        yield store
    finally:
        store.close()


@app.exception_handler(RequestValidationError)
async def invalid_request(_request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "detail": [
                {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                for error in exc.errors()
            ]
        },
    )


def required_run(store, run_id):
    run = store.get(str(run_id))
    if run is None:
        raise HTTPException(404, "Agent run not found")
    return run


def public_run(run, summary=False):
    # Keep large model transcripts out of dashboard responses; detail uses audit events.
    state = {key: value for key, value in run["state"].items() if key != "messages"}
    if summary:
        state = {
            "steps": state.get("steps", 0),
            "experiment_count": len(state.get("experiments", [])),
            "publication": state.get("publication"),
        }
    return {**run, "state": state}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status", dependencies=[Depends(authenticate)])
def status(store=Depends(database)):
    return store.status()


@app.get("/agents", dependencies=[Depends(authenticate)])
def agents():
    return {
        "agents": [
            {
                "role": "researcher",
                "description": "Novel idea, experiments, paper submission to Library",
                "tools": [item["function"]["name"] for item in tools_for("researcher")],
            },
            {
                "role": "attempt",
                "description": "Apply existing research to an assigned benchmark",
                "tools": [item["function"]["name"] for item in tools_for("attempt")],
            },
        ]
    }


@app.get("/resources", dependencies=[Depends(authenticate)])
def resources():
    try:
        with services_from_env() as services:
            return {"resources": services.resources()}
    except ServiceError as exc:
        raise HTTPException(502, exc.code) from None


@app.post("/runs", status_code=202, dependencies=[Depends(authenticate)])
def create_run(body: RunRequest, request: Request, response: Response, store=Depends(database)):
    try:
        run = store.create(body, request.headers.get("idempotency-key"))
    except RunConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    response.headers["Location"] = "/runs/" + run["id"]
    return public_run(run)


@app.get("/runs", dependencies=[Depends(authenticate)])
def runs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0, le=2**63 - 1),
    store=Depends(database),
):
    result = store.list(limit, offset)
    result["runs"] = [public_run(run, summary=True) for run in result["runs"]]
    return result


@app.get("/runs/{run_id}", dependencies=[Depends(authenticate)])
def get_run(run_id: UUID, store=Depends(database)):
    return public_run(required_run(store, run_id))


@app.get("/runs/{run_id}/events", dependencies=[Depends(authenticate)])
def events(
    run_id: UUID,
    after: int = Query(default=0, ge=0, le=2**63 - 1),
    limit: int = Query(default=100, ge=1, le=200),
    store=Depends(database),
):
    required_run(store, run_id)
    items = store.events(str(run_id), after, limit)
    return {"events": items, "next_cursor": items[-1]["sequence"] if items else after}


@app.post("/runs/{run_id}/cancel", dependencies=[Depends(authenticate)])
def cancel(run_id: UUID, store=Depends(database)):
    required_run(store, run_id)
    return public_run(store.cancel(str(run_id)))


@app.get("/runs/{run_id}/publication", dependencies=[Depends(authenticate)])
def publication(run_id: UUID, store=Depends(database)):
    run = required_run(store, run_id)
    published = run["state"].get("publication", {})
    if published.get("status") != "acknowledged":
        raise HTTPException(409, "No acknowledged publication; inspect the run before resubmitting")
    try:
        with services_from_env() as services:
            if published["kind"] == "paper":
                return services.paper(published["id"])
            return services.submission(published["id"])
    except ServiceError as exc:
        raise HTTPException(502, exc.code) from None
