import os
import secrets
from contextlib import asynccontextmanager
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from .clients import ServiceError
from .config import database_path, services_from_env
from .models import ResearchRunRequest, RunRequest
from .profiles import ProfileRequest, ProfileStore, harness_catalog
from .prompts import tools_for
from .providers import check_provider, provider_models
from .settings import SettingsError, SettingsStore
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


@app.middleware("http")
async def private_settings(request: Request, call_next):
    settings_path = request.url.path.startswith("/settings/")
    if settings_path and request.method not in {"GET", "HEAD", "OPTIONS"}:
        # Browsers use the dashboard's authenticated, same-origin server proxy.
        # Reject direct cross-site form requests even on an unauthenticated local harness.
        if request.headers.get("origin"):
            return JSONResponse(
                status_code=403,
                content={"detail": "Use the dashboard settings API"},
                headers={"Cache-Control": "no-store"},
            )
    response = await call_next(request)
    if settings_path:
        response.headers["Cache-Control"] = "no-store"
    return response


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


def settings_database(request: Request):
    with SettingsStore(request.app.state.db_path) as store:
        yield store


class ProviderSettingsBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: StrictStr | None = Field(default=None, max_length=4096)
    model: StrictStr | None = Field(default=None, max_length=256)


def provider_name(provider):
    if provider not in {"crusoe", "you"}:
        raise HTTPException(404, "Provider not found")
    return provider


@app.exception_handler(SettingsError)
async def settings_error(_request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/settings/providers", dependencies=[Depends(authenticate)])
def provider_settings(store=Depends(settings_database)):
    return store.get_public()


@app.post("/settings/providers/{provider}", dependencies=[Depends(authenticate)])
def save_provider(provider: str, body: ProviderSettingsBody, store=Depends(settings_database)):
    provider_name(provider)
    if provider == "you" and body.model is not None:
        raise HTTPException(400, "You.com search does not accept a model ID")
    try:
        return store.save(provider, api_key=body.api_key, model=body.model)
    except SettingsError as exc:
        if str(exc).startswith(("API key must", "Model ID must")):
            raise HTTPException(400, str(exc)) from None
        raise


@app.post("/settings/providers/{provider}/remove", dependencies=[Depends(authenticate)])
def remove_provider(provider: str, store=Depends(settings_database)):
    return store.remove(provider_name(provider))


@app.get("/settings/providers/crusoe/models", dependencies=[Depends(authenticate)])
def crusoe_models(store=Depends(settings_database)):
    private = store.get_private("crusoe")
    if not private["enabled"] or not private["api_key"]:
        raise HTTPException(409, "Save a Crusoe API key before loading models")
    try:
        return {"models": provider_models(private["api_key"])}
    except ServiceError as exc:
        raise HTTPException(502, exc.code) from None


@app.post("/settings/providers/{provider}/test", dependencies=[Depends(authenticate)])
def test_provider(provider: str, store=Depends(settings_database)):
    provider_name(provider)
    private = store.get_private(provider)
    if not private["enabled"] or not private["api_key"]:
        raise HTTPException(409, "Save an API key before testing this provider")
    if provider == "crusoe" and not private["model"]:
        raise HTTPException(409, "Select and save a Crusoe model before testing")
    try:
        result = check_provider(provider, private["api_key"], private["model"])
    except ServiceError:
        result = {
            "ok": False,
            "message": "Provider could not be reached; check credentials and retry",
        }
    if not store.record_check(provider, result["ok"], result["message"], expected=private):
        raise HTTPException(
            409, "Provider settings changed during the test; test the saved settings"
        )
    return store.get_public()


@app.get("/settings/readiness", dependencies=[Depends(authenticate)])
def readiness(request: Request):
    from .readiness import build_readiness

    return build_readiness(request.app.state.db_path)


@app.get("/settings/harnesses", dependencies=[Depends(authenticate)])
def harness_settings(request: Request):
    return harness_catalog(request.app.state.db_path)


@app.post("/settings/harnesses/profiles", dependencies=[Depends(authenticate)])
def save_model_profile(body: ProfileRequest, request: Request):
    with ProfileStore(request.app.state.db_path) as store:
        store.save_profile(body)
    return harness_catalog(request.app.state.db_path)


@app.post("/settings/harnesses/profiles/{profile_id}/remove", dependencies=[Depends(authenticate)])
def remove_model_profile(profile_id: UUID, request: Request):
    with ProfileStore(request.app.state.db_path) as store:
        store.remove_profile(profile_id)
    return harness_catalog(request.app.state.db_path)


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
    state = {
        key: value
        for key, value in run["state"].items()
        if key not in {"messages", "runtime_cleanup"}
    }
    if summary:
        proposal = state.get("research_proposal")
        proposal = proposal if isinstance(proposal, dict) else {}
        sources = state.get("research_sources")
        sources = sources if isinstance(sources, dict) else {}
        access = state.get("research_access")
        access = access if isinstance(access, dict) else {}
        mcp = access.get("you_mcp")
        mcp = mcp if isinstance(mcp, dict) else {}
        urls = []
        candidates = proposal.get("source_urls", [])
        candidates = candidates if isinstance(candidates, list) else []
        for value in [*candidates, *sources]:
            if not isinstance(value, str) or not 1 <= len(value) <= 1024 or value not in sources:
                continue
            try:
                url = urlsplit(value)
                _ = url.port
                if (
                    url.scheme != "https"
                    or not url.hostname
                    or url.username
                    or url.password
                    or url.fragment
                    or "\\" in value
                    or any(ord(character) <= 32 or ord(character) == 127 for character in value)
                ):
                    continue
            except ValueError:
                continue
            if value not in urls:
                urls.append(value)
            if len(urls) == 3:
                break
        research_summary = {
            "source_urls": urls,
            "papers_read": len(sources),
            "access_connected": isinstance(mcp.get("tools"), list) and "you-search" in mcp["tools"],
        }
        hypothesis = proposal.get("hypothesis")
        if isinstance(hypothesis, str) and hypothesis.strip():
            research_summary["hypothesis"] = hypothesis[:1200]
        state = {
            "steps": state.get("steps", 0),
            "experiment_count": len(state.get("experiments", [])),
            "publication": state.get("publication"),
            **(
                {"research_summary": research_summary}
                if run["request"]["role"] == "researcher"
                and (run["request"].get("allocation") is None or sources or proposal or access)
                else {}
            ),
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


@app.get("/resource-status", dependencies=[Depends(authenticate)])
def resource_status():
    try:
        with services_from_env() as services:
            return services.resource_status()
    except ServiceError as exc:
        raise HTTPException(502, exc.code) from None


@app.post("/runs", status_code=202, dependencies=[Depends(authenticate)])
def create_run(body: RunRequest, request: Request, response: Response, store=Depends(database)):
    if body.allocation is None:
        raise HTTPException(422, "Use /research-runs for automatic research setup")
    try:
        original = body.model_copy(deep=True)
        key = request.headers.get("idempotency-key")
        repeated = store.replay(original, key)
        if repeated:
            response.headers["Location"] = "/runs/" + repeated["id"]
            return public_run(repeated)
        if body.model_profile_id:
            with ProfileStore(request.app.state.db_path) as profiles:
                profile = profiles.private_profile(body.model_profile_id)
            if not profile["enabled"] or profile["harness"] != body.harness:
                raise HTTPException(409, "Choose an enabled model profile for this harness")
            # Preserve the chosen model in history even when a profile is later removed.
            body.model = profile["model"]
        if body.harness != "builtin":
            catalog = harness_catalog(request.app.state.db_path)
            runtime = next(item for item in catalog["harnesses"] if item["id"] == body.harness)
            if not runtime["available"]:
                raise HTTPException(409, "This harness runtime is not available; check Connections")
            if body.harness in {"claude-code", "nemoclaw"} and not body.model_profile_id:
                raise HTTPException(409, "Choose a saved model profile for this harness")
        run = store.create(body, key, original_request=original)
    except RunConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    response.headers["Location"] = "/runs/" + run["id"]
    return public_run(run)


@app.post("/research-runs", status_code=202, dependencies=[Depends(authenticate)])
def create_research_run(
    body: ResearchRunRequest, request: Request, response: Response, store=Depends(database)
):
    # Identity is the user's one-field request, independent of mutable profiles,
    # titles and current GPU availability. Existing idempotency machinery owns
    # publication-safe creation atomically.
    identity = RunRequest(
        role="researcher",
        benchmark_id=body.benchmark_id,
        objective=f"Research benchmark {body.benchmark_id}",
        allocation=None,
        harness="codex",
    )
    key = request.headers.get("idempotency-key")
    try:
        repeated = store.replay(identity, key)
        if repeated:
            response.headers["Location"] = "/runs/" + repeated["id"]
            return public_run(repeated)
        with services_from_env() as services:
            benchmark = services.benchmark(str(body.benchmark_id))
        title = benchmark.get("title")
        if not isinstance(title, str) or not title.strip():
            raise HTTPException(502, "The benchmark context is unavailable")
        catalog = harness_catalog(request.app.state.db_path)
        available = {item["id"]: item for item in catalog["harnesses"] if item["available"]}
        profiles = [item for item in catalog["profiles"] if item["enabled"]]
        selected = None
        for harness in ("codex", "builtin", "claude-code", "nemoclaw"):
            if harness not in available:
                continue
            options = [item for item in profiles if item["harness"] == harness]
            profile = next((item for item in options if item["auth_mode"] == "codex-oauth"), None)
            profile = profile or next(
                (item for item in options if item["api_key_set"] and item.get("base_url")), None
            )
            if (
                harness == "codex"
                or profile
                or (harness == "builtin" and available[harness]["models"])
            ):
                selected = harness, profile
                break
        if selected is None:
            raise HTTPException(409, "Sign in to Codex or connect a research model in Connections")
        harness, profile = selected
        run_request = RunRequest(
            role="researcher",
            benchmark_id=body.benchmark_id,
            allocation=None,
            objective=(
                f"Produce an original research paper for {title[:200]} (benchmark {body.benchmark_id}). "
                "Read the complete benchmark contract, search the Library and external papers, "
                "develop your own optimization, run controlled experiments, and write your own paper. "
                "Begin research immediately; request GPU execution only when needed. "
                "Submit the original paper with measured results and reproducible evidence to the Library. "
                "Online papers are references; a reading list or idea alone is not the deliverable."
            ),
            harness=harness,
            model_profile_id=profile["id"] if profile else None,
            model=profile["model"] if profile else None,
        )
        run = store.create(run_request, key, original_request=identity)
    except RunConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except ServiceError as exc:
        raise HTTPException(502, exc.code) from None
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
