from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from agent_harness import main
from agent_harness.clients import ServiceError
from agent_harness.store import RunStore

BENCHMARK = "00000000-0000-4000-8000-000000000001"
PAPER = "00000000-0000-4000-8000-000000000002"
SUBMISSION = "00000000-0000-4000-8000-000000000003"
JOB = "00000000-0000-4000-8000-000000000004"


def request(role="researcher", **changes):
    body = {
        "role": role,
        "objective": "Investigate a reproducible optimization",
        "allocation": {
            "resource_id": "agent-cpu",
            "node_id": "cpu-node",
            "cpus": 0.5,
            "memory_mb": 256,
            "scratch_mb": 64,
            "timeout_seconds": 300,
        },
        "max_steps": 10,
    }
    if role == "attempt":
        body["benchmark_id"] = BENCHMARK
    return {**body, **changes}


@pytest.fixture
def database(monkeypatch, tmp_path):
    path = tmp_path / "harness.sqlite3"
    monkeypatch.setenv("HARNESS_DB_PATH", str(path))
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    return path


@pytest.fixture
def client(database):
    with TestClient(main.app) as connection:
        yield connection


class FakeServices:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def _read(self, kind, identifier):
        self.calls.append((kind, identifier))
        if self.error:
            raise self.error
        if kind == "paper":
            return {"id": identifier, "status": "pending_verification", "report": None}
        return {"id": identifier, "benchmark_id": BENCHMARK, "status": "verified", "score": 1.5}

    def paper(self, identifier):
        return self._read("paper", identifier)

    def submission(self, identifier):
        return self._read("submission", identifier)

    def resources(self):
        self.calls.append(("resources", None))
        if self.error:
            raise self.error
        return [{"id": "agent-cpu", "node_id": "cpu-node", "cpus": 1, "memory_mb": 1024}]


def publication(database, run_id, kind, status="acknowledged"):
    store = RunStore(database)
    try:
        run, token, _ = store.claim()
        assert run["id"] == run_id
        store.update(
            run_id,
            token,
            state={
                "publication": {
                    "kind": kind,
                    "id": PAPER if kind == "paper" else SUBMISSION,
                    "status": status,
                }
            },
        )
        store.finish(run_id, token, "submitted" if status == "acknowledged" else "needs_attention")
    finally:
        store.close()


def test_role_definitions_separate_paper_and_benchmark_publication_permissions(client):
    response = client.get("/agents")
    assert response.status_code == 200
    roles = {role["role"]: set(role["tools"]) for role in response.json()["agents"]}
    assert set(roles) == {"researcher", "attempt"}
    assert {"review_novelty", "submit_paper"} <= roles["researcher"]
    assert "submit_attempt" not in roles["researcher"]
    assert "submit_attempt" in roles["attempt"]
    assert "submit_paper" not in roles["attempt"]
    assert "review_novelty" not in roles["attempt"]
    for tools in roles.values():
        assert {"search_library", "read_paper", "execute", "web_search", "web_read"} <= tools


def test_launch_retries_return_the_same_run_and_reject_changed_settings(client):
    headers = {"Idempotency-Key": "dashboard-launch-001"}
    first = client.post("/runs", json=request(), headers=headers)
    repeated = client.post("/runs", json=request(), headers=headers)
    assert first.status_code == repeated.status_code == 202
    assert first.json()["id"] == repeated.json()["id"]
    changed = client.post("/runs", json=request(objective="A different task"), headers=headers)
    assert changed.status_code == 409
    assert client.get("/runs").json()["total"] == 1


def test_launch_key_survives_api_restart(database):
    with TestClient(main.app) as client:
        first = client.post("/runs", json=request(), headers={"Idempotency-Key": "retry"}).json()
    with TestClient(main.app) as client:
        repeated = client.post("/runs", json=request(), headers={"Idempotency-Key": "retry"})
        assert repeated.json()["id"] == first["id"]


@pytest.mark.parametrize("key", ["", "contains space", "x" * 201])
def test_invalid_launch_keys_are_rejected(client, key):
    assert client.post("/runs", json=request(), headers={"Idempotency-Key": key}).status_code == 400
    assert client.get("/runs").json()["total"] == 0


def test_dashboard_responses_exclude_transcripts_but_preserve_evidence_and_progress(
    client, database
):
    run = client.post("/runs", json=request()).json()
    store = RunStore(database)
    try:
        _, token, _ = store.claim()
        store.update(
            run["id"],
            token,
            state={
                "messages": [{"role": "assistant", "content": "internal model transcript"}],
                "steps": 4,
                "experiments": [{"id": "execute-4", "output": "observed 55"}],
                "novelty": {"approved": True},
            },
        )
    finally:
        store.close()
    item = client.get(f"/runs/{run['id']}").json()
    assert "messages" not in item["state"]
    assert item["state"]["steps"] == 4
    assert item["state"]["experiments"][0]["output"] == "observed 55"
    assert item["state"]["novelty"]["approved"]
    summary = client.get("/runs").json()["runs"][0]
    assert summary["state"]["steps"] == 4
    assert summary["state"]["experiment_count"] == 1
    assert "experiments" not in summary["state"]
    assert "messages" not in summary["state"]


def test_status_reports_workers_separately_from_api_process(client, database):
    assert client.get("/status").json()["workers_online"] == 0
    store = RunStore(database)
    try:
        store.worker_id = "test-worker"
        store.touch_worker()
        assert client.get("/status").json()["workers_online"] == 1
        store.remove_worker()
        assert client.get("/status").json()["workers_online"] == 0
    finally:
        store.close()


@pytest.mark.parametrize("role", ["researcher", "attempt"])
def test_creating_a_run_persists_its_contract_without_launching_an_unsandboxed_agent(client, role):
    body = request(role)
    response = client.post("/runs", json=body)
    assert response.status_code == 202
    run = response.json()
    UUID(run["id"])
    assert run["request"] == {**body, "benchmark_id": body.get("benchmark_id")}
    assert run["status"] == "queued"
    assert run["compute_job_id"] is None
    assert run["sandbox_id"] is None
    assert run["cleanup_required"] is False
    assert response.headers["location"] == "/runs/" + run["id"]
    assert client.get(response.headers["location"]).json() == run


def test_attempt_requires_a_benchmark_but_researcher_can_start_without_one(client):
    assert client.post("/runs", json=request("attempt", benchmark_id=None)).status_code == 422
    assert client.get("/runs").json()["total"] == 0
    response = client.post("/runs", json=request())
    assert response.status_code == 202
    assert response.json()["request"]["benchmark_id"] is None


def test_run_objectives_are_normalized_and_default_step_budget_is_stored(client):
    body = request(objective="  Study the baseline\n ")
    del body["max_steps"]
    saved = client.post("/runs", json=body).json()["request"]
    assert saved["objective"] == "Study the baseline"
    assert saved["max_steps"] == 40


def test_run_history_is_persistent_paginated_and_ordered_newest_first(database):
    with TestClient(main.app) as first:
        created = [
            first.post("/runs", json=request(objective=f"Objective {index}")).json()
            for index in range(3)
        ]
    with TestClient(main.app) as restarted:
        page = restarted.get("/runs?limit=1&offset=1")
        assert page.status_code == 200
        expected = {
            **created[1],
            "state": {
                "steps": 0,
                "experiment_count": 0,
                "publication": None,
            },
        }
        assert page.json() == {"runs": [expected], "total": 3, "limit": 1, "offset": 1}
        summaries = restarted.get("/runs").json()["runs"]
        assert [{**run, "state": {}} for run in summaries] == list(reversed(created))
        assert restarted.get("/runs/" + created[0]["id"]).json() == created[0]
        assert restarted.get("/runs?offset=3").json()["runs"] == []


def test_cancel_queued_run_is_idempotent_and_prevents_worker_claim(client, database):
    run = client.post("/runs", json=request()).json()
    response = client.post(f"/runs/{run['id']}/cancel")
    assert response.status_code == 200
    cancelled = response.json()
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_requested"] is True
    assert cancelled["cleanup_required"] is False
    assert client.post(f"/runs/{run['id']}/cancel").json() == cancelled
    store = RunStore(database)
    try:
        assert store.claim() is None
    finally:
        store.close()


def test_cancel_active_run_preserves_held_sandbox_until_worker_confirms_cleanup(client, database):
    run_id = client.post("/runs", json=request()).json()["id"]
    store = RunStore(database)
    try:
        _, token, _ = store.claim()
        store.update(run_id, token, compute_job_id=JOB, cleanup_required=True)
        cancelled = client.post(f"/runs/{run_id}/cancel").json()
        assert cancelled["status"] == "running"
        assert cancelled["cancel_requested"] is True
        assert cancelled["cleanup_required"] is True
        assert cancelled["compute_job_id"] == JOB
    finally:
        store.close()


def test_events_are_scoped_paginated_and_never_expose_worker_lease_credentials(client, database):
    run_id = client.post("/runs", json=request()).json()["id"]
    other_id = client.post("/runs", json=request(objective="Another run")).json()["id"]
    store = RunStore(database)
    try:
        run, token, _ = store.claim()
        assert run["id"] == run_id
        store.event(run_id, token, "sandbox_allocated", {"job_id": JOB})
        store.event(run_id, token, "experiment", {"exit_code": 0, "output": "Measured 42"})
        store.update(run_id, token, state={"experiment_count": 1})
        response = client.get(f"/runs/{run_id}/events?limit=1")
        assert response.status_code == 200
        first = response.json()
        assert len(first["events"]) == 1
        assert first["events"][0]["kind"] == "sandbox_allocated"
        assert first["next_cursor"] == first["events"][0]["sequence"]
        second = client.get(f"/runs/{run_id}/events?after={first['next_cursor']}").json()
        assert [event["kind"] for event in second["events"]] == ["experiment"]
        final = client.get(f"/runs/{run_id}/events?after={second['next_cursor']}").json()
        assert final == {"events": [], "next_cursor": second["next_cursor"]}
        assert client.get(f"/runs/{other_id}/events").json() == {"events": [], "next_cursor": 0}
        for route in ("/runs", f"/runs/{run_id}", f"/runs/{run_id}/events"):
            public = client.get(route)
            assert token not in public.text
            assert "lease_token" not in public.text
            assert "lease_until" not in public.text
    finally:
        store.close()


@pytest.mark.parametrize(
    "kind,role,identifier",
    [
        ("paper", "researcher", PAPER),
        ("submission", "attempt", SUBMISSION),
    ],
)
def test_publication_status_is_read_from_correct_independent_verifier(
    client,
    database,
    monkeypatch,
    kind,
    role,
    identifier,
):
    run_id = client.post("/runs", json=request(role)).json()["id"]
    publication(database, run_id, kind)
    fake = FakeServices()
    monkeypatch.setattr(main, "services_from_env", lambda: fake)
    response = client.get(f"/runs/{run_id}/publication")
    assert response.status_code == 200
    assert response.json()["id"] == identifier
    assert fake.calls == [(kind, identifier)]
    assert fake.closed is True
    if kind == "paper":
        assert response.json()["status"] == "pending_verification"
        assert response.json()["report"] is None
    else:
        assert response.json()["score"] == 1.5


@pytest.mark.parametrize("status", [None, "dispatching", "unknown"])
def test_unacknowledged_publications_are_not_replayed_or_assumed_accepted(
    client,
    database,
    monkeypatch,
    status,
):
    run_id = client.post("/runs", json=request()).json()["id"]
    if status is not None:
        publication(database, run_id, "paper", status)
    fake = FakeServices()
    monkeypatch.setattr(main, "services_from_env", lambda: fake)
    response = client.get(f"/runs/{run_id}/publication")
    assert response.status_code == 409
    assert fake.calls == []


def test_publication_upstream_failure_returns_safe_error(client, database, monkeypatch):
    run_id = client.post("/runs", json=request()).json()["id"]
    publication(database, run_id, "paper")
    fake = FakeServices(ServiceError("unavailable"))
    monkeypatch.setattr(main, "services_from_env", lambda: fake)
    response = client.get(f"/runs/{run_id}/publication")
    assert response.status_code == 502
    assert response.json() == {"detail": "unavailable"}
    assert fake.closed is True


def test_resource_snapshot_uses_trusted_service_client(client, monkeypatch):
    fake = FakeServices()
    monkeypatch.setattr(main, "services_from_env", lambda: fake)
    response = client.get("/resources")
    assert response.status_code == 200
    assert response.json()["resources"][0]["id"] == "agent-cpu"
    assert fake.calls == [("resources", None)]
    assert fake.closed is True


def test_all_agent_routes_require_configured_bearer_key_but_health_stays_public(
    database, monkeypatch
):
    monkeypatch.setenv("HARNESS_API_KEY", "harness-test-secret")
    fake = FakeServices()
    monkeypatch.setattr(main, "services_from_env", lambda: fake)
    with TestClient(main.app) as client:
        authorized = {"Authorization": "Bearer harness-test-secret"}
        created = client.post("/runs", json=request(), headers=authorized)
        assert created.status_code == 202
        run_id = created.json()["id"]
        routes = [
            ("GET", "/agents"),
            ("GET", "/resources"),
            ("GET", "/runs"),
            ("POST", "/runs"),
            ("GET", f"/runs/{run_id}"),
            ("GET", f"/runs/{run_id}/events"),
            ("POST", f"/runs/{run_id}/cancel"),
            ("GET", f"/runs/{run_id}/publication"),
        ]
        for method, route in routes:
            for headers in ({}, {"Authorization": "Bearer wrong"}):
                response = client.request(method, route, headers=headers, json=request())
                assert response.status_code == 401
                assert response.headers["www-authenticate"] == "Bearer"
                assert "harness-test-secret" not in response.text
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/agents", headers=authorized).status_code == 200
        assert client.get("/resources", headers=authorized).status_code == 200
        assert client.get("/runs", headers=authorized).json()["total"] == 1


@pytest.mark.parametrize(
    "route,method",
    [
        ("", "GET"),
        ("/events", "GET"),
        ("/cancel", "POST"),
        ("/publication", "GET"),
    ],
)
def test_unknown_runs_return_404(client, route, method):
    assert client.request(method, f"/runs/{uuid4()}{route}").status_code == 404


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "task"},
        {"role": "verifier"},
        {"objective": " \n "},
        {"objective": "x" * 20001},
        {"benchmark_id": "not-a-uuid"},
        {"max_steps": 0},
        {"max_steps": 101},
        {"max_steps": True},
        {"max_steps": "10"},
        {"image": "unapproved-agent"},
        {"environment": {"TOKEN": "never-echo-this-input"}},
        {"allocation": None},
    ],
)
def test_invalid_agent_requests_do_not_enqueue_runs_or_echo_submitted_secrets(client, changes):
    response = client.post("/runs", json=request(**changes))
    assert response.status_code == 422
    assert "never-echo-this-input" not in response.text
    assert client.get("/runs").json()["total"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"resource_id": "../admin"},
        {"node_id": "node/escape"},
        {"cpus": 0},
        {"cpus": "1"},
        {"cpus": True},
        {"memory_mb": 63},
        {"memory_mb": True},
        {"scratch_mb": 15},
        {"timeout_seconds": 29},
        {"timeout_seconds": 86401},
        {"image": "unapproved-agent"},
        {"command": "run-on-host"},
    ],
)
def test_invalid_or_privilege_extending_allocations_are_rejected(client, changes):
    body = request()
    body["allocation"].update(changes)
    assert client.post("/runs", json=body).status_code == 422
    assert client.get("/runs").json()["total"] == 0


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1", "limit=invalid"])
def test_run_history_pagination_is_bounded(client, query):
    assert client.get("/runs?" + query).status_code == 422


@pytest.mark.parametrize("query", ["after=-1", "limit=0", "limit=201"])
def test_event_queries_are_bounded(client, query):
    run_id = client.post("/runs", json=request()).json()["id"]
    assert client.get(f"/runs/{run_id}/events?{query}").status_code == 422
