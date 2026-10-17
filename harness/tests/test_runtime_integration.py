"""Regressions at the dashboard launch, queued model, and runtime cleanup boundaries."""

import json
import threading

import pytest
from fastapi.testclient import TestClient

from agent_harness import cli_models, codex_model, main, profiles
from agent_harness.clients import ServiceError
from agent_harness.engine import EndRun, Engine, run_once
from agent_harness.models import RunRequest
from agent_harness.profiles import ProfileRequest, ProfileStore, model_for_run
from agent_harness.store import RunStore

BENCHMARK = "11111111-1111-4111-8111-111111111111"
SECRET = "dummy-runtime-profile-key-never-live"


def launch_body(role="researcher", **changes):
    return {
        "role": role,
        "objective": "Test the runtime integration without making provider requests.",
        "benchmark_id": BENCHMARK if role == "attempt" else None,
        "allocation": {
            "resource_id": "test-pool",
            "node_id": "test-node",
            "cpus": 1,
            "memory_mb": 64,
            "scratch_mb": 16,
            "timeout_seconds": 600,
        },
        "max_steps": 2,
        "harness": "codex",
        "model_profile_id": None,
        "model": None,
        **changes,
    }


def save_profile(path, harness="codex", model="old-model", identifier=None):
    body = ProfileRequest(
        id=identifier,
        name="Integration profile",
        harness=harness,
        model=model,
        base_url=None if harness == "codex" else "https://profile.example.test/v1",
        api_key=None if harness == "codex" else SECRET,
        auth_mode="codex-oauth" if harness == "codex" else "api-key",
    )
    with ProfileStore(path) as store:
        return store.save_profile(body)


@pytest.mark.parametrize("mutation", ["edit", "remove", "offline"])
@pytest.mark.parametrize("role", ["researcher", "attempt"])
def test_accepted_launch_replays_original_identity_before_mutable_checks(
    tmp_path, monkeypatch, mutation, role
):
    path = tmp_path / "runs.sqlite3"
    monkeypatch.setenv("HARNESS_DB_PATH", str(path))
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    selected = save_profile(path)
    catalog_calls = []

    def available_catalog(_path):
        catalog_calls.append(True)
        return {"harnesses": [{"id": "codex", "available": True}]}

    monkeypatch.setattr(main, "harness_catalog", available_catalog)
    body = launch_body(role, model_profile_id=selected["id"])
    headers = {"Idempotency-Key": "stable-dashboard-launch"}
    with TestClient(main.app) as client:
        accepted = client.post("/runs", json=body, headers=headers)
        assert accepted.status_code == 202
        original = accepted.json()
        assert original["request"]["model"] == "old-model"
        if mutation == "edit":
            save_profile(path, model="new-model", identifier=selected["id"])
        elif mutation == "remove":
            with ProfileStore(path) as store:
                store.remove_profile(selected["id"])
        else:
            monkeypatch.setattr(
                main,
                "harness_catalog",
                lambda _path: {"harnesses": [{"id": "codex", "available": False}]},
            )

        replay = client.post("/runs", json=body, headers=headers)
        assert replay.status_code == 202
        assert replay.json()["id"] == original["id"]
        assert replay.json()["request"]["model"] == "old-model"
        assert replay.headers["location"] == accepted.headers["location"]
        assert client.get("/runs").json()["total"] == 1
        assert catalog_calls == [True]
        conflicting = client.post(
            "/runs", json={**body, "objective": "A different objective"}, headers=headers
        )
        assert conflicting.status_code == 409
        fresh = client.post("/runs", json=body, headers={"Idempotency-Key": "new-launch"})
        if mutation == "edit":
            assert fresh.status_code == 202
            assert fresh.json()["id"] != original["id"]
            assert fresh.json()["request"]["model"] == "new-model"
        else:
            assert fresh.status_code == 409


@pytest.mark.parametrize("harness", ["builtin", "codex", "claude-code", "nemoclaw"])
def test_queued_model_stays_frozen_while_selected_profile_credentials_are_used(
    tmp_path, monkeypatch, harness
):
    path = tmp_path / "runs.sqlite3"
    selected = save_profile(path, harness=harness)
    store = RunStore(path)
    try:
        queued = store.create(
            RunRequest.model_validate(
                launch_body(harness=harness, model_profile_id=selected["id"], model="old-model")
            )
        )
        save_profile(path, harness=harness, model="new-model", identifier=selected["id"])
        constructed = []

        class CapturedModel:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]
                constructed.append(kwargs)

        monkeypatch.setattr(codex_model, "CodexModel", CapturedModel)
        monkeypatch.setattr(cli_models, "ClaudeCodeModel", CapturedModel)
        monkeypatch.setattr(cli_models, "NemoClawModel", CapturedModel)
        monkeypatch.setattr(profiles, "ChatModel", CapturedModel)
        monkeypatch.setattr(
            profiles,
            "model_configuration",
            lambda _path: {
                "model": "environment-default",
                "base_url": "https://environment.example.test/v1",
                "api_key": "dummy-environment-key",
            },
        )

        actual = model_for_run(path, queued["request"])
        assert actual.model == "old-model"
        assert constructed[0]["model"] == "old-model"
        if harness == "codex":
            assert constructed == [{"model": "old-model"}]
        else:
            assert constructed[0]["api_key"] == SECRET
            assert constructed[0]["base_url"] == "https://profile.example.test/v1"
        assert SECRET not in json.dumps(queued)
    finally:
        store.close()


def test_runtime_teardown_renews_lease_and_finishes_even_after_cancellation(tmp_path):
    clock = [1000.0]
    store = RunStore(tmp_path / "runs.sqlite3", now=lambda: clock[0])
    release = threading.Event()
    closed = threading.Event()
    try:
        created = store.create(RunRequest.model_validate(launch_body()))
        run, token, _ = store.claim()
        engine = Engine(store, run, token, None, None, None)
        engine.state.update(
            runtime_cleanup={"private": "runtime-authority"}, runtime_cleanup_required=True
        )
        engine.save()
        store.cancel(created["id"])
        original_heartbeat = store.heartbeat
        beats = []

        def heartbeat(run_id, lease_token):
            clock[0] += 60
            try:
                original_heartbeat(run_id, lease_token)
            finally:
                beats.append(clock[0])
                if len(beats) >= 3:
                    release.set()

        store.heartbeat = heartbeat

        class SlowRuntime:
            def close(self):
                assert release.wait(3), "Cleanup stopped renewing ownership"
                closed.set()

        engine.close_runtime(SlowRuntime())
        assert closed.is_set()
        assert len(beats) >= 3
        assert clock[0] >= 1180  # Cleanup outlasted the original 120-second lease.
        current = store.owned(created["id"], token)
        assert current["cancel_requested"]
        assert not current["state"]["runtime_cleanup_required"]
        assert "runtime_cleanup" not in current["state"]
        assert store.events(created["id"])[-1]["data"] == {"confirmed": True}
    finally:
        release.set()
        store.close()


def test_failed_runtime_cleanup_is_private_durable_and_reconciled_without_new_model(
    tmp_path, monkeypatch
):
    path = tmp_path / "runs.sqlite3"
    clock = [1000.0]
    store = RunStore(path, now=lambda: clock[0])
    reference = {"kind": "nemoclaw", "workdir": "/private/runtime-home", "name": "owned-runtime"}
    planner_calls = []
    recovered = []

    class Runtime:
        model = "selected-model"
        api_key = SECRET

        def cleanup_reference(self):
            return reference.copy()

        def close(self):
            raise ServiceError("nemoclaw_cleanup_required")

    def plan(engine):
        durable = store.get(engine.id)
        assert durable["state"]["runtime_cleanup"] == reference
        assert durable["cleanup_required"]
        assert SECRET not in json.dumps(durable)
        planner_calls.append(engine.id)
        raise EndRun("completed", result={"summary": "Fixture completed"})

    monkeypatch.setattr(Engine, "run_agent", plan)
    monkeypatch.setenv("HARNESS_DB_PATH", str(path))
    monkeypatch.delenv("HARNESS_API_KEY", raising=False)
    try:
        created = store.create(RunRequest.model_validate(launch_body(harness="nemoclaw")))
        assert run_once(store, None, None, None, model_factory=lambda _request: Runtime())
        failed = store.get(created["id"])
        assert failed["status"] == "needs_attention"
        assert failed["error"] == "harness_runtime_cleanup_required"
        assert failed["cleanup_required"]
        assert failed["state"]["runtime_cleanup_required"]
        assert failed["state"]["runtime_cleanup"] == reference
        with TestClient(main.app) as client:
            detail = client.get(f"/runs/{created['id']}")
            history = client.get("/runs")
            assert detail.status_code == history.status_code == 200
            assert "runtime_cleanup" not in detail.json()["state"]
            assert detail.json()["state"]["runtime_cleanup_required"]
            assert reference["workdir"] not in detail.text + history.text
            assert SECRET not in detail.text + history.text

        def reconcile(value):
            recovered.append(value)
            return True

        monkeypatch.setattr(cli_models, "reconcile_runtime_cleanup", reconcile)
        clock[0] += 6

        def no_new_model(_request):
            pytest.fail("Cleanup reconciliation must not launch a new model")

        assert run_once(store, None, None, None, model_factory=no_new_model)
        reconciled = store.get(created["id"])
        assert recovered == [reference]
        assert planner_calls == [created["id"]]
        assert not reconciled["cleanup_required"]
        assert not reconciled["state"]["runtime_cleanup_required"]
        assert "runtime_cleanup" not in reconciled["state"]
    finally:
        store.close()
