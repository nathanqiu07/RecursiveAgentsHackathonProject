from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from agent_harness.models import RunRequest
from agent_harness.store import LostLease, RunStore


def request():
    return RunRequest.model_validate(
        {
            "role": "researcher",
            "objective": "Test durable run ownership.",
            "allocation": {
                "resource_id": "cpu",
                "node_id": "node",
                "cpus": 1.0,
                "memory_mb": 128,
                "scratch_mb": 16,
                "timeout_seconds": 600,
            },
        }
    )


def test_concurrent_workers_cannot_claim_the_same_run(tmp_path):
    path = tmp_path / "runs.sqlite3"
    first, second = RunStore(path), RunStore(path)
    try:
        run = first.create(request())
        barrier = Barrier(2)

        def claim(store):
            barrier.wait(timeout=5)
            return store.claim()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(claim, [first, second]))
        claims = [result for result in results if result]
        assert len(claims) == 1
        assert claims[0][0]["id"] == run["id"]
    finally:
        first.close()
        second.close()


def test_expired_owner_cannot_renew_change_state_record_events_or_publish(tmp_path):
    clock = [1000]
    store = RunStore(tmp_path / "runs.sqlite3", now=lambda: clock[0])
    try:
        run = store.create(request())
        _, old_token, _ = store.claim()
        clock[0] += 121
        recovered, new_token, interrupted = store.claim()
        assert recovered["id"] == run["id"] and interrupted
        mutations = [
            lambda: store.heartbeat(run["id"], old_token),
            lambda: store.update(run["id"], old_token, state={"stale": True}),
            lambda: store.event(run["id"], old_token, "stale", {}),
            lambda: store.finish(run["id"], old_token, "submitted"),
        ]
        for mutate in mutations:
            with pytest.raises(LostLease):
                mutate()
        assert store.get(run["id"])["state"] == {}
        assert store.events(run["id"]) == []
        store.finish(run["id"], new_token, "needs_attention", error="interrupted")
        assert store.get(run["id"])["status"] == "needs_attention"
    finally:
        store.close()


def test_worker_heartbeat_expires_and_is_renewed_during_a_run(tmp_path):
    clock = [1000]
    store = RunStore(tmp_path / "runs.sqlite3", now=lambda: clock[0])
    try:
        store.worker_id = "worker"
        store.touch_worker()
        assert store.status()["workers_online"] == 1
        clock[0] += 121
        assert store.status()["workers_online"] == 0
        store.create(request())
        run, token, _ = store.claim()
        store.heartbeat(run["id"], token)
        assert store.status()["workers_online"] == 1
        assert store.status()["running_runs"] == 1
    finally:
        store.close()


def test_simultaneous_launch_retries_create_one_persisted_run(tmp_path):
    path = tmp_path / "runs.sqlite3"
    first, second = RunStore(path), RunStore(path)
    try:
        barrier = Barrier(2)

        def create(store):
            barrier.wait(timeout=5)
            return store.create(request(), "same-launch")

        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = list(pool.map(create, [first, second]))
        assert runs[0]["id"] == runs[1]["id"]
        assert first.list()["total"] == 1
    finally:
        first.close()
        second.close()
