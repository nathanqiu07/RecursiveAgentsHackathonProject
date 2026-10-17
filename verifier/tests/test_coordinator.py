import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from nemotron_verifier.config import digest
from nemotron_verifier.coordinator import run_job, tick_node
from nemotron_verifier.main import app
from nemotron_verifier.pool import NodeBusy, baseline_state, inspect_pool, profile, verify_deployment
from nemotron_verifier.queue import Queue


def metrics(fingerprint):
    return {"identity": fingerprint, "mmlu": 100 * 45 / 57, "mmlu_correct": 45,
            "mmlu_count": 57, "gsm8k": 75, "gsm8k_correct": 12, "gsm8k_count": 16,
            "ppl": 5.0, "throughput": 50.0}


@pytest.fixture
def pool(tmp_path, monkeypatch):
    import nemotron_verifier.pool as module
    monkeypatch.setattr(module, "suite_identity", lambda _: "suite-v2")
    monkeypatch.setattr(module, "identity", lambda c, g: digest({"gpu": g, "reference": c["deployments"][c["reference"]]}))
    config = json.loads((__import__("pathlib").Path(__file__).parents[1] / "pool.example.json").read_text())
    config["nodes"] = []
    receipts, active, cancelled = {}, set(), []
    for i in range(1, 4):
        name = f"a5000-0{i}"
        gpu = {"name": "NVIDIA RTX A5000", "uuid": f"GPU-{i}", "memory_total_mb": 24564,
               "power_limit_w": 230, "driver_version": "580.1"}
        verification = {"gpu": gpu, "boot_verified": True, "image_sha256": "a" * 64}
        receipts[name] = {"node_id": name, "status": "ready", "gpu_uuid": f"GPU-{i}",
                          "address": f"http://127.0.0.1:{18100+i}", "key": "rn_private",
                          "target": {"resource_id": f"vm-{i}", "guest_image_sha256": "a" * 64},
                          "verification": verification}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(receipts[name]))
        path.chmod(0o600)
        config["nodes"].append({"id": name, "connection_file": str(path)})
    class Client:
        def __init__(self, node):
            self.node, self.receipt = node, receipts[node["id"]]
        def close(self):
            pass
        def health(self):
            return {"busy": self.node["id"] in active, "verification": self.receipt["verification"]}
        def start(self, job_id, alias, health, config):
            if self.node["id"] in active:
                raise NodeBusy()
            active.add(self.node["id"])
            return {"id": job_id}
        def cancel(self, job_id):
            cancelled.append((self.node["id"], job_id))
            active.discard(self.node["id"])
        def job(self, job_id):
            return {"cleanup_confirmed": self.node["id"] not in active}
    import nemotron_verifier.coordinator as coordinator
    monkeypatch.setattr(coordinator, "wait_for_engine", lambda *a: None)
    return config, Client, active, cancelled


def test_three_lanes_run_in_parallel_and_each_gpu_gets_its_own_baseline(pool, tmp_path):
    config, factory, active, cancelled = pool
    queue = Queue(str(tmp_path / "jobs.sqlite"))
    fingerprint, nodes = inspect_pool(config, factory)
    assert len({n["identity"] for n in nodes}) == 3
    jobs = [queue.submit_baseline(n["identity"], config["reference"], n["id"], fingerprint) for n in nodes]
    barrier = threading.Barrier(3)
    def evaluator(config, alias, identity, reference, *args):
        assert reference is None
        barrier.wait(timeout=3)
        assert len(active) == 3
        barrier.wait(timeout=3)
        return metrics(identity)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(tick_node, queue, config, node["id"], threading.Event(), factory, evaluator) for node in nodes]
        assert all(f.result(timeout=10) for f in futures)
    assert not active and len(cancelled) == 3
    assert all(queue.get(job["id"])["status"] == "completed" for job in jobs)
    assert all("score" not in queue.get(job["id"]) for job in jobs)
    assert baseline_state(queue, nodes)[1] is True
    queue.db.close()


def test_atomic_same_node_claim_and_cross_node_reference_is_never_reused(pool, tmp_path):
    config, factory, _, _ = pool
    queue = Queue(str(tmp_path / "queue.sqlite"))
    fingerprint, nodes = inspect_pool(config, factory)
    for node in nodes:
        job = queue.submit_baseline(node["identity"], config["reference"], node["id"], fingerprint)
        claimed = queue.claim_node(node["id"], fingerprint, node["identity"])
        if node == nodes[0]:
            queue.finish(job["id"], "completed", {"metrics": metrics(node["identity"])})
        else:
            queue.finish(job["id"], "failed", {})
    for i in range(3):
        queue.submit(str(i), "candidate", {"pool_identity": fingerprint, "deployment": "sglang-candidate"})
    assert queue.claim_node(nodes[1]["id"], fingerprint, nodes[1]["identity"]) is None
    with ThreadPoolExecutor(max_workers=4) as executor:
        claims = list(executor.map(lambda _: queue.claim_node(nodes[0]["id"], fingerprint, nodes[0]["identity"]), range(4)))
    assert sum(claim is not None for claim in claims) == 1
    queue.db.close()


def test_experiment_wins_race_so_candidate_waits_without_score(pool, tmp_path):
    config, factory, active, _ = pool
    queue = Queue(str(tmp_path / "queue.sqlite"))
    fingerprint, nodes = inspect_pool(config, factory)
    node = nodes[0]
    job = queue.submit_baseline(node["identity"], config["reference"], node["id"], fingerprint)
    claimed = queue.claim_node(node["id"], fingerprint, node["identity"])
    client = factory(node)
    active.add(node["id"])
    run_job(queue, config, node, fingerprint, client, client.health(), claimed, threading.Event())
    assert queue.get(job["id"])["status"] == "queued"
    assert "score" not in queue.get(job["id"])
    assert node["id"] in active
    queue.db.close()


def test_cancellation_cannot_race_into_a_score_and_source_attestation_is_exact(tmp_path):
    queue = Queue(str(tmp_path / "queue.sqlite"))
    job = queue.submit("candidate", "candidate", {})
    queue.claim()
    queue.cancel(job["id"])
    queue.finish(job["id"], "completed", {"score": 123})
    assert queue.get(job["id"])["status"] == "cancelled" and "score" not in queue.get(job["id"])
    queued = queue.submit("queued", "candidate", {})
    assert queue.cancel(queued["id"])["status"] == "cancelled"
    assert queue.claim() is None
    registered = {"repository": "https://github.com/owner/repo", "commit": "a" * 40,
                  "model_revision": "b" * 40, "execution": "W4A16", "settings": {"x": 1}}
    verify_deployment({"fresh": True, "deployment": registered}, registered)
    bad = copy.deepcopy(registered)
    bad["settings"]["x"] = 2
    with pytest.raises(ValueError):
        verify_deployment({"fresh": True, "deployment": bad}, registered)
    queue.db.close()


def test_pool_api_batches_reference_and_lists_jobs_without_keys(pool, tmp_path, monkeypatch):
    import nemotron_verifier.main as module
    config, factory, _, _ = pool
    fingerprint, states = inspect_pool(config, factory)
    monkeypatch.setenv("NEMOTRON_VERIFIER_DB", str(tmp_path / "api.sqlite"))
    monkeypatch.setenv("NEMOTRON_VERIFIER_API_KEY", "api-private")
    monkeypatch.setattr(module, "setup", lambda: (config, fingerprint))
    monkeypatch.setattr(module, "inspect_pool", lambda _: (fingerprint, states))
    headers = {"Authorization": "Bearer api-private"}
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["configured_nodes"] == 3 and health["runtime_target_is_advisory"]
        assert client.get("/nodes").status_code == 401
        response = client.post("/baseline", headers=headers)
        assert response.status_code == 202 and len(response.json()["jobs"]) == 3
        assert len(client.get("/evaluations", headers=headers).json()["evaluations"]) == 3
        assert "rn_private" not in client.get("/nodes", headers=headers).text
        job_id = response.json()["id"]
        assert client.post(f"/evaluations/{job_id}/cancel", headers=headers).json()["status"] == "cancelled"


def test_live_gpu_drift_disables_node_until_reenrollment(pool):
    config, factory, _, _ = pool
    fingerprint, before = inspect_pool(config, factory)
    class ChangedClient(factory):
        def health(self):
            document = copy.deepcopy(super().health())
            if self.node["id"] == "a5000-01":
                document["verification"]["gpu"]["driver_version"] = "590.1"
            return document
    unchanged, after = inspect_pool(config, ChangedClient)
    assert unchanged == fingerprint
    assert after[0]["status"] == "offline" and after[0]["identity"] == before[0]["identity"]
    assert after[1]["status"] == "available"


def test_new_probe_timestamp_or_vm_id_does_not_change_physical_pool_identity(pool):
    from pathlib import Path
    config, factory, _, _ = pool
    fingerprint, _ = inspect_pool(config, factory)
    for definition in config["nodes"]:
        path = Path(definition["connection_file"])
        document = json.loads(path.read_text())
        document["verification"]["gpu"].update(vm_id="new-probe-vm", observed_at=999999)
        path.write_text(json.dumps(document))
    assert inspect_pool(config, factory)[0] == fingerprint


def test_cancel_and_finish_are_atomic_across_api_worker_connections(tmp_path):
    path = str(tmp_path / "shared.sqlite")
    api, worker = Queue(path), Queue(path)
    job = api.submit("race", "candidate", {})
    worker.claim()
    read, release = threading.Event(), threading.Event()
    actual = worker.db
    class Cursor:
        def __init__(self, cursor):
            self.cursor = cursor
        def fetchone(self):
            result = self.cursor.fetchone()
            read.set()
            assert release.wait(3)
            return result
    class Connection:
        def execute(self, sql, *args):
            cursor = actual.execute(sql, *args)
            return Cursor(cursor) if sql.startswith("SELECT cancel_requested") else cursor
    worker.db = Connection()
    with ThreadPoolExecutor(max_workers=2) as executor:
        finishing = executor.submit(worker.finish, job["id"], "completed", {"score": 100})
        assert read.wait(3)
        cancelling = executor.submit(api.cancel, job["id"])
        release.set()
        finishing.result(timeout=3)
        cancelling.result(timeout=3)
    result = api.get(job["id"])
    assert not (result["cancel_requested"] and "score" in result)
    # When cancellation obtains its write lock first, no late worker score wins.
    worker.db = actual
    second = api.submit("cancel-first", "candidate", {})
    worker.claim()
    api.cancel(second["id"])
    worker.finish(second["id"], "completed", {"score": 999})
    assert api.get(second["id"])["status"] == "cancelled" and "score" not in api.get(second["id"])
    api.db.close()
    worker.db.close()
