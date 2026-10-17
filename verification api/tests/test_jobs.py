from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from verification_api.jobs import JobSubmission, NodeRegistration, Queue
from verification_api.main import app


def submission():
    return {
        "repository": "https://github.com/example/engine.git",
        "commit": "a" * 40,
        "manifest_path": "bench/config.json",
    }


def report():
    return {
        "schema_version": 1,
        "benchmarks": [{"id": "c1", "status": "completed", "metrics": {"tokens_per_second": 10}}],
    }


def claim(client, node_id):
    client.post("/nodes/register", json={"id": node_id, "config": {}}).raise_for_status()
    return client.post("/worker/claim", json={"node_id": node_id})


def test_job_lifecycle_preserves_request_and_result(client):
    response = client.post("/jobs", json=submission())
    assert response.status_code == 202
    queued = response.json()
    assert queued["status"] == "queued"
    assert queued["submission"] == submission()
    claimed = claim(client, "gpu1").json()["job"]
    assert claimed["id"] == queued["id"]
    assert claimed["status"] == "running"
    assert "lease_token" not in client.get(f"/jobs/{queued['id']}").json()
    token = {"lease_token": claimed["lease_token"]}
    assert client.post(f"/worker/jobs/{queued['id']}/heartbeat", json=token).status_code == 200
    finished = client.post(
        f"/worker/jobs/{queued['id']}/finish",
        json={**token, "status": "completed", "report": report()},
    )
    assert finished.status_code == 200
    assert finished.json()["status"] == "completed"
    assert client.get(f"/jobs/{queued['id']}").json()["report"] == report()
    assert client.post(f"/worker/jobs/{queued['id']}/heartbeat", json=token).status_code == 409


def test_queue_is_fifo_and_durable(tmp_path):
    path = str(tmp_path / "queue.sqlite3")
    queue = Queue(path)
    first = queue.submit(JobSubmission.model_validate(submission()))
    second = queue.submit(JobSubmission.model_validate(submission()))
    restarted = Queue(path)
    for node_id in ("worker1", "worker2", "worker3"):
        restarted.register_node(NodeRegistration(id=node_id, config={}))
    assert restarted.claim("worker1")["id"] == first["id"]
    assert restarted.claim("worker2")["id"] == second["id"]
    assert restarted.claim("worker3") is None
    assert restarted.get(first["id"])["status"] == "running"


def test_only_one_worker_can_claim_a_job(tmp_path):
    queue = Queue(str(tmp_path / "queue.sqlite3"))
    queue.submit(JobSubmission.model_validate(submission()))
    for i in range(8):
        queue.register_node(NodeRegistration(id=f"gpu{i}", config={}))
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(queue.claim, [f"gpu{i}" for i in range(8)]))
    assert sum(claim is not None for claim in claims) == 1


@pytest.mark.parametrize("running", [False, True])
def test_cancel_cannot_be_overwritten(client, running):
    job = client.post("/jobs", json=submission()).json()
    token = None
    if running:
        token = claim(client, "gpu").json()["job"]["lease_token"]
    assert client.post(f"/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
    if token:
        response = client.post(
            f"/worker/jobs/{job['id']}/finish",
            json={"lease_token": token, "status": "completed", "report": report()},
        )
        assert response.status_code == 409
    assert claim(client, "other").json()["job"] is None


def test_expired_lease_fails_without_automatic_retry(client):
    job = client.post("/jobs", json=submission()).json()
    claimed = claim(client, "gpu").json()["job"]
    with app.state.queue.connection() as db:
        db.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (job["id"],))
    response = client.post(
        f"/worker/jobs/{job['id']}/heartbeat", json={"lease_token": claimed["lease_token"]}
    )
    assert response.status_code == 409
    expired = client.get(f"/jobs/{job['id']}").json()
    assert expired["status"] == "failed"
    assert expired["error"] == "worker_lease_expired"
    assert claim(client, "other").json()["job"] is None


def test_wrong_lease_cannot_finish(client):
    job = client.post("/jobs", json=submission()).json()
    claim(client, "gpu")
    assert (
        client.post(
            f"/worker/jobs/{job['id']}/finish",
            json={"lease_token": "wrong", "status": "failed", "error": "test"},
        ).status_code
        == 409
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"commit": "main"},
        {"commit": "abc123"},
        {"repository": "file:///tmp/repo"},
        {"repository": "https://user:secret@example.com/repo"},
        {"manifest_path": "../secret"},
        {"manifest_path": "/etc/passwd"},
        {"manifest_path": "..\\secret"},
        {"benchmarks": []},
        {"runner_command": ["untrusted"]},
    ],
)
def test_submission_validation(client, updates):
    assert client.post("/jobs", json={**submission(), **updates}).status_code == 422


@pytest.mark.parametrize(
    "bad_report",
    [
        {},
        {"schema_version": 1, "benchmarks": []},
        {"schema_version": 1, "benchmarks": [{"id": "c1", "status": "failed"}]},
        {
            "schema_version": 1,
            "benchmarks": [{"id": "c1", "status": "completed", "metrics": {"speed": -1}}],
        },
    ],
)
def test_invalid_report_rejected(client, bad_report):
    job = client.post("/jobs", json=submission()).json()
    claimed = claim(client, "gpu").json()["job"]
    response = client.post(
        f"/worker/jobs/{job['id']}/finish",
        json={"lease_token": claimed["lease_token"], "status": "completed", "report": bad_report},
    )
    assert response.status_code == 422
    assert client.get(f"/jobs/{job['id']}").json()["status"] == "running"


def test_auth_and_missing_job(monkeypatch, tmp_path):
    monkeypatch.setenv("VERIFIER_API_KEY", "service-secret")
    monkeypatch.setenv("VERIFIER_DB_PATH", str(tmp_path / "auth.sqlite3"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/jobs", json=submission()).status_code == 401
        assert client.post("/worker/claim", json={"worker_id": "gpu"}).status_code == 401
        headers = {"Authorization": "Bearer service-secret"}
        assert client.post("/jobs", json=submission(), headers=headers).status_code == 202
        assert client.get("/jobs/missing", headers=headers).status_code == 404
        assert client.post("/jobs/missing/cancel", headers=headers).status_code == 404
