import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from verification_api.jobs import JobSubmission, NodeRegistration, Queue
from verification_api.main import app


def submit(client):
    return client.post(
        "/jobs",
        json={
            "repository": "https://example.test/engine.git",
            "commit": "a" * 40,
        },
    ).json()


def register(client, node_id="gpu-node-01", config=None):
    response = client.post(
        "/nodes/register",
        json={
            "id": node_id,
            "config": config if config is not None else {"hardware": {"gpu_count": 1}},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_node_registration_list_get_and_durability(client):
    node = register(client)
    assert node["id"] == "gpu-node-01"
    assert node["status"] == "idle"
    assert node["active_job_id"] is None
    assert client.get("/nodes/gpu-node-01").json() == node
    assert client.get("/nodes").json()["nodes"] == [node]
    restarted = Queue(app.state.queue.path)
    assert restarted.get_node(node["id"])["config"] == node["config"]


def test_node_must_register_before_claim(client):
    submit(client)
    response = client.post("/worker/claim", json={"node_id": "unregistered"})
    assert response.status_code == 409


def test_node_config_is_frozen_per_job(client):
    original = {"hardware": {"gpu_model": "H100", "gpu_count": 1}}
    register(client, config=original)
    job = submit(client)
    claimed = client.post("/worker/claim", json={"node_id": "gpu-node-01"}).json()["job"]
    assert claimed["node_id"] == "gpu-node-01"
    assert claimed["node_config"] == original
    busy = client.get("/nodes/gpu-node-01").json()
    assert busy["status"] == "busy"
    assert busy["active_job_id"] == job["id"]
    assert (
        client.post(
            "/nodes/register",
            json={
                "id": "gpu-node-01",
                "config": {"hardware": {"gpu_model": "H200"}},
            },
        ).status_code
        == 409
    )
    result = client.post(
        f"/worker/jobs/{job['id']}/finish",
        json={
            "lease_token": claimed["lease_token"],
            "status": "failed",
            "error": "fixture",
        },
    )
    assert result.status_code == 200
    register(client, config={"hardware": {"gpu_model": "H200"}})
    assert client.get(f"/jobs/{job['id']}").json()["node_config"] == original
    assert client.get("/nodes/gpu-node-01").json()["status"] == "idle"


def test_same_node_cannot_claim_two_jobs_concurrently(tmp_path):
    queue = Queue(str(tmp_path / "nodes.sqlite3"))
    queue.register_node(NodeRegistration(id="single-gpu", config={}))
    for _ in range(2):
        queue.submit(JobSubmission(repository="https://example.test/engine.git", commit="a" * 40))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(queue.claim, ["single-gpu", "single-gpu"]))
    assert sum(result is not None for result in results) == 1


def test_node_liveness_and_job_heartbeats(client):
    register(client)
    with app.state.queue.connection() as db:
        db.execute("UPDATE nodes SET last_seen=0")
    assert client.get("/nodes/gpu-node-01").json()["status"] == "offline"
    assert client.post("/nodes/gpu-node-01/heartbeat").json()["status"] == "idle"
    job = submit(client)
    claimed = client.post("/worker/claim", json={"node_id": "gpu-node-01"}).json()["job"]
    with app.state.queue.connection() as db:
        db.execute("UPDATE nodes SET last_seen=0")
    response = client.post(
        f"/worker/jobs/{job['id']}/heartbeat",
        json={
            "lease_token": claimed["lease_token"],
        },
    )
    assert response.status_code == 200
    assert client.get("/nodes/gpu-node-01").json()["status"] == "busy"


@pytest.mark.parametrize(
    "body",
    [
        {"id": "", "config": {}},
        {"id": "bad/id", "config": {}},
        {"id": "a" * 101, "config": {}},
        {"id": "missing-config"},
        {"id": "node", "config": []},
    ],
)
def test_node_registration_validation(client, body):
    assert client.post("/nodes/register", json=body).status_code == 422


def test_node_missing_and_authentication(client, monkeypatch):
    assert client.get("/nodes/missing").status_code == 404
    assert client.post("/nodes/missing/heartbeat").status_code == 404
    monkeypatch.setattr(app.state, "api_key", "service-key")
    assert client.get("/nodes").status_code == 401
    assert client.post("/nodes/register", json={"id": "node", "config": {}}).status_code == 401


def test_existing_database_gets_node_config_column(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE jobs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at REAL NOT NULL,
            updated_at REAL NOT NULL, payload TEXT NOT NULL, worker_id TEXT,
            lease_token TEXT, lease_until REAL, report TEXT, error TEXT
        )""")
    queue = Queue(str(path))
    queued = queue.submit(
        JobSubmission(repository="https://example.test/engine.git", commit="a" * 40)
    )
    assert queued["node_config"] is None
    assert queue.list_nodes() == []
