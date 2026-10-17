import json
import os
import subprocess
import sys
import time

import pytest

from verification_api.jobs import BenchmarkManifest, validate_report
from verification_api.worker import Worker, WorkerError

FIXTURE_RUNNER = """
import json, os
assert "VERIFIER_API_KEY" not in os.environ
spec = json.load(open(os.environ["VERIFIER_JOB_FILE"]))
node = json.load(open(os.environ["VERIFIER_NODE_CONFIG_FILE"]))
assert node["id"] == os.environ["VERIFIER_NODE_ID"]
assert "runtime" in node["config"]
assert spec["benchmarks"][0]["concurrency"] == 2
assert spec["benchmarks"][0]["parameters"]["seed"] == 42
print(json.dumps({"schema_version": 1, "benchmarks": [
    {"id": b["id"], "status": "completed", "metrics": {"fixture_only": 1}}
    for b in spec["benchmarks"]
]}))
"""


def manifest():
    return {
        "schema_version": 1,
        "runner_script": "scripts/run_benchmarks.py",
        "engine": {"entrypoint": "engine:Engine"},
        "benchmarks": [
            {
                "id": "speed-c2",
                "workload": "host-eval",
                "concurrency": 2,
                "num_requests": 10,
                "input_tokens": 128,
                "output_tokens": 32,
                "warmup_requests": 2,
                "parameters": {"seed": 42},
            }
        ],
        "timeout_seconds": 30,
    }


def git(cwd, *args):
    return subprocess.check_output(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", *args],
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    git(repo, "init")
    (repo / "verification.json").write_text(json.dumps(manifest()))
    (repo / "scripts").mkdir()
    (repo / "scripts" / "run_benchmarks.py").write_text(FIXTURE_RUNNER)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Fixture benchmark config")
    return repo, git(repo, "rev-parse", "HEAD")


def install_local_fetch(worker, source, monkeypatch):
    original = worker.command

    def local_command(argv, *args, **kwargs):
        # Exercise real Git/worktrees offline; substitute only the remote transport.
        if "fetch" in argv:
            argv = [
                str(source)
                if arg == "https://example.test/engine.git"
                else "protocol.file.allow=always"
                if arg == "protocol.file.allow=never"
                else arg
                for arg in argv
            ]
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(worker, "command", local_command)


def test_git_manifest_job_runner_end_to_end(client, source, tmp_path, monkeypatch):
    repo, commit = source
    monkeypatch.setenv("VERIFIER_API_KEY", "must-not-reach-runner")
    job = client.post(
        "/jobs", json={"repository": "https://example.test/engine.git", "commit": commit}
    ).json()
    worker = Worker(client, "fixture-worker", tmp_path / "cache")
    install_local_fetch(worker, repo, monkeypatch)
    assert worker.once() is True
    completed = client.get(f"/jobs/{job['id']}").json()
    assert completed["status"] == "completed", completed
    assert completed["report"]["worker"]["commit"] == commit
    assert completed["node_id"] == "fixture-worker"
    assert completed["node_config"]["runtime"]["runner_python"] == sys.executable
    assert completed["report"]["worker"]["config"] == completed["node_config"]
    assert completed["report"]["worker"]["runner_script"] == "scripts/run_benchmarks.py"
    assert completed["report"]["benchmarks"][0]["id"] == "speed-c2"
    assert worker.once() is False
    assert len(list((tmp_path / "cache" / "git").rglob("HEAD"))) == 1


def test_idle_worker_registers_its_id_and_config(client, tmp_path):
    worker = Worker(
        client, "idle-node", tmp_path / "cache", node_config={"hardware": {"gpu_count": 2}}
    )
    assert worker.once() is False
    node = client.get("/nodes/idle-node").json()
    assert node["status"] == "idle"
    assert node["config"]["hardware"]["gpu_count"] == 2
    assert node["config"]["runtime"]["checkout_timeout_seconds"] == 120


def test_missing_manifest_marks_job_failed(client, source, tmp_path, monkeypatch):
    repo, commit = source
    job = client.post(
        "/jobs",
        json={
            "repository": "https://example.test/engine.git",
            "commit": commit,
            "manifest_path": "missing.json",
        },
    ).json()
    worker = Worker(client, "fixture", tmp_path / "cache")
    install_local_fetch(worker, repo, monkeypatch)
    worker.once()
    result = client.get(f"/jobs/{job['id']}").json()
    assert result["status"] == "failed"
    assert result["error"] == "invalid_or_missing_manifest"


def test_runner_failure_is_reported(client, source, tmp_path, monkeypatch):
    repo, _ = source
    (repo / "scripts" / "run_benchmarks.py").write_text("raise SystemExit(7)")
    git(repo, "commit", "-am", "Failing runner")
    commit = git(repo, "rev-parse", "HEAD")
    job = client.post(
        "/jobs", json={"repository": "https://example.test/engine.git", "commit": commit}
    ).json()
    worker = Worker(client, "fixture", tmp_path / "cache")
    install_local_fetch(worker, repo, monkeypatch)
    worker.once()
    result = client.get(f"/jobs/{job['id']}").json()
    assert result["status"] == "failed"
    assert result["error"] == "benchmark_exit_7"


def test_subprocess_timeout_stops_command(client, tmp_path, monkeypatch):
    worker = Worker(client, "fixture", tmp_path)
    monkeypatch.setattr(worker, "heartbeat", lambda job: None)
    with pytest.raises(WorkerError, match="job_timeout"):
        worker.command(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            tmp_path,
            {"id": "timeout"},
            time.monotonic() + 0.1,
            "benchmark",
            os.environ.copy(),
        )


@pytest.mark.parametrize("ids", [[], ["wrong"], ["speed-c2", "extra"], ["speed-c2", "speed-c2"]])
def test_runner_report_must_match_requested_tests(ids):
    report = {
        "schema_version": 1,
        "benchmarks": [{"id": item, "status": "completed", "metrics": {}} for item in ids],
    }
    with pytest.raises(ValueError):
        validate_report(report, BenchmarkManifest.model_validate(manifest()))


def test_manifest_requires_explicit_benchmark_settings():
    data = manifest()
    del data["benchmarks"][0]["concurrency"]
    with pytest.raises(ValueError):
        BenchmarkManifest.model_validate(data)


def test_lease_loss_stops_subprocess(client, tmp_path, monkeypatch):
    worker = Worker(client, "fixture", tmp_path)

    def rejected_heartbeat(job):
        raise WorkerError("lease_lost_or_job_cancelled")

    monkeypatch.setattr(worker, "heartbeat", rejected_heartbeat)
    with pytest.raises(WorkerError, match="lease_lost"):
        worker.command(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            tmp_path,
            {"id": "cancelled"},
            time.monotonic() + 10,
            "benchmark",
            os.environ.copy(),
        )


def test_invalid_job_runner_report_fails_job(client, source, tmp_path, monkeypatch):
    repo, _ = source
    (repo / "scripts" / "run_benchmarks.py").write_text("print('{}')")
    git(repo, "commit", "-am", "Invalid runner report")
    commit = git(repo, "rev-parse", "HEAD")
    job = client.post(
        "/jobs",
        json={
            "repository": "https://example.test/engine.git",
            "commit": commit,
        },
    ).json()
    worker = Worker(client, "fixture", tmp_path / "cache")
    install_local_fetch(worker, repo, monkeypatch)
    worker.once()
    result = client.get(f"/jobs/{job['id']}").json()
    assert result["status"] == "failed"
    assert result["error"] == "invalid_runner_report"


def test_generic_job_executes_runner_and_returns_result(client, source, tmp_path, monkeypatch):
    repo, _ = source
    config = {
        "schema_version": 1,
        "runner_script": "scripts/generic.py",
        "parameters": {"value": 42},
        "timeout_seconds": 30,
    }
    (repo / "verification.json").write_text(json.dumps(config))
    (repo / "scripts" / "generic.py").write_text("""
import json, os
spec = json.load(open(os.environ["VERIFIER_JOB_FILE"]))
print(json.dumps({"schema_version": 1, "result": {"observed": spec["parameters"]["value"]}}))
""")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Generic job runner")
    job = client.post(
        "/jobs",
        json={
            "repository": "https://example.test/engine.git",
            "commit": git(repo, "rev-parse", "HEAD"),
        },
    ).json()
    worker = Worker(client, "mechanical-node", tmp_path / "cache")
    install_local_fetch(worker, repo, monkeypatch)
    worker.once()
    completed = client.get(f"/jobs/{job['id']}").json()
    assert completed["status"] == "completed", completed
    assert completed["report"]["result"] == {"observed": 42}


def test_manifest_requires_runner_script():
    data = manifest()
    del data["runner_script"]
    with pytest.raises(ValueError, match="runner_script"):
        BenchmarkManifest.model_validate(data)


@pytest.mark.parametrize("path", ["", "../runner.py", "/tmp/runner.py", "..\\runner.py", "a\0b"])
def test_invalid_runner_paths_rejected(path):
    with pytest.raises(ValueError):
        BenchmarkManifest.model_validate({**manifest(), "runner_script": path})


@pytest.mark.parametrize("runner_path", ["missing.py", "scripts"])
def test_missing_runner_or_directory_fails_before_execution(
    client, source, tmp_path, monkeypatch, runner_path
):
    repo, _ = source
    (repo / "verification.json").write_text(
        json.dumps({**manifest(), "runner_script": runner_path})
    )
    git(repo, "commit", "-am", "Invalid runner path")
    job = client.post(
        "/jobs",
        json={
            "repository": "https://example.test/engine.git",
            "commit": git(repo, "rev-parse", "HEAD"),
        },
    ).json()
    worker = Worker(client, "fixture", tmp_path / "cache")
    install_local_fetch(worker, repo, monkeypatch)
    worker.once()
    result = client.get(f"/jobs/{job['id']}").json()
    assert result["status"] == "failed"
    assert result["error"] == "runner_script_missing_or_not_file"
    assert not (tmp_path / "cache" / "jobs" / job["id"] / "benchmark.stdout").exists()


def test_runner_symlink_must_stay_inside_checkout(client, source, tmp_path, monkeypatch):
    repo, _ = source
    outside = tmp_path / "outside.py"
    outside.write_text("raise RuntimeError('Must not execute')")
    runner = repo / "scripts" / "run_benchmarks.py"
    runner.unlink()
    runner.symlink_to(outside)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Runner symlink")
    job = client.post(
        "/jobs",
        json={
            "repository": "https://example.test/engine.git",
            "commit": git(repo, "rev-parse", "HEAD"),
        },
    ).json()
    worker = Worker(client, "fixture", tmp_path / "cache")
    install_local_fetch(worker, repo, monkeypatch)
    worker.once()
    result = client.get(f"/jobs/{job['id']}").json()
    assert result["status"] == "failed"
    assert result["error"] == "runner_outside_checkout"


def test_each_job_uses_its_committed_runner(client, source, tmp_path, monkeypatch):
    repo, first_commit = source
    (repo / "scripts" / "second.py").write_text(
        FIXTURE_RUNNER.replace('"fixture_only": 1', '"fixture_only": 2')
    )
    (repo / "verification.json").write_text(
        json.dumps({**manifest(), "runner_script": "scripts/second.py"})
    )
    git(repo, "add", ".")
    git(repo, "commit", "-m", "Second job runner")
    second_commit = git(repo, "rev-parse", "HEAD")
    jobs = [
        client.post(
            "/jobs",
            json={
                "repository": "https://example.test/engine.git",
                "commit": commit,
            },
        ).json()
        for commit in (first_commit, second_commit)
    ]
    worker = Worker(client, "fixture", tmp_path / "cache")
    install_local_fetch(worker, repo, monkeypatch)
    for job, expected in zip(jobs, (1, 2)):
        worker.once()
        result = client.get(f"/jobs/{job['id']}").json()
        assert result["status"] == "completed", result
        assert result["report"]["benchmarks"][0]["metrics"]["fixture_only"] == expected
