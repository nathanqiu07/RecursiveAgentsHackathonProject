"""Host worker for trusted Git submissions. Run on the benchmark machine."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from .jobs import JobManifest, JobSubmission, NodeRegistration, validate_report


class WorkerError(Exception):
    pass


class Worker:
    def __init__(
        self,
        client: httpx.Client,
        worker_id: str,
        cache: Path,
        runner_python: str = sys.executable,
        checkout_timeout: int = 120,
        node_config: dict | None = None,
    ):
        self.client = client
        self.worker_id = worker_id
        self.cache = cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        if not isinstance(runner_python, str) or not runner_python:
            raise ValueError("runner_python must be a Python interpreter path or command")
        if checkout_timeout <= 0:
            raise ValueError("checkout_timeout must be positive")
        self.runner_python = runner_python
        self.checkout_timeout = checkout_timeout
        config = dict(node_config or {})
        runtime = config.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("Node runtime config must be an object")
        config["runtime"] = {
            **runtime,
            "runner_python": runner_python,
            "checkout_timeout_seconds": checkout_timeout,
            "cache_dir": str(self.cache),
        }
        self.registration = NodeRegistration(id=worker_id, config=config)
        self.registered = False

    def refresh_node(self):
        if not self.registered:
            response = self.client.post("/nodes/register", json=self.registration.model_dump())
        else:
            response = self.client.post(f"/nodes/{self.worker_id}/heartbeat")
            if response.status_code == 404:
                self.registered = False
                return self.refresh_node()
        response.raise_for_status()
        self.registered = True

    def heartbeat(self, job):
        response = self.client.post(
            f"/worker/jobs/{job['id']}/heartbeat", json={"lease_token": job["lease_token"]}
        )
        if response.status_code != 200:
            raise WorkerError("lease_lost_or_job_cancelled")

    @staticmethod
    def terminate(process):
        # Stop the entire build/benchmark process group, including spawned engine processes.
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        except ProcessLookupError:
            process.wait()
            return
        # A parent may exit while its engine child ignores SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def command(self, argv, cwd, job, deadline, stage, env, capture=False):
        logs = self.cache / "jobs" / job["id"]
        logs.mkdir(parents=True, exist_ok=True)
        stdout_path = logs / f"{stage}.stdout"
        stderr_path = logs / f"{stage}.stderr"
        next_heartbeat = 0
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except OSError as exc:
                raise WorkerError(f"{stage}_could_not_start") from exc
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise WorkerError("job_timeout")
                    if time.monotonic() >= next_heartbeat:
                        self.heartbeat(job)
                        next_heartbeat = time.monotonic() + 20
                    if stdout_path.stat().st_size > (2 if capture else 16) * 1024 * 1024:
                        raise WorkerError(f"{stage}_output_too_large")
                    if stderr_path.stat().st_size > 16 * 1024 * 1024:
                        raise WorkerError(f"{stage}_log_too_large")
                    time.sleep(0.1)
                if process.returncode != 0:
                    raise WorkerError(f"{stage}_exit_{process.returncode}")
            finally:
                # Also clean up children after a successful benchmark command exits.
                self.terminate(process)
        limit = (2 if capture else 16) * 1024 * 1024
        if stdout_path.stat().st_size > limit or stderr_path.stat().st_size > 16 * 1024 * 1024:
            raise WorkerError(f"{stage}_output_too_large")
        return stdout_path.read_text() if capture else None

    def execute(self, job):
        submission = JobSubmission.model_validate(job["submission"])
        started = time.monotonic()
        deadline = started + self.checkout_timeout
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        # The API's access token is control-plane authentication, not a benchmark credential.
        env.pop("VERIFIER_API_KEY", None)
        repository = str(submission.repository)
        repo_key = hashlib.sha256(repository.encode()).hexdigest()
        # Each worker owns its Git cache. A worker processes only one job at a time.
        worker_key = hashlib.sha256(self.worker_id.encode()).hexdigest()[:16]
        bare = self.cache / "git" / worker_key / repo_key
        bare.parent.mkdir(parents=True, exist_ok=True)
        git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]

        def run(argv, stage, cwd=None, capture=False):
            return self.command(argv, cwd, job, deadline, stage, env, capture)

        object_format = "sha256" if len(submission.commit) == 64 else "sha1"
        run(git + ["init", "--bare", f"--object-format={object_format}", str(bare)], "git-init")
        run(
            git
            + [
                "--git-dir",
                str(bare),
                "fetch",
                "--no-tags",
                "--depth=1",
                repository,
                f"{submission.commit}:refs/verification/{submission.commit}",
            ],
            "git-fetch",
        )
        resolved = run(
            git + ["--git-dir", str(bare), "rev-parse", f"{submission.commit}^{{commit}}"],
            "git-resolve",
            capture=True,
        ).strip()
        if resolved.lower() != submission.commit.lower():
            raise WorkerError("git_commit_mismatch")

        with tempfile.TemporaryDirectory(prefix="inference-benchmark-") as temporary:
            checkout = Path(temporary) / "source"
            run(
                git
                + ["--git-dir", str(bare), "worktree", "add", "--detach", str(checkout), resolved],
                "git-checkout",
            )
            try:
                config_path = (checkout / submission.manifest_path).resolve()
                if not config_path.is_relative_to(checkout.resolve()):
                    raise WorkerError("manifest_outside_checkout")
                try:
                    if config_path.stat().st_size > 1024 * 1024:
                        raise WorkerError("manifest_too_large")
                    raw_manifest = config_path.read_bytes()
                    specification = JobManifest.model_validate_json(raw_manifest)
                except (ValueError, OSError) as exc:
                    raise WorkerError("invalid_or_missing_manifest") from exc
                runner = (checkout / specification.runner_script).resolve()
                if not runner.is_relative_to(checkout.resolve()):
                    raise WorkerError("runner_outside_checkout")
                if not runner.is_file():
                    raise WorkerError("runner_script_missing_or_not_file")
                manifest = Path(temporary) / "job.json"
                manifest.write_text(specification.model_dump_json(indent=2))
                env["VERIFIER_JOB_FILE"] = str(manifest)
                env["VERIFIER_CHECKOUT_DIR"] = str(checkout)
                node_config = Path(temporary) / "node.json"
                node_config.write_text(
                    json.dumps(
                        {
                            "id": self.worker_id,
                            "config": job["node_config"],
                        }
                    )
                )
                env["VERIFIER_NODE_ID"] = self.worker_id
                env["VERIFIER_NODE_CONFIG_FILE"] = str(node_config)
                env["VERIFIER_CACHE_DIR"] = str(self.cache / "artifacts")
                Path(env["VERIFIER_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
                setup_seconds = time.monotonic() - started
                deadline = time.monotonic() + specification.timeout_seconds
                output = run(
                    [self.runner_python, str(runner)], "benchmark", cwd=checkout, capture=True
                )
                try:
                    report = json.loads(output)
                    report = validate_report(report, specification)
                except (ValueError, KeyError, TypeError, RecursionError) as exc:
                    raise WorkerError("invalid_runner_report") from exc
                report["worker"] = {
                    "id": self.worker_id,
                    "node_id": self.worker_id,
                    "config": job["node_config"],
                    "commit": resolved,
                    "setup_seconds": round(setup_seconds, 6),
                    "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                    "runner_script": specification.runner_script,
                }
                return report
            finally:
                # Remove the disposable checkout; keep Git objects and host artifacts cached.
                subprocess.run(
                    git + ["--git-dir", str(bare), "worktree", "remove", "--force", str(checkout)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )

    def once(self):
        self.refresh_node()
        response = self.client.post("/worker/claim", json={"node_id": self.worker_id})
        response.raise_for_status()
        job = response.json()["job"]
        if job is None:
            return False
        outcome = {"lease_token": job["lease_token"]}
        try:
            self.heartbeat(job)
            outcome.update(status="completed", report=self.execute(job))
        except (WorkerError, OSError, httpx.HTTPError, subprocess.SubprocessError) as exc:
            error = str(exc) if isinstance(exc, WorkerError) else "worker_execution_error"
            outcome.update(status="failed", error=error)
        response = self.client.post(f"/worker/jobs/{job['id']}/finish", json=outcome)
        if response.status_code != 409:  # Cancellation/expired leases cannot be overwritten.
            response.raise_for_status()
        return True


def run_worker(worker, once=False):
    while True:
        try:
            had_job = worker.once()
        except httpx.HTTPError as exc:
            print(f"Queue connection failed ({type(exc).__name__}); retrying.", flush=True)
            had_job = False
        if once:
            break
        if not had_job:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--node-id", "--worker-id", dest="node_id")
    parser.add_argument(
        "--node-config", type=Path, help="JSON file containing the node id and config"
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/worker-cache"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--runner-python",
        default=sys.executable,
        help="Python interpreter for each job's committed runner script",
    )
    parser.add_argument("--checkout-timeout", type=int, default=120)
    args = parser.parse_args()
    registration = None
    if args.node_config:
        registration = NodeRegistration.model_validate_json(args.node_config.read_bytes())
        if args.node_id and args.node_id != registration.id:
            parser.error("--node-id must match the ID in --node-config")
    node_id = args.node_id or (registration.id if registration else socket.gethostname())
    headers = {}
    if os.environ.get("VERIFIER_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['VERIFIER_API_KEY']}"
    with httpx.Client(
        base_url=args.api_url, headers=headers, timeout=30, trust_env=False
    ) as client:
        worker = Worker(
            client,
            node_id,
            args.cache_dir,
            args.runner_python,
            args.checkout_timeout,
            registration.config if registration else None,
        )
        run_worker(worker, once=args.once)


if __name__ == "__main__":
    main()
