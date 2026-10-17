"""Check a machine, synchronize its node config, then run the mechanical job executor."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

from .jobs import NodeInfo, NodeList, NodeRegistration
from .worker import Worker, run_worker

GIB = 1024**3


class SetupError(Exception):
    pass


def command(argv, timeout=20):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SetupError(f"{Path(argv[0]).name} could not run or timed out") from exc
    if result.returncode:
        raise SetupError(f"{Path(argv[0]).name} failed (exit {result.returncode})")
    return result.stdout.strip()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(data, file, indent=2, allow_nan=False)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


@contextmanager
def state_lock(directory):
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "node.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SetupError("A setup or node process already owns this state directory") from exc
        yield


def machine_id():
    machine_file = Path("/etc/machine-id")
    identity = machine_file.read_text().strip() if machine_file.is_file() else socket.gethostname()
    return hashlib.sha256(identity.encode()).hexdigest()


def python_probe(python):
    executable = shutil.which(python)
    if not executable:
        raise SetupError("Runner Python was not found")
    data = json.loads(
        command(
            [
                executable,
                "-I",
                "-c",
                "import json,sys; print(json.dumps({'version':list(sys.version_info[:3]),"
                "'executable':sys.executable}))",
            ]
        )
    )
    if tuple(data["version"]) < (3, 10):
        raise SetupError("Runner Python must be version 3.10 or later")
    return data


def git_probe(cache):
    git = shutil.which("git")
    if not git:
        raise SetupError("Git is not installed")
    prefix = [
        git,
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "user.name=Node readiness",
        "-c",
        "user.email=node@localhost",
    ]
    with tempfile.TemporaryDirectory(dir=cache) as temporary:
        root = Path(temporary)
        repo, checkout = root / "repo", root / "checkout"
        command(prefix + ["init", "--object-format=sha1", str(repo)])
        (repo / "probe.txt").write_text("node-ready")
        command(prefix + ["-C", str(repo), "add", "probe.txt"])
        command(prefix + ["-C", str(repo), "commit", "-m", "Readiness check"])
        command(prefix + ["-C", str(repo), "worktree", "add", "--detach", str(checkout)])
        if (checkout / "probe.txt").read_text() != "node-ready":
            raise SetupError("Git checkout contents did not match")
        command(prefix + ["-C", str(repo), "worktree", "remove", str(checkout)])
        command(prefix + ["init", "--bare", "--object-format=sha256", str(root / "sha256")])
    return command([git, "--version"])


def ram_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        if platform.system() == "Darwin":
            return int(command(["sysctl", "-n", "hw.memsize"]))
        raise SetupError("Could not detect system RAM")


def gpu_probe(python):
    if platform.system() != "Linux":
        raise SetupError("GPU mode requires Linux with NVIDIA CUDA drivers")
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise SetupError("nvidia-smi is missing; install the NVIDIA driver first")
    inventory = command(
        [
            smi,
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = list(csv.reader(inventory.splitlines(), skipinitialspace=True))
    if not rows or any(len(row) != 4 for row in rows):
        raise SetupError("NVIDIA GPU inventory could not be read")
    probe_file = Path(__file__).with_name("cuda_probe.py")
    result = json.loads(command([python, "-I", str(probe_file)], timeout=60))
    if not result.get("gpus") or any(gpu.get("compute_test") != "passed" for gpu in result["gpus"]):
        raise SetupError("CUDA compute smoke test did not pass")
    return {
        "gpus": result["gpus"],
        "driver_cuda_version": result["driver_cuda_version"],
        "nvidia_smi_inventory": [
            dict(zip(("uuid", "name", "memory_mib", "driver_version"), row)) for row in rows
        ],
    }


def collect_machine(args, cache, checks):
    def check(name, action):
        try:
            result = action()
        except (SetupError, OSError, ValueError, KeyError) as exc:
            checks.append({"name": name, "status": "failed", "detail": str(exc)})
            print(f"[FAIL] {name}: {exc}", flush=True)
            return None
        checks.append({"name": name, "status": "passed"})
        print(f"[PASS] {name}", flush=True)
        return result

    def operating_system():
        if platform.system() not in {"Linux", "Darwin"}:
            raise SetupError("The node executor requires Linux or macOS")
        return {
            "os": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
        }

    def storage():
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=cache) as file:
            file.write(b"node-ready")
            file.flush()
            os.fsync(file.fileno())
        available = shutil.disk_usage(cache).free
        if available < args.min_free_disk_gb * GIB:
            raise SetupError("Free disk space is below --min-free-disk-gb")
        return available

    def memory():
        total = ram_bytes()
        if total < args.min_ram_gb * GIB:
            raise SetupError("System RAM is below --min-ram-gb")
        return total

    software = check("operating-system", operating_system) or {}
    python = check("runner-python", lambda: python_probe(args.runner_python))
    disk = check("writable-cache-and-disk", storage)
    ram = check("system-memory", memory)
    git = check("git-checkout", lambda: git_probe(cache)) if disk is not None else None
    hardware = {"cpu_cores": os.cpu_count(), "ram_bytes": ram, "free_disk_bytes": disk}
    if python:
        software.update(python_version=".".join(map(str, python["version"])), git_version=git)
    if args.require_gpu and python:
        gpu = check("cuda-compute", lambda: gpu_probe(python["executable"]))
        if gpu:
            hardware.update(gpu_count=len(gpu["gpus"]), gpus=gpu["gpus"])
            software.update(
                driver_cuda_version=gpu["driver_cuda_version"],
                nvidia_smi_inventory=gpu["nvidia_smi_inventory"],
            )
    if any(item["status"] == "failed" for item in checks):
        raise SetupError("Machine checks failed; the node was not registered")
    return {"machine_id": machine_id(), "hardware": hardware, "software": software}, python[
        "executable"
    ]


def server_check(client):
    try:
        health = client.get("/health")
        health.raise_for_status()
        if health.json().get("status") != "ok":
            raise SetupError("Server health response is invalid")
        nodes = client.get("/nodes")
        nodes.raise_for_status()
        NodeList.model_validate(nodes.json())
    except httpx.HTTPStatusError as exc:
        raise SetupError(f"Server access check failed (HTTP {exc.response.status_code})") from exc
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        raise SetupError("Could not reach a compatible verification server") from exc


def synchronize(client, worker):
    existing = client.get(f"/nodes/{worker.worker_id}")
    if existing.status_code == 200:
        node = NodeInfo.model_validate(existing.json())
        previous_machine = node.config.get("machine_id")
        if previous_machine and previous_machine != worker.registration.config["machine_id"]:
            raise SetupError(
                "Node ID belongs to another machine; use a different node ID/state directory"
            )
        if node.status == "busy":
            raise SetupError("Node ID already has a running job")
    elif existing.status_code != 404:
        existing.raise_for_status()
    worker.refresh_node()
    heartbeat = client.post(f"/nodes/{worker.worker_id}/heartbeat")
    heartbeat.raise_for_status()
    response = client.get(f"/nodes/{worker.worker_id}")
    response.raise_for_status()
    node = NodeInfo.model_validate(response.json())
    if node.id != worker.worker_id or node.config != worker.registration.config:
        raise SetupError("Server config does not match the local node config")


def enroll(args, client=None):
    state = args.state_dir.expanduser().resolve()
    if os.name != "posix":
        raise SetupError("The node executor requires a POSIX machine")
    with state_lock(state):
        config_file = state / "node.json"
        previous = (
            NodeRegistration.model_validate_json(config_file.read_bytes())
            if config_file.exists()
            else None
        )
        if previous and args.node_id and args.node_id != previous.id:
            raise SetupError("State directory belongs to another node ID; use a new --state-dir")
        node_id = args.node_id or (previous.id if previous else f"node-{uuid4().hex[:12]}")
        NodeRegistration(id=node_id, config={})
        checks = []
        report = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "node_id": node_id,
            "checks": checks,
            "registered": False,
        }
        try:
            config, python = collect_machine(args, state / "cache", checks)
            config["labels"] = dict(previous.config.get("labels", {}) if previous else {})
            for label in args.label:
                key, separator, value = label.partition("=")
                if not separator or not key:
                    raise SetupError("Labels must use KEY=VALUE")
                config["labels"][key] = value
            config["readiness"] = {
                "checked_at": report["checked_at"],
                "checks": list(checks),
                "gpu_required": args.require_gpu,
            }
            headers = {}
            if os.environ.get("VERIFIER_API_KEY"):
                headers["Authorization"] = f"Bearer {os.environ['VERIFIER_API_KEY']}"
            context = (
                nullcontext(client)
                if client is not None
                else httpx.Client(
                    base_url=args.server,
                    headers=headers,
                    timeout=15,
                    trust_env=False,
                )
            )
            with context as connection:
                server_check(connection)
                checks.append({"name": "server-access", "status": "passed"})
                print("[PASS] server access and authentication", flush=True)
                worker = Worker(
                    connection,
                    node_id,
                    state / "cache",
                    python,
                    args.checkout_timeout,
                    node_config=config,
                )
                if args.check_only:
                    print("Checks passed; no node was registered.", flush=True)
                    return
                synchronize(connection, worker)
                write_json(config_file, worker.registration.model_dump())
                report["registered"] = True
                checks.append({"name": "server-sync", "status": "passed"})
                write_json(state / "preflight.json", report)
                print(
                    f"[PASS] node {node_id} synchronized; config saved to {config_file}", flush=True
                )
                if not args.enroll_only:
                    print(
                        "Node is ready. Waiting for jobs to execute through their runner scripts.",
                        flush=True,
                    )
                    run_worker(worker)
        except (SetupError, httpx.HTTPError, OSError, ValueError) as exc:
            # Do not echo HTTP bodies, credentials, or request headers.
            detail = str(exc) if isinstance(exc, SetupError) else type(exc).__name__
            checks.append({"name": "setup", "status": "failed", "detail": detail})
            raise SetupError(detail) from exc
        finally:
            write_json(state / "preflight.json", report)


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Must be a finite nonnegative number")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--server", required=True)
    result.add_argument("--node-id")
    result.add_argument("--state-dir", type=Path, default=Path("data/node"))
    result.add_argument("--runner-python", default=sys.executable)
    result.add_argument("--checkout-timeout", type=int, default=120)
    result.add_argument("--label", action="append", default=[])
    result.add_argument("--min-free-disk-gb", type=nonnegative, default=1)
    result.add_argument("--min-ram-gb", type=nonnegative, default=1)
    result.add_argument("--require-gpu", action="store_true")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--enroll-only", action="store_true")
    return result


def main():
    args = parser().parse_args()
    url = httpx.URL(args.server)
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.userinfo
        or url.query
        or url.fragment
    ):
        raise SystemExit("--server must be an HTTP(S) URL without credentials, query, or fragment")
    try:
        enroll(args)
    except (SetupError, OSError, ValueError) as exc:
        print(f"Node setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("Node stopped.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
