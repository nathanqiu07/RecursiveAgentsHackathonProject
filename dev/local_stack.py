#!/usr/bin/env python3
"""Run the control plane on this computer; remote resource VMs only run jobs.

No third-party Python packages are needed by this launcher. Component environments
must already be installed; `check` and `up --dry-run` never install or run a job.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import ipaddress
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dev" / "data"
OPENER = build_opener(ProxyHandler({}))
LOCAL_LINKS = {
    "AGENT_HARNESS_URL": "harness-api",
    "HARNESS_LIBRARY_URL": "library-api",
    "HARNESS_BENCHMARK_URL": "dashboard-api",
    "HARNESS_RESOURCE_MANAGER_URL": "resource-manager-api",
    "LIBRARY_RESOURCE_MANAGER_URL": "resource-manager-api",
    "LIBRARY_VERIFICATION_API_URL": "verification-api",
    "NEMOTRON_VERIFIER_URL": "nemotron-api",
}


class StackError(Exception):
    pass


@dataclass(frozen=True)
class Service:
    name: str
    cwd: Path
    command: tuple
    port: int | None = None
    health_path: str = "/health"
    title: str | None = None
    worker_module: str | None = None
    worker_pattern: str | None = None
    prerequisites: tuple = ()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}" if self.port else None


def private_read(path):
    """Credentials/state are ordinary private files, never followed symlinks."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise StackError(f"{path}: expected a regular file with permissions 0600")
        if info.st_uid != os.getuid() or info.st_size > 262144:
            raise StackError(f"{path}: invalid file owner or size")
        return stream.read()


def read_env(path):
    source = private_read(path)
    result = {}
    for line in (source or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise StackError(f"{path}: expected KEY=VALUE; no shell expressions")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise StackError(f"{path}: invalid environment variable name")
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise StackError(f"{path}: unmatched quote")
            value = value[1:-1]
        result[key] = value
    return result


def ensure_private_dir(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise StackError(f"{path}: expected an owned directory")
    path.chmod(0o700)


def initialize(path):
    if path.exists():
        read_env(path)
        return
    ensure_private_dir(path.parent)
    token = secrets.token_urlsafe(32)
    body = (
        "# Local only. Values are literal; this file is never sourced as shell code.\n"
        f"NEMOTRON_VERIFIER_API_KEY={token}\n"
        f"BENCHMARK_VERIFIER_NEMOTRON_KEY={token}\n"
        "LIBRARY_PLANNER_BACKEND=codex\n"
    )
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        read_env(path)
        return
    with os.fdopen(fd, "w") as stream:
        stream.write(body)


def atomic_json(path, value):
    ensure_private_dir(path.parent)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".stack-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_state(path):
    source = private_read(path)
    if source is None:
        return {"version": 1, "processes": {}}
    try:
        value = json.loads(source)
        if value["version"] != 1 or not isinstance(value["processes"], dict):
            raise ValueError
        return value
    except (ValueError, KeyError, TypeError):
        raise StackError(
            f"{path}: invalid state; refusing to guess process ownership"
        ) from None


@contextlib.contextmanager
def lock(data):
    ensure_private_dir(data)
    fd = os.open(
        data / "stack.lock",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(fd, "w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise StackError(
                "Another local stack command is already changing process state"
            ) from None
        yield


def local_url(value, name):
    parsed = urlsplit(value)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
        port = parsed.port
    except ValueError:
        loopback, port = False, None
    if (
        parsed.scheme != "http"
        or not loopback
        or not port
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise StackError(
            f"{name} must be an explicit HTTP loopback address with a port"
        )
    return value.rstrip("/")


def port_number(env, name, default):
    try:
        port = int(env.get(name, default))
        if not 1 <= port <= 65535:
            raise ValueError
        return port
    except ValueError:
        raise StackError(f"{name} must be a port from 1 to 65535") from None


def definitions(root, env):
    node = env.get("LOCAL_STACK_NODE_BIN") or shutil.which("node") or "node"
    ports = {
        "dashboard-api": port_number(env, "DASHBOARD_API_PORT", 3001),
        "dashboard-web": port_number(env, "LOCAL_STACK_WEB_PORT", 5173),
        "library-api": port_number(env, "LOCAL_STACK_LIBRARY_PORT", 8001),
        "resource-manager-api": port_number(
            env, "LOCAL_STACK_RESOURCE_MANAGER_PORT", 8002
        ),
        "harness-api": port_number(env, "LOCAL_STACK_HARNESS_PORT", 8003),
        "verification-api": port_number(env, "LOCAL_STACK_VERIFICATION_PORT", 8000),
        "nemotron-api": port_number(env, "LOCAL_STACK_NEMOTRON_PORT", 9000),
    }
    if len(set(ports.values())) != len(ports):
        raise StackError("Local service ports must be distinct")

    def python(directory):
        override = env.get(
            "LOCAL_STACK_" + directory.upper().replace(" ", "_") + "_PYTHON"
        )
        return override or str(root / directory / ".venv" / "bin" / "python")

    services = []

    def api(name, directory, module, title):
        services.append(
            Service(
                name,
                root / directory,
                (
                    python(directory),
                    "-m",
                    "uvicorn",
                    module + ":app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(ports[name]),
                ),
                ports[name],
                title=title,
            )
        )

    api(
        "resource-manager-api",
        "ResourceManager",
        "resource_manager.main",
        "Resource Manager",
    )
    api(
        "verification-api",
        "verification api",
        "verification_api.main",
        "Verification API",
    )
    api("library-api", "library", "paper_library.main", "Library")
    api("harness-api", "harness", "agent_harness.main", "Agent Harness")
    api("nemotron-api", "verifier", "nemotron_verifier.main", "Nemotron")
    services.append(
        Service(
            "dashboard-api",
            root / "dashboard",
            (node, "--env-file-if-exists=.env", "server/index.ts"),
            ports["dashboard-api"],
            "/api/health",
        )
    )
    services.append(
        Service(
            "dashboard-web",
            root / "dashboard",
            (
                node,
                "node_modules/vite/bin/vite.js",
                "--host",
                "127.0.0.1",
                "--port",
                str(ports["dashboard-web"]),
                "--strictPort",
            ),
            ports["dashboard-web"],
            "/",
        )
    )

    def worker(name, directory, module, dependencies, extra=(), interpreter=None):
        services.append(
            Service(
                name,
                root / directory,
                (interpreter or python(directory), "-m", module, *extra),
                worker_module=module,
                prerequisites=dependencies,
            )
        )

    worker(
        "resource-manager-worker",
        "ResourceManager",
        "resource_manager.worker",
        ("resource-manager-api",),
    )
    worker(
        "verification-worker",
        "verification api",
        "verification_api.worker",
        ("verification-api",),
        (
            "--api-url",
            f"http://127.0.0.1:{ports['verification-api']}",
            "--node-id",
            "local-verifier",
        ),
    )
    worker(
        "library-worker",
        "library",
        "paper_library.worker",
        ("library-api", "verification-api", "resource-manager-api"),
        interpreter=python("harness"),
    )
    worker(
        "harness-worker",
        "harness",
        "agent_harness.worker",
        ("harness-api", "resource-manager-api"),
    )
    worker("nemotron-worker", "verifier", "nemotron_verifier.worker", ("nemotron-api",))
    if env.get("LOCAL_STACK_NODE_STATE"):
        state = Path(env["LOCAL_STACK_NODE_STATE"]).expanduser()
        if not state.is_absolute():
            raise StackError(
                "LOCAL_STACK_NODE_STATE must be an absolute local node-state directory"
            )
        controller = str(root / "nodes/controller.py")
        services.append(
            Service(
                "node-heartbeat",
                root,
                (
                    python("harness"),
                    controller,
                    "heartbeat",
                    "--state",
                    str(state),
                    "--resources-dir",
                    str(
                        env.get("RESOURCE_MANAGER_RESOURCES_DIR") or root / "Resources"
                    ),
                    "--resource-manager",
                    f"http://127.0.0.1:{ports['resource-manager-api']}",
                ),
                worker_pattern=re.escape(controller) + r"\s+heartbeat(?:\s|$)",
                prerequisites=("resource-manager-api",),
            )
        )
    return services


def environment(root, env_path):
    # Component .env files are intentionally not loaded implicitly. One private
    # file keeps API/worker keys and databases consistent. Shell exports win.
    env = {**read_env(env_path), **os.environ}
    services = definitions(root, env)
    by_name = {service.name: service for service in services}
    for variable, name in LOCAL_LINKS.items():
        expected = by_name[name].url
        configured = env.get(variable)
        if configured and local_url(configured, variable) != expected:
            raise StackError(
                f"{variable} must match the local {name} endpoint {expected}"
            )
        env[variable] = expected
    env["DASHBOARD_API_PORT"] = str(by_name["dashboard-api"].port)
    for variable, consumer in (
        ("HARNESS_API_KEY", "AGENT_HARNESS_API_KEY"),
        ("LIBRARY_API_KEY", "HARNESS_LIBRARY_API_KEY"),
        ("DASHBOARD_API_KEY", "HARNESS_BENCHMARK_API_KEY"),
        ("RESOURCE_MANAGER_API_KEY", "HARNESS_RESOURCE_MANAGER_API_KEY"),
        ("RESOURCE_MANAGER_API_KEY", "LIBRARY_RESOURCE_MANAGER_API_KEY"),
        ("VERIFIER_API_KEY", "LIBRARY_VERIFICATION_API_KEY"),
        ("NEMOTRON_VERIFIER_API_KEY", "BENCHMARK_VERIFIER_NEMOTRON_KEY"),
    ):
        if env.get(variable):
            if env.get(consumer) and env[consumer] != env[variable]:
                raise StackError(f"{consumer} must match {variable}")
            env[consumer] = env[variable]
    defaults = {
        "HARNESS_DB_PATH": root / "harness/data/harness.sqlite3",
        "LIBRARY_DB_PATH": root / "library/data/library.sqlite3",
        "RESOURCE_MANAGER_DB_PATH": root
        / "ResourceManager/data/resource-manager.sqlite3",
        "VERIFIER_DB_PATH": root / "verification api/data/jobs.sqlite3",
        "NEMOTRON_VERIFIER_DB": root / "verifier/data/verifier.sqlite",
        "NEMOTRON_VERIFIER_CONFIG": root / "verifier/config.json",
        "RESOURCE_MANAGER_RESOURCES_DIR": root / "Resources",
    }
    for name, value in defaults.items():
        configured = Path(env.get(name) or value).expanduser()
        if not configured.is_absolute():
            raise StackError(
                f"{name} must be absolute so APIs and workers share the same files"
            )
        env[name] = str(configured)
    env.setdefault("LIBRARY_PLANNER_BACKEND", "codex")
    env["PYTHONUNBUFFERED"] = "1"
    # Library's Codex adapter imports the local harness; no OAuth material is
    # opened or copied. Both run on the operator's computer.
    env["PYTHONPATH"] = os.pathsep.join(
        str(root / directory)
        for directory in (
            "harness",
            "library",
            "ResourceManager",
            "verification api",
            "verifier",
        )
    )
    return env, services


def request_json(url, key=None):
    headers = {"Authorization": "Bearer " + key} if key else {}
    with OPENER.open(Request(url, headers=headers), timeout=2) as response:
        raw = response.read(1048577)
        if len(raw) > 1048576:
            raise ValueError("Response too large")
        return json.loads(raw)


def healthy(service):
    if service.port is None:
        return False
    try:
        if service.name == "dashboard-web":
            with OPENER.open(service.url + "/", timeout=2) as response:
                body = response.read(1048576).decode("utf-8")
                return "/@vite/client" in body and "/src/main.tsx" in body
        health = request_json(service.url + service.health_path)
        if health.get("status") != "ok":
            return False
        if service.title:
            title = request_json(service.url + "/openapi.json")["info"]["title"]
            return service.title in title
        return health.get("database") == "sqlite"
    except (OSError, URLError, ValueError, KeyError, TypeError):
        return False


def occupied(port):
    with socket.socket() as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def identity(pid, nonce=None):
    if type(pid) is not int or pid <= 1:
        return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    value = result.stdout.strip() if result.returncode == 0 else None
    if nonce and value:
        # macOS's Python launcher can replace argv[0] during startup. The
        # persistent wrapper carries a unique marker that survives that exec
        # and cannot match a later process which happens to reuse this PID.
        if not re.search(r"\s_child\s+" + re.escape(nonce) + r"(?:\s|$)", value):
            return None
        parts = value.split(None, 5)
        return " ".join(parts[:5]) + "|" + nonce if len(parts) == 6 else None
    return value


def owned(record):
    try:
        return (
            isinstance(record, dict)
            and bool(record.get("identity"))
            and bool(record.get("nonce"))
            and identity(record.get("pid"), record["nonce"]) == record["identity"]
            and os.getpgid(record["pid"]) == record["pid"]
        )
    except ProcessLookupError:
        return False


def process_cwd(pid):
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    except OSError:
        binary = shutil.which("lsof") or "/usr/sbin/lsof"
        try:
            result = subprocess.run(
                [binary, "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            return next(
                Path(line[1:]).resolve()
                for line in result.stdout.splitlines()
                if line.startswith("n")
            )
        except (OSError, subprocess.TimeoutExpired, StopIteration):
            return None


def existing_worker(service):
    if not service.worker_module and not service.worker_pattern:
        return None
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    pattern = re.compile(
        service.worker_pattern
        or (r"(?:^|\s)-m\s+" + re.escape(service.worker_module) + r"(?:\s|$)")
    )
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and pattern.search(parts[1]):
            pid = int(parts[0])
            if process_cwd(pid) == service.cwd.resolve():
                return pid
    return None


def installed(service):
    binary = service.command[0]
    found = (
        os.access(binary, os.X_OK) if os.path.isabs(binary) else shutil.which(binary)
    )
    if not service.cwd.is_dir() or not found:
        return False
    if service.name == "dashboard-web":
        return (service.cwd / "node_modules/vite/bin/vite.js").is_file()
    return True


def runtime_check(service, env):
    if not installed(service):
        return False
    try:
        if service.name.startswith("dashboard-"):
            result = subprocess.run(
                [service.command[0], "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                env=env,
                check=False,
            )
            return (
                result.returncode == 0
                and int(result.stdout.strip().lstrip("v").split(".")[0]) >= 24
            )
        modules = ["uvicorn", "fastapi", "httpx"]
        if service.name in {"harness-api", "harness-worker", "library-worker"}:
            modules += ["cryptography", "jsonschema"]
        version = (3, 12) if service.name.startswith("nemotron-") else (3, 10)
        program = (
            "import importlib.util,json,sys; "
            "modules,version=json.loads(sys.argv[1]); "
            "sys.exit(0 if sys.version_info >= tuple(version) and "
            "all(importlib.util.find_spec(name) is not None for name in modules) else 1)"
        )
        result = subprocess.run(
            [service.command[0], "-c", program, json.dumps([modules, version])],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            env=env,
            check=False,
        )
        return result.returncode == 0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


class Stack:
    def __init__(self, data, env, services, timeout=15):
        self.data, self.env, self.services, self.timeout = data, env, services, timeout
        self.state_path = data / "stack.json"
        self.state = load_state(self.state_path)

    def save(self):
        atomic_json(self.state_path, self.state)

    def launch(self, service):
        previous = self.state["processes"].get(service.name)
        if previous and owned(previous):
            if service.port and not healthy(service):
                raise StackError(
                    f"{service.name}: owned process is running but unhealthy; inspect its log"
                )
            return "owned", previous["pid"]
        if previous:
            del self.state["processes"][service.name]
            self.save()
        if service.port and healthy(service):
            return "reused", None
        other = existing_worker(service)
        if other:
            return "reused", other
        if service.port and occupied(service.port):
            raise StackError(
                f"{service.name}: port {service.port} is occupied by an unhealthy or different service"
            )
        if not runtime_check(service, self.env):
            raise StackError(
                f"{service.name}: install its component dependencies first (see dev/README.md)"
            )
        for name in service.prerequisites:
            dependency = next(item for item in self.services if item.name == name)
            if not healthy(dependency):
                raise StackError(
                    f"{service.name}: local dependency {name} is unavailable"
                )
        if (
            service.name == "nemotron-worker"
            and not Path(self.env["NEMOTRON_VERIFIER_CONFIG"]).is_file()
        ):
            return "waiting for verifier/config.json", None
        ensure_private_dir(self.data / "logs")
        log_path = self.data / "logs" / (service.name + ".log")
        fd = os.open(
            log_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(fd, 0o600)
        nonce = secrets.token_hex(24)
        with os.fdopen(fd, "ab", buffering=0) as log:
            child = subprocess.Popen(
                (sys.executable, str(Path(__file__).resolve()), "_child", nonce),
                cwd=service.cwd,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        child.stdin.write(json.dumps(list(service.command)).encode() + b"\n")
        child.stdin.close()
        record = {
            "pid": child.pid,
            "identity": identity(child.pid, nonce),
            "nonce": nonce,
            "log": str(log_path),
        }
        if not record["identity"]:
            raise StackError(
                f"{service.name}: process exited immediately; inspect {log_path}"
            )
        self.state["processes"][service.name] = record
        self.save()
        # Keep the child handle while this launcher is alive so it can reap
        # exited wrappers. Detached `up` intentionally leaves live wrappers.
        if not hasattr(self, "children"):
            self.children = {}
        self.children[service.name] = child
        deadline = time.monotonic() + (self.timeout if service.port else 0.5)
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise StackError(f"{service.name}: process exited; inspect {log_path}")
            if service.port and healthy(service):
                return "started", child.pid
            time.sleep(0.1)
        if service.port:
            raise StackError(
                f"{service.name}: health check timed out; inspect {log_path}"
            )
        return "started", child.pid

    def up(self, names):
        results = []
        for service in self.services:
            if service.name in names:
                try:
                    status, pid = self.launch(service)
                    results.append(
                        {
                            "name": service.name,
                            "status": status,
                            "pid": pid,
                            "url": service.url,
                        }
                    )
                except StackError as error:
                    results.append(
                        {"name": service.name, "status": "error", "message": str(error)}
                    )
        return results

    def down(self, names):
        results = []
        for name, record in reversed(list(self.state["processes"].items())):
            if name not in names:
                continue
            try:
                if not owned(record):
                    results.append(
                        {
                            "name": name,
                            "status": "not owned or already exited; left untouched",
                        }
                    )
                    del self.state["processes"][name]
                    continue
                os.killpg(record["pid"], signal.SIGTERM)
                deadline = time.monotonic() + self.timeout
                while time.monotonic() < deadline and owned(record):
                    time.sleep(0.1)
                if owned(record):
                    results.append(
                        {
                            "name": name,
                            "status": "still stopping; cleanup may be in progress",
                        }
                    )
                else:
                    del self.state["processes"][name]
                    results.append({"name": name, "status": "stopped"})
            except ProcessLookupError:
                del self.state["processes"][name]
                results.append({"name": name, "status": "stopped"})
            finally:
                child = getattr(self, "children", {}).get(name)
                if child is not None:
                    child.poll()
        self.save()
        return results

    def status(self, names):
        results = []
        for service in self.services:
            if service.name not in names:
                continue
            record = self.state["processes"].get(service.name)
            managed = bool(record and owned(record))
            alive = (
                healthy(service)
                if service.port
                else managed or bool(existing_worker(service))
            )
            results.append(
                {
                    "name": service.name,
                    "status": "healthy" if alive else "unavailable",
                    "ownership": "managed"
                    if managed
                    else "external"
                    if alive
                    else "none",
                    "url": service.url,
                    "installed": runtime_check(service, self.env),
                }
            )
        return results


def readiness(env):
    checks = {}
    for name, url, key in (
        (
            "resource_capacity",
            env["HARNESS_RESOURCE_MANAGER_URL"] + "/resources",
            env.get("RESOURCE_MANAGER_API_KEY"),
        ),
        (
            "agent_readiness",
            env["AGENT_HARNESS_URL"] + "/settings/readiness",
            env.get("HARNESS_API_KEY"),
        ),
        ("nemotron_setup", env["NEMOTRON_VERIFIER_URL"] + "/health", None),
    ):
        try:
            result = request_json(url, key)
            if name == "resource_capacity":
                pools = result.get("resources")
                if not isinstance(pools, list):
                    raise ValueError
                checks[name] = {
                    "ready": bool(pools),
                    "eligible_pools": len(pools),
                    "message": "Declared capacity; remote VM execution has not been smoke-tested here."
                    if pools
                    else "No eligible registered capacity. Enroll an A5000 node and keep its local heartbeat running.",
                }
            elif name == "nemotron_setup":
                checks[name] = {
                    "ready": result.get("ready") is True,
                    "baseline_ready": result.get("baseline_ready") is True,
                    "message": str(result.get("message", ""))[:500],
                }
            else:
                checks[name] = result
        except (OSError, HTTPError, URLError, ValueError, TypeError):
            checks[name] = {
                "ready": False,
                "message": "Unavailable or authentication failed; check local service configuration.",
            }
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("init", "up", "run", "down", "status", "check")
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--only",
        help="Comma-separated service names; default is all services and workers",
    )
    parser.add_argument(
        "--no-workers",
        action="store_true",
        help="Start/check HTTP services only; do not consume queues",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without writing files or starting processes",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15,
        help="Startup/shutdown wait per process (1-60 seconds)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 60:
        parser.error("--timeout must be between 1 and 60 seconds")
    data = args.data_dir.expanduser().absolute()
    env_path = (args.env_file or data / "local-stack.env").expanduser().absolute()
    if args.command == "init":
        if not args.dry_run:
            initialize(env_path)
        print(
            json.dumps(
                {
                    "env_file": str(env_path),
                    "permissions": "0600",
                    "key": "kept private in env file",
                }
            )
        )
        return 0
    if args.command in {"up", "run"} and not args.dry_run:
        initialize(env_path)
    env, services = environment(ROOT, env_path)
    all_names = {service.name for service in services}
    names = set(args.only.split(",")) if args.only else all_names
    if names - all_names:
        parser.error("Unknown service(s): " + ", ".join(sorted(names - all_names)))
    if args.no_workers:
        names = {
            service.name
            for service in services
            if service.name in names and service.port
        }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "env_file": str(env_path),
                    "links": {name: env[name] for name in LOCAL_LINKS},
                    "services": [
                        {
                            "name": service.name,
                            "cwd": str(service.cwd),
                            "command": shlex.join(service.command),
                            "url": service.url,
                            "installed": bool(installed(service)),
                        }
                        for service in services
                        if service.name in names
                    ],
                },
                indent=2,
            )
        )
        return 0
    stack = Stack(data, env, services, args.timeout)
    if args.command in {"up", "run", "down"}:
        with lock(data):
            results = stack.down(names) if args.command == "down" else stack.up(names)
    else:
        results = stack.status(names)
    output = {"services": results}
    if args.command != "down":
        output["readiness"] = readiness(env)
    if args.command in {"up", "run"}:
        output["notes"] = [
            "Reused processes keep their current environment. Restart them manually to load the shared local verifier key.",
            "Codex, Claude Code, NemoClaw and built-in harness orchestration stay local; their own runtime/profile checks still apply.",
            "API startup does not create evaluations. Started workers may process existing queues; use --no-workers to avoid that.",
        ]
    print(json.dumps(output, indent=2))
    failed = any(item["status"] == "error" for item in results)
    if args.command == "run" and failed:
        with lock(data):
            print(json.dumps({"services": stack.down(names)}, indent=2))
    if args.command == "run" and not failed:
        print(
            "Local services running. Ctrl-C stops only processes owned by this launcher.",
            flush=True,
        )
        stop = False

        def stopping(*_args):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, stopping)
        signal.signal(signal.SIGTERM, stopping)
        try:
            while not stop:
                time.sleep(1)
                exited = [
                    name
                    for name, record in stack.state["processes"].items()
                    if name in names and not owned(record)
                ]
                if exited:
                    print("Owned service exited: " + ", ".join(exited), file=sys.stderr)
                    failed = True
                    break
        finally:
            with lock(data):
                stack.state = load_state(stack.state_path)
                print(json.dumps({"services": stack.down(names)}, indent=2))
    if args.command == "check":
        return (
            1
            if any(
                item["status"] != "healthy" or not item["installed"] for item in results
            )
            else 0
        )
    return 1 if failed else 0


def child_main():
    """Internal session leader; keep its unique process identity until cleanup."""
    command = json.loads(sys.stdin.buffer.readline(65536))
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) for part in command)
    ):
        return 2
    # Parent signals the whole owned session. Stay alive while the service
    # handles that signal and finishes its own cleanup.
    signal.signal(signal.SIGTERM, lambda *_args: None)
    signal.signal(signal.SIGINT, lambda *_args: None)
    return subprocess.call(command, stdin=subprocess.DEVNULL, close_fds=True)


if __name__ == "__main__":
    try:
        raise SystemExit(
            child_main() if len(sys.argv) == 3 and sys.argv[1] == "_child" else main()
        )
    except (StackError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Local stack: {error}", file=sys.stderr)
        raise SystemExit(1)
