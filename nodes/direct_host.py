#!/usr/bin/env python3
"""Explicitly trusted direct GPU VM execution; systemd cgroups, no nested VM."""
import argparse
import fcntl
import json
import os
import pwd
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from http.server import ThreadingHTTPServer

from common import Blocked
from host import Node, make_handler, run

RUNTIME = "direct-host-systemd"


def gpu_probe():
    output = run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,power.limit,driver_version", "--format=csv,noheader,nounits"]).stdout.decode()
    rows = [list(map(str.strip, row.split(","))) for row in output.strip().splitlines()]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise Blocked("exactly_one_visible_gpu_required")
    uid, name, memory, power, driver = rows[0]
    return {"uuid": uid, "name": name, "memory_total_mb": int(float(memory)), "power_limit_w": float(power), "driver_version": driver, "observed_at": time.time(), "source": "direct_host_nvidia_smi"}


class DirectNode(Node):
    def __init__(self, config):
        self.config, self.target = config, config["target"]
        if not re.fullmatch(r"[a-z0-9-]{1,48}", self.target["id"]):
            raise ValueError("invalid_node_id")
        self.runtime = RUNTIME
        self.lock = threading.RLock()
        self.jobs, self.active, self.probe_result = {}, None, None
        self.disk_mb, self.cleanup_blocked = 0, False
        self.state = Path(config["state_dir"])
        self.state.mkdir(mode=0o700, parents=True, exist_ok=True)

    def cleanup(self, unit):
        try:
            run(["systemctl", "stop", unit], timeout=30, check=False)
            state = run(["systemctl", "show", unit, "--property=ActiveState", "--value"], check=False).stdout.decode().strip()
            return state in {"inactive", "failed"}
        except Blocked:
            return False

    def startup(self):
        for receipt in self.state.glob("*.unit"):
            unit = receipt.read_text().strip()
            if not re.fullmatch(r"resource-job-[a-f0-9-]{36}\.service", unit) or not self.cleanup(unit):
                raise Blocked("orphan_direct_job_cleanup_unconfirmed")
            receipt.unlink()
        self.probe()

    def probe(self):
        with self.lock:
            if self.active or self.cleanup_blocked:
                raise Blocked("node_busy_or_cleanup_required")
            gpu = gpu_probe()
            if gpu["uuid"] != self.target["gpu_uuid"]:
                raise Blocked("gpu_identity_mismatch")
            result = run(["runuser", "-u", self.config["job_user"], "--", sys.executable, str(Path(__file__).with_name("guest_runner.py")), "--self-test"])
            if not json.loads(result.stdout).get("ready"):
                raise Blocked("runner_not_ready")
            self.probe_result = {"gpu": gpu, "runtime": RUNTIME, "vm_isolation": None, "boot_verified": False, "runner_verified": True, "direct_execution_verified": True, "verified_at": time.time(), "resource_limits": "systemd_cpu_memory_cgroups", "scratch_limit_enforced": False}
            return self.probe_result

    def submit(self, data, request_id=None, deployment_alias=None):
        if deployment_alias is not None:
            raise ValueError("direct_deployments_not_configured")
        return super().submit(data, request_id, deployment_alias)

    def hardware(self):
        return gpu_probe()

    def work(self, job):
        unit = "resource-job-" + job["id"] + ".service"
        receipt = self.state / (job["id"] + ".unit")
        workspace = Path(self.config["workspace_root"]) / job["id"]
        process = None
        try:
            workspace.mkdir(parents=True, mode=0o700)
            user = pwd.getpwnam(self.config["job_user"])
            os.chown(workspace, user.pw_uid, user.pw_gid)
            receipt.write_text(unit)
            command = ["systemd-run", "--quiet", "--pipe", "--wait", "--collect", "--unit=" + unit,
                "--property=User=" + self.config["job_user"], "--property=KillMode=control-group", "--property=TimeoutStopSec=5",
                "--property=MemoryMax=" + str(job["memory_mb"]) + "M", "--property=CPUQuota=" + str(job["cpus"] * 100) + "%",
                "--property=RuntimeMaxSec=" + str(job["timeout_seconds"]), "--property=NoNewPrivileges=yes",
                "--property=ProtectSystem=strict", "--property=ProtectHome=yes", "--property=PrivateTmp=yes",
                "--property=ReadWritePaths=" + str(workspace), "--working-directory=" + str(workspace),
                sys.executable, str(Path(__file__).with_name("guest_runner.py")), "--workspace", str(workspace)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            job["status"] = "running"
            self.relay(process, job, time.monotonic() + job["timeout_seconds"])
            job["exit_code"] = process.wait(timeout=10)
            job["status"] = "completed" if job["exit_code"] == 0 else "failed"
        except Exception:
            job["status"] = "cancelled" if job["cancel"].is_set() else "failed"
            job["error"] = "direct_job_execution_failed"
        finally:
            cleaned = self.cleanup(unit)
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if not pipe.closed:
                        pipe.close()
            if cleaned:
                receipt.unlink(missing_ok=True)
            with self.lock:
                job["cleanup_confirmed"] = cleaned
                self.cleanup_blocked = not cleaned
                if not cleaned:
                    job["status"], job["error"] = "failed", "direct_cleanup_unconfirmed"
                self.active = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    node = DirectNode(config)
    lock = (node.state / "node.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    node.startup()
    server = ThreadingHTTPServer(("127.0.0.1", node.target["remote_port"]), make_handler(node))
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        if isinstance(node.active, dict):
            job = node.active
            job["cancel"].set()
            node.cleanup("resource-job-" + job["id"] + ".service")
        server.server_close()


if __name__ == "__main__":
    main()
