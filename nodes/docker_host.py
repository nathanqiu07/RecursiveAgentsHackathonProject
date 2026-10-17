#!/usr/bin/env python3
"""Explicit Docker GPU node alternative. Containers are never reported as KVM VMs."""
import argparse
import json
import hashlib
import re
import signal
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from common import ID, Blocked, integer, private_json
from host import Node, make_handler, run

RUNTIME = "docker-wsl2-nvidia"


def validate_target(data):
    fields = {"id", "resource_id", "ssh_host", "remote_port", "local_port",
              "container_image", "gpu_uuid", "guest_memory_mb", "guest_vcpus",
              "scratch_mb", "max_timeout_seconds", "boot_timeout_seconds",
              "guest_job_command", "deployments", "inference_port"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("invalid_docker_target_fields")
    for field in ("id", "resource_id", "ssh_host"):
        if not isinstance(data[field], str) or not ID.fullmatch(data[field]):
            raise ValueError("invalid_" + field)
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", data["container_image"]):
        raise ValueError("container_image_must_be_immutable_id")
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]{8,80}", data["gpu_uuid"]):
        raise ValueError("invalid_gpu_uuid")
    for field, low, high in (("remote_port", 1024, 65535), ("local_port", 1024, 65535),
                             ("inference_port", 1024, 65535), ("guest_memory_mb", 1024, 1048576),
                             ("guest_vcpus", 1, 256), ("scratch_mb", 16, 4096),
                             ("max_timeout_seconds", 60, 86400), ("boot_timeout_seconds", 30, 900)):
        integer(data, field, low, high)
    if data["guest_job_command"] != ["/usr/local/bin/gpu-job-runner"] or data["deployments"] != {}:
        raise ValueError("this_docker_image_supports_only_the_generic_runner")
    return dict(data)


def preflight(config):
    target = validate_target(config["target"])
    info = json.loads(run(["docker", "info", "--format", "{{json .}}" ]).stdout)
    if info["OSType"] != "linux" or target["guest_vcpus"] > info["NCPU"]:
        raise Blocked("docker_linux_cpu_capacity_unavailable")
    if info["MemTotal"] < (target["guest_memory_mb"] + 2048) * 1024**2:
        raise Blocked("insufficient_docker_host_ram")
    image = json.loads(run(["docker", "image", "inspect", target["container_image"]]).stdout)[0]
    if image["Id"] != target["container_image"] or image["Config"].get("Volumes"):
        raise Blocked("docker_image_identity_or_volume_mismatch")
    return {"virtual_disk_mb": 0, "runtime": RUNTIME, "image": image["Id"]}


class DockerGuest:
    def __init__(self, config, job):
        self.config, self.job, self.target = config, job, config["target"]
        self.name = "crusoe-" + self.target["id"] + "-" + job["id"]
        self.directory = Path(config["state_dir"]) / "jobs" / job["id"]
        self.ip = None  # No inference alias/network is approved for the generic image.
        self.create_uncertain = job.get("creation_pending", False)

    def persist(self):
        private_json(self.directory / "job.json", {
            **{key: self.job[key] for key in ("id", "memory_mb", "cpus", "timeout_seconds")},
            "creation_pending": self.create_uncertain})

    def ssh_args(self):
        # Node's transport interface; no SSH server or credentials enter a job.
        return ["docker", "exec", "-i", self.name, "/bin/sh", "-c"]

    def execute(self, argv, timeout=30):
        return run(["docker", "exec", self.name, *argv], timeout=timeout)

    def boot(self):
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.create_uncertain = True
        self.persist()
        scratch = self.job.get("scratch_mb", self.target["scratch_mb"])
        run(["docker", "create", "--name", self.name, "--label", "crusoe.node=" + self.target["id"],
             "--label", "crusoe.job=" + self.job["id"], "--network", "none", "--read-only",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--user", "1000:1000",
             "--pids-limit", "128", "--cpus", str(self.job["cpus"]),
             "--memory", str(self.job["memory_mb"]) + "m", "--memory-swap", str(self.job["memory_mb"]) + "m",
             "--tmpfs", f"/tmp:rw,nosuid,nodev,size={scratch}m,mode=1777", "--gpus", "all",
             "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "--init",
             self.target["container_image"], "/bin/sleep", "infinity"], timeout=60)
        self.create_uncertain = False
        self.persist()
        run(["docker", "start", self.name], timeout=60)
        value = json.loads(self.execute([*self.target["guest_job_command"], "--self-test"]).stdout)
        if value != {"ready": True, "protocol": "execution_result-jsonl-v1"}:
            raise Blocked("container_runner_self_test_failed")
        self.compute = json.loads(self.execute(["python3", "/usr/local/bin/gpu-compute-smoke"]).stdout)
        if self.compute.get("gpu_compute_verified") is not True:
            raise Blocked("container_cuda_kernel_failed")
        return self.hardware()

    def hardware(self):
        output = self.execute(["nvidia-smi", "--query-gpu=uuid,name,memory.total,power.limit,driver_version",
                               "--format=csv,noheader,nounits"]).stdout.decode()
        rows = [[v.strip() for v in row.split(",")] for row in output.strip().splitlines()]
        if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0] != self.target["gpu_uuid"]:
            raise Blocked("container_gpu_identity_mismatch")
        gpu_uuid, name, memory, power, driver = rows[0]
        return {"uuid": gpu_uuid, "name": name, "memory_total_mb": int(float(memory)),
                "power_limit_w": None if power == "[N/A]" else float(power), "driver_version": driver,
                "observed_at": time.time(), "source": "active_docker_container_nvidia_smi",
                "container_id": self.job["id"]}

    def cleanup(self):
        try:
            containers = json.loads(run(["docker", "container", "ls", "--all", "--format", "json",
                                        "--filter", "name=^/" + self.name + "$" ]).stdout.decode().strip() or "null")
            if containers:
                details = json.loads(run(["docker", "inspect", self.name]).stdout)[0]
                labels = details["Config"].get("Labels") or {}
                if labels.get("crusoe.node") != self.target["id"] or labels.get("crusoe.job") != self.job["id"]:
                    return False
                run(["docker", "rm", "--force", self.name], timeout=30)
            remaining = run(["docker", "container", "ls", "--all", "--quiet",
                             "--filter", "name=^/" + self.name + "$" ]).stdout.strip()
            if remaining or self.create_uncertain:
                return False
            if self.directory.exists():
                (self.directory / "job.json").unlink(missing_ok=True)
                self.directory.rmdir()
            return True
        except (Blocked, OSError, ValueError, KeyError):
            return False


class DockerNode(Node):
    target_validator = staticmethod(validate_target)
    preflight = staticmethod(preflight)
    guest_class = DockerGuest
    runtime = RUNTIME

    def compatibility(self):
        patch = (Path(__file__).parent / "docker-support.patch").read_bytes()
        if len(patch) > 1024 * 1024:
            raise Blocked("source_patch_limit")
        return {"runtime": self.runtime, "source_commit": self.config.get("source_commit"),
                "patch_sha256": hashlib.sha256(patch).hexdigest(), "patch": patch.decode(),
                "target": self.target,
                "receipt_schema": {"node_id": self.target["id"], "resource_id": self.target["resource_id"],
                    "status": "ready", "runtime": self.runtime, "transport": "ssh-relay",
                    "address": "http://127.0.0.1:18100", "gpu_uuid": self.target["gpu_uuid"],
                    "target": self.target, "verification": self.probe_result,
                    "key": "USE_EXISTING_PRIVATE_SCOPED_KEY"},
                "inference_ready": False, "approved_deployments": list(self.target["deployments"])}

    def verification(self, gpu):
        return {"gpu": gpu, "vm_isolation": False, "boot_verified": False,
                "container_verified": True, "runner_verified": True, "gpu_compute_verified": True,
                "runtime": self.runtime, "verified_at": time.time(),
                "image_sha256": self.target["container_image"].removeprefix("sha256:")}


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    state = Path(config["state_dir"])
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state / "node.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        node = DockerNode(config)
        node.startup()
        node.probe()
        server = ThreadingHTTPServer(("127.0.0.1", node.target["remote_port"]), make_handler(node))
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        try:
            server.serve_forever()
        finally:
            if isinstance(node.active, dict):
                node.active["cancel"].set()
                vm = node.active.get("vm")
                if vm:
                    vm.cleanup()


if __name__ == "__main__":
    main()
