#!/usr/bin/env python3
"""Explicit Docker GPU node alternative. Containers are never reported as KVM VMs."""
import argparse
import json
import hashlib
import re
import signal
import threading
import time
import os
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from common import ID, Blocked, integer, private_json, validate_deployments
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
    if data["guest_job_command"] != ["/usr/local/bin/gpu-job-runner"]:
        raise ValueError("invalid_generic_runner")
    deployments=data["deployments"]
    if not isinstance(deployments,dict) or len(deployments)>16:
        raise ValueError("invalid_deployments")
    for alias,deployment in deployments.items():
        if not ID.fullmatch(alias) or set(deployment)!={"command","metadata","mounts","environment","network"}:
            raise ValueError("invalid_inference_deployment")
        validate_deployments({alias:{k:deployment[k] for k in ("command","metadata")}})
        if not ID.fullmatch(deployment["network"]):
            raise ValueError("invalid_internal_network")
        allowed_env={"PYTHONPATH","PYTHONDONTWRITEBYTECODE","HOME","HF_HOME","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE",
                     "VLLM_LOGGING_LEVEL","OMP_NUM_THREADS","TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR",
                     "CUDA_CACHE_PATH","VLLM_CACHE_ROOT","LD_LIBRARY_PATH","TOKENIZERS_PARALLELISM"}
        allowed_env.add('PYTORCH_ALLOC_CONF')
        if not isinstance(deployment["environment"],dict) or set(deployment["environment"])-allowed_env:
            raise ValueError("unapproved_inference_environment")
        if any(not isinstance(value,str) or '\0' in value or len(value)>4096 for value in deployment["environment"].values()):
            raise ValueError("invalid_inference_environment")
        if not isinstance(deployment["mounts"],list) or not 1<=len(deployment["mounts"])<=4:
            raise ValueError("invalid_approved_mounts")
        destinations=set()
        for mount in deployment["mounts"]:
            if set(mount)!={"source","target","manifest","manifest_sha256"}:
                raise ValueError("invalid_approved_mount")
            for field in ("source","target","manifest"):
                path=mount[field]
                if not isinstance(path,str) or not path.startswith('/') or any(c in path for c in (',','\0','\n','\r')) or '..' in Path(path).parts:
                    raise ValueError("unsafe_mount_path")
            if not mount["target"].startswith(('/opt/','/models/')) or mount["target"] in destinations:
                raise ValueError("unsafe_mount_destination")
            destinations.add(mount["target"])
            if not re.fullmatch('[a-f0-9]{64}',mount["manifest_sha256"]):
                raise ValueError("mount_manifest_must_be_pinned")
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


def verify_mount(mount):
    payload=Path(mount["manifest"]).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=mount["manifest_sha256"]:
        raise Blocked("approved_mount_manifest_changed")
    manifest=json.loads(payload)
    root=Path(mount["source"]).resolve(strict=True)
    actual={str(path.relative_to(root)) for path in root.rglob('*') if path.is_file() or path.is_symlink()}
    if actual != set(manifest['files']) | set(manifest.get('symlinks',{})):
        raise Blocked("approved_mount_file_set_changed")
    for relative,expected in manifest["files"].items():
        path=root/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.resolve(strict=True).is_relative_to(root):
            raise Blocked("approved_mount_path_escape")
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(8*1024**2): digest.update(chunk)
            if hasattr(os,'posix_fadvise'):
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        if digest.hexdigest()!=expected:
            raise Blocked("approved_mount_content_changed")
    for relative,expected in manifest.get("symlinks",{}).items():
        path=root/relative
        if os.readlink(path)!=expected or not path.resolve(strict=True).is_relative_to(root):
            raise Blocked("approved_mount_symlink_changed")


class DockerGuest:
    def __init__(self, config, job):
        self.config, self.job, self.target = config, job, config["target"]
        self.name = "crusoe-" + self.target["id"] + "-" + job["id"]
        self.directory = Path(config["state_dir"]) / "jobs" / job["id"]
        self.ip = None
        self.create_uncertain = job.get("creation_pending", False)
        self.cache_stop=threading.Event()

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
        scratch = self.job.get("scratch_mb", self.target["scratch_mb"])
        alias=self.job.get("deployment_alias")
        deployment=self.target["deployments"].get(alias) if alias else None
        network="none"
        extras=[]
        if deployment:
            network=deployment["network"]
            details=json.loads(run(["docker","network","inspect",network]).stdout)[0]
            if details.get("Internal") is not True:
                raise Blocked("inference_network_must_be_internal")
            for mount in deployment["mounts"]:
                verify_mount(mount)
                extras += ["--mount",f'type=bind,src={mount["source"]},dst={mount["target"]},readonly']
            for name,value in deployment["environment"].items():
                extras += ["--env",name+'='+value]
        self.create_uncertain = True
        self.persist()
        run(["docker", "create", "--name", self.name, "--label", "crusoe.node=" + self.target["id"],
             "--label", "crusoe.job=" + self.job["id"], "--network", network, "--read-only",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--user", "1000:1000",
             "--pids-limit", "512" if deployment else "128", "--shm-size", "512m", "--cpus", str(self.job["cpus"]),
             "--memory", str(self.job["memory_mb"]) + "m", "--memory-swap", str(self.job["memory_mb"]) + "m",
             "--tmpfs", f"/tmp:rw,nosuid,nodev,{'exec' if deployment else 'noexec'},size={scratch}m,mode=1777", "--gpus", "all",
             "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "--init",
             *extras,self.target["container_image"], "/bin/sleep", "infinity"], timeout=60)
        self.create_uncertain = False
        self.persist()
        run(["docker", "start", self.name], timeout=60)
        if deployment:
            details=json.loads(run(["docker","inspect",self.name]).stdout)[0]
            self.ip=details["NetworkSettings"]["Networks"][network]["IPAddress"]
            if deployment['metadata']['settings'].get('checkpoint_page_cache')=='evict_during_load':
                threading.Thread(target=self.limit_checkpoint_cache,args=(deployment,),daemon=True).start()
        value = json.loads(self.execute([*self.target["guest_job_command"], "--self-test"]).stdout)
        if value != {"ready": True, "protocol": "execution_result-jsonl-v1"}:
            raise Blocked("container_runner_self_test_failed")
        self.compute = json.loads(self.execute(["python3", "/usr/local/bin/gpu-compute-smoke"]).stdout)
        if self.compute.get("gpu_compute_verified") is not True:
            raise Blocked("container_cuda_kernel_failed")
        return self.hardware()

    def limit_checkpoint_cache(self,deployment):
        # WSL file cache can consume Windows commit needed for WDDM GPU allocations.
        # Advise away clean checkpoint pages only; never delete or modify model files.
        paths=[path for mount in deployment['mounts'] if mount['target'].startswith('/models/')
               for path in Path(mount['source']).glob('*.safetensors')]
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while not self.cache_stop.is_set():
            for path in paths:
                try:
                    with path.open('rb') as stream:
                        os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            try:
                with opener.open(f'http://{self.ip}:{self.target["inference_port"]}/health',timeout=1) as response:
                    if response.status==200:
                        return
            except OSError:
                pass
            if self.cache_stop.wait(2):
                return

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
        self.cache_stop.set()
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
        preparation=Path(__file__).parent / "baseline-preparation.json"
        evidence_path=Path(__file__).parent/'inference-verification.json'
        provenance_path=Path(__file__).parent/'runtime-provenance.json'
        evidence=json.loads(evidence_path.read_text()) if evidence_path.exists() else None
        ready=False
        if evidence and evidence.get('status')=='ready':
            deployment=self.target['deployments'].get(evidence.get('deployment_alias'))
            ready=bool(deployment and evidence.get('metadata')==deployment['metadata']
                       and evidence.get('container_image')==self.target['container_image']
                       and evidence.get('mount_manifests')==[x['manifest_sha256'] for x in deployment['mounts']]
                       and evidence.get('gpu_uuid')==self.target['gpu_uuid'])
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
                "inference_ready": ready, "approved_deployments": list(self.target["deployments"]),
                "baseline_preparation":json.loads(preparation.read_text()) if preparation.exists() else None,
                "inference_verification":evidence,
                "runtime_provenance":json.loads(provenance_path.read_text()) if provenance_path.exists() else None}

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
