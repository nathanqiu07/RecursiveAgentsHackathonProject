"""Local three-node coordinator configuration and authenticated node control."""

import copy
import json
import os
import re
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from . import GPU
from .config import digest, key_headers, loopback_url, suite_identity
from .evaluate import identity


def source_identity(deployment):
    return {key: deployment[key] for key in
            ("runtime", "repository", "commit", "model_revision", "execution", "settings")}


def configured_nodes(config):
    definitions = config.get("nodes")
    if not isinstance(definitions, list) or not 1 <= len(definitions) <= 3:
        raise ValueError("Configure between one and three physical GPU nodes")
    nodes, ids = [], set()
    config_root = Path(config.get("_config_root", "."))
    for definition in definitions:
        name = definition.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,59}", name) or name in ids:
            raise ValueError("Node IDs must be unique")
        ids.add(name)
        path = Path(definition["connection_file"]).expanduser()
        if not path.is_absolute():
            path = config_root / path
        nodes.append({"id": name, "connection_file": str(path.absolute())})
    return nodes


def receipt(node):
    descriptor = os.open(node["connection_file"], os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > 65536:
            raise ValueError("Node receipts must be private regular files")
        document = json.load(stream)
    if document.get("status") != "ready" or document.get("node_id") != node["id"]:
        raise ValueError("Node has no verified enrollment receipt")
    enrolled_image(document)
    loopback_url(document["address"], "")
    key_headers(node)  # Validate the locally stored scoped credential, never expose it.
    return document


def enrolled_image(document):
    """Docker is opt-in, and its receipt must pin an immutable image ID."""
    runtime = document.get("runtime", "libvirt-kvm-vfio")
    if runtime == "libvirt-kvm-vfio":
        return document["target"]["guest_image_sha256"]
    if runtime == "docker-wsl2-nvidia":
        image = document.get("target", {}).get("container_image", "")
        if isinstance(image, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", image):
            return image.removeprefix("sha256:")
    raise ValueError("Unsupported node runtime or unpinned container image")


class NodeClient:
    def __init__(self, node, transport=None):
        self.node = node
        self.receipt = receipt(node)
        self.client = httpx.Client(base_url=self.receipt["address"] + "/v1/",
                                  headers=key_headers(node), timeout=5,
                                  follow_redirects=False, trust_env=False, transport=transport)

    def close(self):
        self.client.close()

    def request(self, method, route, **kwargs):
        with self.client.stream(method, route, **kwargs) as response:
            if response.status_code == 409:
                raise NodeBusy()
            if response.status_code not in {200, 202}:
                raise ValueError("Node request failed")
            body, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > 65536:
                    raise ValueError("Node response too large")
                body.append(chunk)
        return json.loads(b"".join(body))

    def health(self):
        document = self.request("GET", "health")
        verification = document.get("verification") or {}
        gpu = verification.get("gpu") or {}
        runtime = self.receipt.get("runtime", "libvirt-kvm-vfio")
        image = enrolled_image(self.receipt)
        if runtime == "docker-wsl2-nvidia":
            verified = verification.get("boot_verified") is False and all(
                verification.get(field) is True for field in
                ("container_verified", "runner_verified", "gpu_compute_verified", "cleanup_confirmed"))
        else:
            verified = verification.get("boot_verified") is True
        if (document.get("node_id") != self.node["id"] or document.get("status") != "ready"
                or document.get("runtime") != runtime or not verified
                or gpu.get("name") != GPU or gpu.get("uuid") != self.receipt.get("gpu_uuid")
                or not isinstance(gpu.get("uuid"), str) or not gpu["uuid"].startswith("GPU-")
                or not 24000 <= float(gpu.get("memory_total_mb", 0)) <= 25000):
            raise ValueError("Node GPU or runtime verification is invalid")
        if verification.get("image_sha256") != image:
            raise ValueError("Node job image changed after enrollment")
        return document

    def start(self, job_id, alias, health, config):
        target = self.receipt["target"]
        limits = health["limits"]
        return self.request("POST", "jobs", headers={"Idempotency-Key": job_id, "X-Deployment-Alias": alias}, json={
            "resource_id": target["resource_id"], "cpus": limits["cpus"],
            "memory_mb": limits["memory_mb"], "scratch_mb": limits["scratch_mb"],
            "timeout_seconds": min(config.get("vm_safety_timeout_seconds", 21600), limits["max_timeout_seconds"]),
        })

    def cancel(self, job_id):
        return self.request("POST", f"jobs/{job_id}/cancel", json={})

    def job(self, job_id):
        return self.request("GET", f"jobs/{job_id}")


class NodeBusy(Exception):
    pass


def profile(config, node, document):
    result = copy.deepcopy(config)
    base_url = document["address"] + "/v1"
    result.pop("nodes", None)
    result["hardware"] = {"base_url": base_url, "connection_file": node["connection_file"],
                          "gpu_uuid": document["gpu_uuid"]}
    for deployment in result["deployments"].values():
        deployment.pop("api_key_env", None)
        deployment.update(base_url=base_url, connection_file=node["connection_file"])
    return result


def gpu_identity(health):
    gpu = health["verification"]["gpu"]
    return {"name": gpu["name"], "uuid": gpu["uuid"], "memory_mb": float(gpu["memory_total_mb"]),
            "power_limit_w": float(gpu["power_limit_w"]), "driver": str(gpu["driver_version"])}


def inspect_pool(config, client_factory=NodeClient):
    """Missing hosts are visible, never manufactured as available capacity."""
    nodes = configured_nodes(config)
    suite = suite_identity(config)
    def inspect(node):
        try:
            client = client_factory(node)
            try:
                health = client.health()
                if gpu_identity(health) != gpu_identity({"verification": client.receipt["verification"]}):
                    raise ValueError("GPU configuration drift requires re-enrollment and a new pool identity")
                node_config = profile(config, node, client.receipt)
                fingerprint = identity(node_config, gpu_identity(health))
                status = "busy" if health.get("busy") else "available"
                if (client.receipt.get("runtime") == "docker-wsl2-nvidia"
                        and config["reference"] not in client.receipt.get("target", {}).get("deployments", {})):
                    status = "inference_pending"
                return {**node, "status": status,
                        "gpu_uuid": client.receipt["gpu_uuid"], "identity": fingerprint,
                        "image_sha256": health["verification"]["image_sha256"]}
            finally:
                client.close()
        except (OSError, ValueError, KeyError, TypeError, httpx.HTTPError, NodeBusy):
            # Retain the enrolled baseline identity during a temporary outage.
            # Only live available nodes may execute a job.
            try:
                document = receipt(node)
                fingerprint = identity(profile(config, node, document), gpu_identity({"verification": document["verification"]}))
                return {**node, "status": "offline", "identity": fingerprint, "gpu_uuid": document["gpu_uuid"]}
            except (OSError, ValueError, KeyError, TypeError):
                return {**node, "status": "offline", "identity": None, "gpu_uuid": None}
    with ThreadPoolExecutor(max_workers=3) as executor:
        states = list(executor.map(inspect, nodes))
    # Pool identity does not change merely because a node is temporarily busy/offline.
    # Enrollment receipts are immutable inputs; live measurements additionally bind each GPU.
    enrolled = []
    for node in nodes:
        try:
            document = receipt(node)
            enrolled.append({"id": node["id"], "gpu_uuid": document["gpu_uuid"],
                             "image_sha256": enrolled_image(document),
                             "gpu": gpu_identity({"verification": document["verification"]})})
        except (OSError, ValueError, KeyError, TypeError):
            enrolled.append({"id": node["id"], "gpu_uuid": None})
    fingerprint = digest({"nodes": enrolled, "suite": suite,
                          "reference": source_identity(config["deployments"][config["reference"]])})
    return fingerprint, states


def baseline_state(queue, states):
    entries = []
    for node in states:
        fingerprint = node.get("identity")
        job = queue.baseline_job(fingerprint) if fingerprint else None
        entries.append({"node_id": node["id"], "status": job["status"] if job else "unmeasured",
                        "metrics": queue.baseline(fingerprint) if fingerprint else None})
    complete = bool(entries) and all(entry["metrics"] is not None for entry in entries)
    status = "completed" if complete else next((s for s in ("running", "queued", "failed", "rejected", "cancelled")
                                                if any(e["status"] == s for e in entries)), "unmeasured")
    return entries, complete, status


def verify_deployment(document, registered):
    expected = {key: registered[key] for key in ("repository", "commit", "model_revision", "execution", "settings")}
    if document.get("fresh") is not True or document.get("deployment") != expected:
        raise ValueError("VM is not running the exact approved deployment")


def wait_for_engine(client, job_id, deployment, cancelled, stop_event):
    while True:
        if cancelled() or stop_event.is_set():
            raise InterruptedError("Evaluation cancelled")
        job = client.job(job_id)
        if job["status"] in {"failed", "cancelled", "completed"}:
            raise ValueError("GPU VM exited before evaluation")
        if job["status"] == "running":
            try:
                document = client.request("GET", "hardware")
                verify_deployment(document, deployment)
                models = client.request("GET", "models")
                if config_model(deployment) in [row.get("id") for row in models.get("data", [])]:
                    return
            except (NodeBusy, httpx.HTTPError):
                pass
            except ValueError:
                # Hardware deployment mismatches fail immediately; engine startup can return503.
                verify_deployment(client.request("GET", "hardware"), deployment)
        stop_event.wait(1)


def config_model(_deployment):
    from . import MODEL
    return MODEL
