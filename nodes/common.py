"""Validated operator inventory and private receipts (standard library only)."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,59}\Z")
PCI = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\Z")
SSH_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
MAX_BODY = 2 * 1024 * 1024
MAX_OUTPUT = 16 * 1024 * 1024


class Blocked(Exception):
    """Action cannot be verified with the supplied host prerequisites."""


def private_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_private(path):
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise Blocked(f"credential_file_permissions: {path} must be mode 0600")
    return json.loads(path.read_text())


def integer(data, field, low, high, default=None):
    value = data.get(field, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field} must be an integer in [{low}, {high}]")
    return value


def validate_target(data):
    if not isinstance(data, dict):
        raise ValueError("each target must be an object")  # noqa: TRY004
    allowed = {
        "id",
        "ssh_host",
        "local_port",
        "remote_port",
        "guest_image",
        "guest_image_sha256",
        "gpu_pci_devices",
        "gpu_uuid",
        "guest_memory_mb",
        "guest_vcpus",
        "guest_user",
        "guest_job_command",
        "libvirt_network",
        "max_timeout_seconds",
        "boot_timeout_seconds",
        "inference_port",
        "resource_id",
        "scratch_mb",
        "deployments",
    }
    if set(data) - allowed:
        raise ValueError(
            "unknown target fields: " + ", ".join(sorted(set(data) - allowed))
        )
    result = dict(data)
    for field in ("id", "resource_id"):
        value = result.get(field, result.get("id") if field == "resource_id" else "")
        if not isinstance(value, str) or not ID.fullmatch(value):
            raise ValueError(f"invalid {field}")
        result[field] = value
    if not SSH_ALIAS.fullmatch(str(result.get("ssh_host", ""))):
        raise ValueError(
            "ssh_host must be a known, trusted SSH config alias (no user@host or options)"
        )
    for field, default in (
        ("local_port", None),
        ("remote_port", 8090),
        ("inference_port", 8000),
    ):
        result[field] = integer(data, field, 1024, 65535, default)
    result["guest_memory_mb"] = integer(data, "guest_memory_mb", 1024, 1048576)
    result["guest_vcpus"] = integer(data, "guest_vcpus", 1, 256)
    result["scratch_mb"] = integer(data, "scratch_mb", 16, 1048576, 65536)
    result["max_timeout_seconds"] = integer(
        data, "max_timeout_seconds", 60, 86400, 7200
    )
    result["boot_timeout_seconds"] = integer(data, "boot_timeout_seconds", 30, 900, 300)
    for field in ("guest_user", "libvirt_network"):
        result[field] = result.get(
            field, "ubuntu" if field == "guest_user" else "default"
        )
        if not ID.fullmatch(str(result[field])):
            raise ValueError(f"invalid {field}")
    image = result.get("guest_image", "")
    if (
        not isinstance(image, str)
        or not image.startswith("/")
        or any(ord(c) < 32 for c in image)
    ):
        raise ValueError("guest_image must be an absolute operator-approved qcow2 path")
    if not re.fullmatch(r"[a-f0-9]{64}", str(result.get("guest_image_sha256", ""))):
        raise ValueError("guest_image_sha256 must pin the approved guest image bytes")
    devices = result.get("gpu_pci_devices")
    if not isinstance(devices, list) or not devices or len(devices) > 8:
        raise ValueError(
            "gpu_pci_devices must explicitly list the GPU's complete IOMMU group"
        )
    if any(not isinstance(x, str) or not PCI.fullmatch(x) for x in devices) or len(
        set(devices)
    ) != len(devices):
        raise ValueError("invalid/duplicate PCI address")
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]{8,80}", str(result.get("gpu_uuid", ""))):
        raise ValueError("gpu_uuid must identify the approved physical GPU")
    command = result.get("guest_job_command")
    if (
        not isinstance(command, list)
        or not command
        or not str(command[0]).startswith("/")
        or len(command) > 64
        or any(not isinstance(x, str) or "\0" in x or len(x) > 4096 for x in command)
    ):
        raise ValueError(
            "guest_job_command must be an approved absolute executable argv"
        )
    result["deployments"] = validate_deployments(result.get("deployments", {}))
    return result


def validate_deployments(deployments):
    if not isinstance(deployments, dict) or len(deployments) > 16:
        raise ValueError("deployments must be an operator-approved alias map")
    for alias, deployment in deployments.items():
        if (
            not isinstance(alias, str)
            or not ID.fullmatch(alias)
            or not isinstance(deployment, dict)
            or set(deployment) != {"command", "metadata"}
        ):
            raise ValueError("invalid approved deployment")
        argv = deployment["command"]
        if (
            not isinstance(argv, list)
            or not argv
            or not isinstance(argv[0], str)
            or not argv[0].startswith("/")
            or len(argv) > 64
            or any(not isinstance(x, str) or "\0" in x or len(x) > 4096 for x in argv)
        ):
            raise ValueError("deployment command must be an approved absolute argv")
        metadata = deployment["metadata"]
        if (
            not isinstance(metadata, dict)
            or set(metadata)
            != {"repository", "commit", "model_revision", "execution", "settings"}
            or not all(
                isinstance(metadata[x], str)
                for x in ("repository", "commit", "model_revision", "execution")
            )
            or not isinstance(metadata["settings"], dict)
            or len(json.dumps(metadata)) > 32768
        ):
            raise ValueError(
                "deployment metadata requires repository, commit, model_revision, execution and settings"
            )
        if (
            not re.fullmatch(
                r"https://github\.com/[A-Za-z0-9-]+/[A-Za-z0-9_.-]+/?",
                metadata["repository"],
            )
            or not re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", metadata["commit"])
            or not re.fullmatch(
                r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", metadata["model_revision"]
            )
        ):
            raise ValueError("deployment source and model revisions must be immutable")
    return deployments


def load_inventory(path):
    data = json.loads(Path(path).read_text())
    if (
        not isinstance(data, dict)
        or set(data) != {"nodes"}
        or not isinstance(data["nodes"], list)
    ):
        raise ValueError("inventory must contain exactly {nodes: [...]}")
    targets = [validate_target(x) for x in data["nodes"]]
    if not targets:
        raise Blocked(
            "no_ssh_hosts: add existing A5000 server SSH aliases and approved guest image details"
        )
    for field in ("id", "resource_id", "local_port", "ssh_host", "gpu_uuid"):
        if len({x[field] for x in targets}) != len(targets):
            raise ValueError(
                f"duplicate {field}; this version allows one node/GPU group per physical host"
            )
    return targets
