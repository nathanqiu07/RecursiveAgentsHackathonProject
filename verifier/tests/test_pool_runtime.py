"""Explicit container enrollment never impersonates VM or inference readiness."""
import copy
import json
from pathlib import Path

import httpx
import pytest

from nemotron_verifier.pool import NodeClient, baseline_state, inspect_pool
from nemotron_verifier.queue import Queue


@pytest.fixture
def docker_node(tmp_path):
    gpu = {"name": "NVIDIA RTX A5000", "uuid": "GPU-test1234", "memory_total_mb": 24564,
           "power_limit_w": 230, "driver_version": "597.16"}
    verification = {"gpu": gpu, "boot_verified": False, "container_verified": True,
                    "runner_verified": True, "gpu_compute_verified": True,
                    "cleanup_confirmed": True, "image_sha256": "a" * 64}
    document = {"node_id": "desktop", "status": "ready", "runtime": "docker-wsl2-nvidia",
                "address": "http://127.0.0.1:18100", "key": "rn_test_only",
                "gpu_uuid": gpu["uuid"], "verification": verification,
                "target": {"resource_id": "desktop", "container_image": "sha256:" + "a" * 64,
                           "deployments": {}}}
    health = {"node_id": "desktop", "status": "ready", "runtime": "docker-wsl2-nvidia",
              "verification": copy.deepcopy(verification), "busy": False}
    path = tmp_path / "connection.json"
    def write():
        path.write_text(json.dumps(document))
        path.chmod(0o600)
    write()
    node = {"id": "desktop", "connection_file": str(path)}
    def factory(definition=node):
        return NodeClient(definition, httpx.MockTransport(lambda _: httpx.Response(200, json=health)))
    return document, health, node, write, factory


def test_explicit_docker_gpu_is_connected_but_baseline_stays_pending(docker_node, tmp_path, monkeypatch):
    document, health, node, write, factory = docker_node
    client = factory()
    try:
        assert client.health()["verification"]["boot_verified"] is False
    finally:
        client.close()
    config = json.loads((Path(__file__).parents[1] / "pool.example.json").read_text())
    config["nodes"] = [node]
    monkeypatch.setattr("nemotron_verifier.pool.suite_identity", lambda _: "test-suite")
    monkeypatch.setattr("nemotron_verifier.pool.identity", lambda *_: "test-identity")
    fingerprint, states = inspect_pool(config, factory)
    assert states[0]["status"] == "inference_pending"
    assert states[0]["image_sha256"] == "a" * 64
    queue = Queue(str(tmp_path / "jobs.sqlite"))
    try:
        assert baseline_state(queue, states)[1:] == (False, "unmeasured")
    finally:
        queue.db.close()
    # Changing the immutable image changes the pool even when physical GPU is unchanged.
    document["target"]["container_image"] = "sha256:" + "b" * 64
    write()
    assert inspect_pool(config, factory)[0] != fingerprint


@pytest.mark.parametrize("field", ["container_verified", "runner_verified", "gpu_compute_verified", "cleanup_confirmed"])
@pytest.mark.parametrize("value", [False, None, 1])
def test_docker_requires_every_real_verification_flag(docker_node, field, value):
    _, health, _, _, factory = docker_node
    health["verification"][field] = value
    client = factory()
    try:
        with pytest.raises(ValueError, match="verification"):
            client.health()
    finally:
        client.close()


@pytest.mark.parametrize("change", ["implicit_runtime", "direct_runtime", "wrong_gpu", "wrong_uuid",
                                     "small_gpu", "mutable_image", "changed_image", "claims_vm"])
def test_wrong_runtime_gpu_or_image_cannot_enter_benchmark(docker_node, change):
    document, health, _, write, factory = docker_node
    if change == "implicit_runtime":
        document.pop("runtime")
        document["target"]["guest_image_sha256"] = "a" * 64
    elif change == "direct_runtime":
        document["runtime"] = health["runtime"] = "direct-process"
    elif change == "wrong_gpu":
        health["verification"]["gpu"]["name"] = "NVIDIA RTX A4000"
    elif change == "wrong_uuid":
        health["verification"]["gpu"]["uuid"] = "GPU-another"
    elif change == "small_gpu":
        health["verification"]["gpu"]["memory_total_mb"] = 16376
    elif change == "mutable_image":
        document["target"]["container_image"] = "gpu-node:latest"
    elif change == "changed_image":
        health["verification"]["image_sha256"] = "b" * 64
    else:
        health["verification"]["boot_verified"] = True
    write()
    with pytest.raises(ValueError):
        client = factory()
        try:
            client.health()
        finally:
            client.close()


def test_kvm_still_requires_boot_and_exact_guest_image(docker_node):
    document, health, _, write, factory = docker_node
    document["runtime"] = health["runtime"] = "libvirt-kvm-vfio"
    document["target"] = {"guest_image_sha256": "a" * 64}
    write()
    client = factory()
    try:
        with pytest.raises(ValueError, match="verification"):
            client.health()
        health["verification"]["boot_verified"] = True
        assert client.health()["runtime"] == "libvirt-kvm-vfio"
        health["verification"]["image_sha256"] = "b" * 64
        with pytest.raises(ValueError, match="image"):
            client.health()
    finally:
        client.close()
