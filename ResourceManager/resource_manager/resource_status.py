"""Live discovery installed inside the local ResourceManager package.

Both experiment discovery and verifier scheduling use the same node daemon lock.
This module reports reservations without allocating, starting, or stopping a VM.
"""

import concurrent.futures
import json
import os
import stat
import time
import urllib.parse
import urllib.request


def remote_state(resource):
    if getattr(resource, "runtime", "docker") != "remote_vm":
        return None
    try:
        descriptor = os.open(
            resource.node_manifest, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        with os.fdopen(descriptor) as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_mode & 0o077
                or info.st_size > 262144
            ):
                raise ValueError
            receipt = json.load(stream)
        url = urllib.parse.urlsplit(receipt["address"])
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or not url.port
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path
            or receipt.get("node_id") != resource.node_id
            or receipt.get("status") != "ready"
            or not isinstance(receipt.get("key"), str)
            or not receipt["key"].startswith("rn_")
        ):
            raise ValueError
        request = urllib.request.Request(
            receipt["address"] + "/v1/health",
            headers={"Authorization": "Bearer " + receipt["key"]},
        )
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=0.75
        ) as response:
            raw = response.read(1048577)
        if len(raw) > 1048576:
            raise ValueError
        health = json.loads(raw)
        if (
            health.get("node_id") != resource.node_id
            or health.get("status") != "ready"
            or not (
                (health.get("runtime") == "libvirt-kvm-vfio" and health.get("verification", {}).get("boot_verified") is True)
                or (health.get("runtime") == "direct-host-systemd" and receipt.get("runtime") == "direct-host-systemd" and health.get("verification", {}).get("direct_execution_verified") is True and health.get("verification", {}).get("runner_verified") is True)
            )
            or not isinstance(health.get("busy"), bool)
        ):
            raise ValueError
        return {"status": "busy" if health["busy"] else "available"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"status": "unavailable"}


def live_states(catalog):
    resources = [
        resource
        for resource in catalog.resources
        if resource.enabled and resource.node_id is not None
    ]
    if not resources:
        return {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(16, len(resources))
    ) as pool:
        states = list(pool.map(remote_state, resources))
    return {resource.id: state for resource, state in zip(resources, states)}


def filter_available(resources, catalog):
    states = live_states(catalog)
    return [
        resource
        for resource in resources
        if states.get(resource["id"]) in (None, {"status": "available"})
    ]


def resource_status(queue, catalog):
    states = live_states(catalog)
    result = []
    with queue.connection() as database:
        now = time.time()
        for resource in catalog.resources:
            if not resource.enabled or resource.node_id is None:
                continue
            node = database.execute(
                "SELECT * FROM compute_nodes WHERE id=?", (resource.node_id,)
            ).fetchone()
            held, cpus, memory = queue.usage(database, resource.node_id)
            status = (
                "unregistered"
                if node is None
                else queue.node_public(node, now)["status"]
            )
            slots = max(0, resource.max_concurrent_jobs - held)
            available_cpus = (
                max(0.0, min(resource.cpus, node["cpus"]) - float(cpus))
                if node
                else 0.0
            )
            available_memory = (
                max(0, min(resource.memory_mb, node["memory_mb"]) - memory)
                if node
                else 0
            )
            live = states.get(resource.id)
            if status == "online":
                if live and live["status"] == "unavailable":
                    status = "unavailable"
                elif (
                    held
                    or (live and live["status"] == "busy")
                    or slots == 0
                    or available_cpus <= 0
                    or available_memory < 64
                ):
                    status = "busy"
                else:
                    status = "available"
            if status != "available":
                slots = 0
            result.append(
                {
                    "id": resource.id,
                    "node_id": resource.node_id,
                    "description": resource.description,
                    "status": status,
                    "runtime": getattr(resource, "runtime", "docker"),
                    "available_slots": slots,
                    "total_slots": resource.max_concurrent_jobs,
                    "held_jobs": held,
                    "available_cpus": available_cpus,
                    "available_memory_mb": available_memory,
                    "memory_mb": resource.memory_mb,
                    "max_timeout_seconds": resource.max_timeout_seconds,
                }
            )
    return {
        "resources": result,
        "checked_at": time.time(),
        "source": "resource_manager",
    }
