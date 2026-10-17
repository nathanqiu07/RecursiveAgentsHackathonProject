#!/usr/bin/env python3
"""Bootstrap directly on a GPU server; import its private receipt on a controller."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from common import SSH_ALIAS, Blocked, private_json, read_private, validate_target
from controller import (
    INSTALLER,
    PACKAGES,
    bundle_for,
    http,
    open_tunnel,
    register_resource,
    verify_connection,
)


def inspect_server():
    """Read actual host metadata without rebinding GPUs, installing, or rebooting."""

    def command(argv):
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, timeout=15, check=False
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    groups = []
    root = Path("/sys/kernel/iommu_groups")
    for group in sorted(root.glob("*")):
        devices = []
        for device in sorted((group / "devices").glob("*")):
            driver = device / "driver"
            devices.append(
                {
                    "pci": device.name,
                    "driver": driver.resolve().name if driver.exists() else None,
                }
            )
        if devices:
            groups.append({"group": group.name, "devices": devices})
    host_key = Path("/etc/ssh/ssh_host_ed25519_key.pub")
    return {
        "system": platform.system(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
        "addresses": command(["hostname", "-I"]),
        "kvm_present": Path("/dev/kvm").exists(),
        "iommu_groups": groups,
        "gpus": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,pci.bus_id,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "pci_devices": command(["lspci", "-Dnn"]),
        "ssh_host_public_key": host_key.read_text().strip()
        if host_key.is_file()
        else None,
        "ssh_host_fingerprint": command(["ssh-keygen", "-lf", str(host_key)]),
    }


def target_from_file(path, node_id=None):
    document = json.loads(Path(path).expanduser().read_text())
    definitions = (
        document.get("nodes")
        if isinstance(document, dict) and "nodes" in document
        else [document]
    )
    if not isinstance(definitions, list):
        raise TypeError("target_file_must_contain_a_node_or_nodes_list")
    if any(not isinstance(item, dict) for item in definitions):
        raise ValueError("target_nodes_must_be_objects")
    if node_id:
        definitions = [item for item in definitions if item.get("id") == node_id]
    if len(definitions) != 1:
        raise ValueError("select_exactly_one_local_node_with_--node")
    return validate_target(definitions[0])


def ssh_metadata(hostname, user, port):
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}", hostname)
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", user)
        or not 1 <= port <= 65535
    ):
        raise ValueError("provide_actual_ssh_hostname_user_and_port")
    public_key = Path("/etc/ssh/ssh_host_ed25519_key.pub")
    return {
        "hostname": hostname,
        "user": user,
        "port": port,
        "host_public_key": public_key.read_text().strip()
        if public_key.is_file()
        else None,
        "reachability": "controller_must_verify_ssh_route_and_pinned_host_key",
    }


def _privileged(argv):
    return argv if os.geteuid() == 0 else ["sudo", "-n", *argv]


def install_local(target, state, ssh, *, install_packages=False):
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise Blocked("supported_host_is_linux_x86_64")
    directory = Path(state).expanduser().resolve() / target["id"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    path = directory / "server-connection.json"
    if path.exists():
        receipt = read_private(path)
        if receipt.get("target") != target or not receipt.get("key"):
            raise Blocked(
                "existing_server_receipt_differs_do_not_rotate_live_configuration"
            )
        key = receipt["key"]
    else:
        key = "rn_" + secrets.token_urlsafe(32)
    receipt = {
        "node_id": target["id"],
        "status": "provisioning",
        "address": f"http://127.0.0.1:{target['remote_port']}",
        "key": key,
        "gpu_uuid": target["gpu_uuid"],
        "transport": "server-local",
        "target": target,
        "ssh_connection": ssh,
        "verification": None,
        "error": None,
    }
    private_json(path, receipt)
    if install_packages:
        result = subprocess.run(
            _privileged(["sh", "-c", PACKAGES]),
            capture_output=True,
            timeout=900,
            check=False,
        )
        if result.returncode:
            raise Blocked("host_package_install_failed")
    # Same deterministic installer as SSH enrollment, with private input on stdin.
    result = subprocess.run(
        _privileged([sys.executable, "-c", INSTALLER]),
        input=bundle_for(target, key),
        capture_output=True,
        timeout=1000,
        check=False,
    )
    if result.returncode:
        for line in reversed(result.stdout.decode(errors="replace").splitlines()):
            try:
                report = json.loads(line)
            except ValueError:
                continue
            if report.get("status") == "blocked":
                raise Blocked(report.get("error", "server_preflight_blocked"))
        raise Blocked("local_install_or_sudo_failed_check_iommu_vfio_image_and_service")
    deadline = time.monotonic() + 120
    while True:
        try:
            health = http(receipt["address"], key, "/v1/health")
            break
        except Blocked:
            if time.monotonic() >= deadline:
                raise Blocked("local_node_service_startup_failed") from None
            time.sleep(2)
    if health.get("status") != "ready":
        http(
            receipt["address"],
            key,
            "/v1/probe",
            method="POST",
            body={},
            timeout=target["boot_timeout_seconds"] + 90,
        )
    receipt["status"] = "ready"
    receipt["verification"] = verify_connection(receipt)["verification"]
    private_json(path, receipt)
    return receipt, path


def connect_receipt(
    server_path,
    state,
    ssh_alias,
    local_port=None,
    resources_dir=None,
    resource_manager="http://127.0.0.1:8002",
):
    if not SSH_ALIAS.fullmatch(ssh_alias):
        raise ValueError("use_a_trusted_local_ssh_config_alias")
    endpoint = urlsplit(resource_manager)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname != "127.0.0.1"
        or not endpoint.port
        or endpoint.path
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("resource_manager_must_be_a_local_loopback_service")
    server = read_private(Path(server_path).expanduser())
    if (
        server.get("status") != "ready"
        or server.get("transport") != "server-local"
        or not server.get("key")
    ):
        raise Blocked("server_export_is_not_ready")
    target = validate_target(server["target"])
    target["ssh_host"] = ssh_alias
    if local_port is not None:
        if not 1024 <= local_port <= 65535:
            raise ValueError("local_port_out_of_range")
        target["local_port"] = local_port
    directory = Path(state).expanduser().resolve() / target["id"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    path = directory / "connection.json"
    receipt = {
        **server,
        "target": target,
        "transport": "ssh-tunnel",
        "ssh_host": ssh_alias,
        "address": f"http://127.0.0.1:{target['local_port']}",
    }
    connected = False
    if path.exists():
        existing = read_private(path)
        if existing.get("key") != server["key"] or existing.get("target") != target:
            raise Blocked("existing_controller_receipt_conflicts")
        try:
            verify_connection(existing)
            receipt, connected = existing, True
        except Blocked:
            pass
    if not connected:
        receipt["control_socket"] = open_tunnel(target, directory)
        receipt["verification"] = verify_connection(receipt)["verification"]
    private_json(path, receipt)
    if resources_dir:
        register_resource(
            receipt,
            SimpleNamespace(
                state=str(Path(state).expanduser().resolve()),
                resources_dir=str(Path(resources_dir).resolve()),
                resource_manager=resource_manager,
                resource_manager_key_file=None,
            ),
        )
    return receipt, path


def safe_result(receipt, path):
    return {
        "status": "ready",
        "node_id": receipt["node_id"],
        "address": receipt["address"],
        "transport": receipt["transport"],
        "gpu_uuid": receipt["gpu_uuid"],
        "ssh_connection": receipt.get("ssh_connection"),
        "receipt_path": str(path),
        "key_location": "private receipt file; transfer privately, never commit",
        "verification": receipt["verification"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "inspect", help="Read actual Linux server/GPU/IOMMU metadata only"
    )
    install = commands.add_parser(
        "install", help="Install directly on this GPU server and verify one VM"
    )
    install.add_argument("--target", required=True)
    install.add_argument("--node")
    install.add_argument("--state", default="~/.crusoe-node-server")
    install.add_argument("--ssh-hostname", required=True)
    install.add_argument("--ssh-user", required=True)
    install.add_argument("--ssh-port", type=int, default=22)
    install.add_argument("--install-packages", action="store_true")
    connect = commands.add_parser(
        "connect", help="Import a private server receipt on the controller laptop"
    )
    connect.add_argument("--receipt", required=True)
    connect.add_argument("--ssh-alias", required=True)
    connect.add_argument("--local-port", type=int)
    connect.add_argument("--state", default="~/.crusoe-nodes")
    connect.add_argument("--resources-dir")
    connect.add_argument("--resource-manager", default="http://127.0.0.1:8002")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_server(), indent=2))
            return 0
        if args.command == "install":
            target = target_from_file(args.target, args.node)
            receipt, path = install_local(
                target,
                args.state,
                ssh_metadata(args.ssh_hostname, args.ssh_user, args.ssh_port),
                install_packages=args.install_packages,
            )
        else:
            receipt, path = connect_receipt(
                args.receipt,
                args.state,
                args.ssh_alias,
                args.local_port,
                args.resources_dir,
                args.resource_manager,
            )
        print(json.dumps(safe_result(receipt, path), indent=2))
        return 0
    except (
        Blocked,
        ValueError,
        TypeError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        # Domain errors are sanitized by helpers. OS/subprocess diagnostics may contain paths or host details.
        reason = (
            str(error)
            if isinstance(error, (Blocked, ValueError, TypeError))
            else type(error).__name__
        )
        print(
            json.dumps(
                {"status": "blocked", "address": None, "key": None, "error": reason}
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
