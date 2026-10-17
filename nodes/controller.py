#!/usr/bin/env python3
"""Enroll existing A5000 hosts through SSH, optionally with parallel local Codex agents."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import secrets
import shlex
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import Blocked, load_inventory, private_json, read_private

HERE = Path(__file__).resolve().parent
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "ConnectTimeout=10",
]

# Target is data in the private tar stream. No key is exposed in process arguments.
INSTALLER = r"""
import io,json,os,pathlib,subprocess,sys,tarfile
if os.geteuid()!=0: raise SystemExit('root_required')
bundle=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read(4*1024*1024)),mode='r:')
allowed={'host.py','common.py','config.json'}
files={m.name:bundle.extractfile(m).read() for m in bundle.getmembers() if m.isfile() and m.name in allowed}
if set(files)!=allowed: raise SystemExit('invalid_bundle')
config=json.loads(files['config.json']); node_id=config['target']['id']
import re
if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,59}',node_id) is None: raise SystemExit('invalid_node_id')
root=pathlib.Path('/opt/crusoe-resource-nodes')/node_id
root.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(root,0o700)
unit='crusoe-node-'+node_id+'.service'
unit_file=pathlib.Path('/etc/systemd/system')/unit
# Refuse replacement while a VM/job is running. Existing endpoints support explicit rotation after stopping jobs.
if unit_file.exists():
 if not (root/'config.json').exists() or json.loads((root/'config.json').read_bytes())!=config:
  raise SystemExit('node_already_installed_with_different_configuration')
 subprocess.run(['systemctl','start',unit],check=True,capture_output=True)
 print(json.dumps({'status':'installed','node_id':node_id})); raise SystemExit(0)
for name,data in files.items():
 p=root/name; p.write_bytes(data); os.chmod(p,0o600)
check=subprocess.run([sys.executable,str(root/'host.py'),'check','--config',str(root/'config.json')],capture_output=True,text=True,timeout=900)
if check.returncode:
 print(check.stdout); raise SystemExit(2)
unit_file.write_text('[Unit]\nDescription=Private GPU resource node '+node_id+'\nAfter=libvirtd.service network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\nUser=root\nUMask=0077\nEnvironment=LC_ALL=C\nExecStart=/usr/bin/python3 '+str(root/'host.py')+' serve --config '+str(root/'config.json')+'\nRestart=on-failure\nRestartSec=10\nTimeoutStopSec=30\nKillMode=mixed\n\n[Install]\nWantedBy=multi-user.target\n')
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','enable','--now',unit],check=True,capture_output=True)
print(json.dumps({'status':'installed','node_id':node_id}))
"""

CHECKER = r"""
import io,json,os,pathlib,subprocess,sys,tarfile,tempfile
bundle=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read(4*1024*1024)),mode='r:')
with tempfile.TemporaryDirectory(prefix='crusoe-node-check-') as directory:
 root=pathlib.Path(directory)
 for name in ('host.py','common.py','config.json'):
  item=bundle.getmember(name)
  if not item.isfile(): raise SystemExit('invalid_bundle')
  (root/name).write_bytes(bundle.extractfile(item).read())
 result=subprocess.run([sys.executable,str(root/'host.py'),'check','--config',str(root/'config.json')],timeout=900)
 raise SystemExit(result.returncode)
"""

PACKAGES = "export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq qemu-kvm libvirt-daemon-system libvirt-clients cloud-image-utils openssh-client python3"


def http(
    address,
    key,
    path,
    *,
    method="GET",
    body=None,
    timeout=15,
    idempotency_key=None,
    deployment_alias=None,
):
    request = urllib.request.Request(
        address + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    if idempotency_key:
        request.add_header("Idempotency-Key", idempotency_key)
    if deployment_alias is not None:
        request.add_header("X-Deployment-Alias", deployment_alias)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(20 * 1024 * 1024 + 1)
        if len(payload) > 20 * 1024 * 1024:
            raise Blocked("node_reply_limit")
        return json.loads(payload)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read(8192)).get("error", "http_" + str(exc.code))
        except (ValueError, AttributeError):
            detail = "http_" + str(exc.code)
        raise Blocked(str(detail)) from None
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise Blocked("node_unreachable_or_invalid_response") from None


def bundle_for(target, key):
    config = {
        "target": target,
        "state_dir": "/var/lib/crusoe-resource-nodes/" + target["id"],
        "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
    }
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as bundle:
        for name, payload in (
            ("host.py", (HERE / "host.py").read_bytes()),
            ("common.py", (HERE / "common.py").read_bytes()),
            ("config.json", json.dumps(config).encode()),
        ):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(payload), 0o600
            bundle.addfile(info, io.BytesIO(payload))
    return out.getvalue()


def remote_bundle(target, bundle, *, check_only=False):
    script = CHECKER if check_only else INSTALLER
    command = "sudo -n python3 -c " + shlex.quote(script)
    try:
        result = subprocess.run(
            ["ssh", *SSH_OPTIONS, target["ssh_host"], command],
            input=bundle,
            capture_output=True,
            timeout=1000,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Blocked("ssh_bootstrap_unavailable_or_timed_out") from exc
    if result.returncode:
        # Include only JSON safe error from our deterministic preflight, never SSH diagnostics.
        for line in reversed(result.stdout.decode(errors="replace").splitlines()):
            try:
                report = json.loads(line)
                if report.get("status") == "blocked":
                    raise Blocked(report["error"])
            except ValueError:
                continue
        known_errors = {
            "root_required",
            "invalid_bundle",
            "invalid_node_id",
            "node_already_installed_with_different_configuration",
        }
        for line in result.stderr.decode(errors="replace").splitlines():
            if line.strip() in known_errors:
                raise Blocked(line.strip())
        raise Blocked(
            "ssh_or_sudo_or_bootstrap_failed: verify trusted SSH alias, passwordless sudo and host prerequisites"
        )
    try:
        return json.loads(result.stdout.decode().strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise Blocked("bootstrap_response_invalid") from exc


def open_tunnel(target, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    control = str(directory / "ssh.sock")
    # macOS OpenSSH UNIX socket paths have a short limit; error before creating a broken tunnel.
    if len(control.encode()) > 100:
        raise Blocked("state_path_too_long: choose a shorter --state directory")
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", target["local_port"]))
    except OSError as exc:
        raise Blocked("local_forward_port_in_use") from exc
    command = [
        "ssh",
        *SSH_OPTIONS,
        "-M",
        "-S",
        control,
        "-o",
        "ControlPersist=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-fNT",
        "-L",
        f"127.0.0.1:{target['local_port']}:127.0.0.1:{target['remote_port']}",
        target["ssh_host"],
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Blocked("ssh_tunnel_failed") from exc
    if result.returncode:
        raise Blocked("ssh_tunnel_failed")
    return control


def close_tunnel(target, control):
    subprocess.run(
        ["ssh", *SSH_OPTIONS, "-S", control, "-O", "exit", target["ssh_host"]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )


def verify_connection(receipt, *, require_boot=True):
    if receipt.get("status") != "ready" or not receipt.get("key"):
        raise Blocked("connection_receipt_not_ready")
    url = urllib.parse.urlsplit(receipt.get("address", ""))
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or not url.port
        or url.path
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise Blocked("connection_address_must_be_private_local_tunnel")
    result = http(receipt["address"], receipt["key"], "/v1/health")
    expected_runtime = receipt.get("runtime", "libvirt-kvm-vfio")
    if expected_runtime not in {"libvirt-kvm-vfio", "docker-wsl2-nvidia"}:
        raise Blocked("unsupported_receipt_runtime")
    if (
        result.get("node_id") != receipt["node_id"]
        or result.get("runtime") != expected_runtime
    ):
        raise Blocked("node_identity_or_vm_runtime_mismatch")
    if result.get("status") != "ready":
        raise Blocked("node_readiness_not_verified")
    verification = result.get("verification") or {}
    if expected_runtime == "docker-wsl2-nvidia" and (
        "sha256:" + str(verification.get("image_sha256", ""))
        != receipt.get("target", {}).get("container_image")
    ):
        raise Blocked("container_image_receipt_mismatch")
    verified = verification.get("boot_verified") if expected_runtime == "libvirt-kvm-vfio" else (
        verification.get("container_verified") is True
        and verification.get("gpu_compute_verified") is True
        and verification.get("runner_verified") is True
        and verification.get("cleanup_confirmed") is True
        and verification.get("boot_verified") is False
    )
    if require_boot and (not verified or verification.get("gpu", {}).get("uuid") != receipt["gpu_uuid"]):
        raise Blocked("gpu_boot_readiness_not_verified")
    # The endpoint must actually enforce the scoped key, not accept anonymous/incorrect credentials.
    try:
        http(receipt["address"], secrets.token_urlsafe(32), "/v1/health")
    except Blocked as exc:
        if str(exc) != "invalid_node_key":
            raise Blocked("node_key_enforcement_unconfirmed") from exc
    else:
        raise Blocked("node_does_not_enforce_key")
    return result


def enroll_one(target, state, *, check_only=False, install_packages=False):
    directory = Path(state).resolve() / target["id"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    receipt_path = directory / "connection.json"
    if receipt_path.exists() and not check_only:
        receipt = read_private(receipt_path)
        if not receipt.get("key") or receipt.get("target") != target:
            raise Blocked("existing_receipt_mismatch_requires_operator_review")
        control = receipt.get("control_socket")
        try:
            health = http(receipt["address"], receipt["key"], "/v1/health")
        except Blocked:
            remote_bundle(target, bundle_for(target, receipt["key"]))
            control = open_tunnel(target, directory)
            time.sleep(2)
            health = http(receipt["address"], receipt["key"], "/v1/health")
        if health.get("status") != "ready":
            http(
                receipt["address"],
                receipt["key"],
                "/v1/probe",
                method="POST",
                body={},
                timeout=target["boot_timeout_seconds"] + 90,
            )
        receipt.update(status="ready", error=None, control_socket=control)
        receipt["verification"] = verify_connection(receipt)["verification"]
        private_json(receipt_path, receipt)
        return receipt
    key = "rn_" + secrets.token_urlsafe(32)
    if install_packages and not check_only:
        result = subprocess.run(
            [
                "ssh",
                *SSH_OPTIONS,
                target["ssh_host"],
                "sudo -n sh -c " + shlex.quote(PACKAGES),
            ],
            capture_output=True,
            timeout=900,
            check=False,
        )
        if result.returncode:
            raise Blocked("host_package_install_failed")
    if check_only:
        check = remote_bundle(target, bundle_for(target, key), check_only=True)
        return {
            "node_id": target["id"],
            "status": "prerequisites_ready",
            "address": None,
            "key": None,
            "checks": check.get("checks"),
            "error": None,
        }
    # Save recoverable credentials before a long boot probe. This is not a ready receipt.
    address = f"http://127.0.0.1:{target['local_port']}"
    receipt = {
        "node_id": target["id"],
        "status": "provisioning",
        "address": address,
        "key": key,
        "gpu_uuid": target["gpu_uuid"],
        "transport": "ssh-tunnel",
        "ssh_host": target["ssh_host"],
        "target": target,
        "verification": None,
        "control_socket": None,
        "error": None,
    }
    private_json(receipt_path, receipt)
    remote_bundle(target, bundle_for(target, key))
    control = open_tunnel(target, directory)
    receipt["control_socket"] = control
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                http(address, key, "/v1/health")
                break
            except Blocked:
                if time.monotonic() >= deadline:
                    raise Blocked("node_service_startup_not_ready")
                time.sleep(2)
        verification = http(
            address,
            key,
            "/v1/probe",
            method="POST",
            body={},
            timeout=target["boot_timeout_seconds"] + 90,
        )
        receipt.update(status="ready", verification=verification)
        verify_connection(receipt)
        private_json(receipt_path, receipt)
        return receipt
    except Exception:
        receipt.update(status="blocked", error="node_boot_readiness_unverified")
        private_json(receipt_path, receipt)
        close_tunnel(target, control)
        raise


def agent_one(target, args):
    directory = Path(args.state).resolve() / target["id"]
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(directory, 0o700)
    inventory = directory / "inventory.json"
    private_json(inventory, {"nodes": [target]})
    agent_result = directory / "agent-result.json"
    command = [
        sys.executable,
        str(HERE / "controller.py"),
        "enroll",
        "--inventory",
        str(inventory),
        "--state",
        str(Path(args.state).resolve()),
        "--quiet",
    ]
    if args.check_only:
        command.append("--check-only")
    if args.install_packages:
        command.append("--install-packages")
    prompt = (
        "Bring this already-owned GPU server online using the approved deterministic enrollment tool. "
        "Run exactly the following command once. Do not edit files, run additional provisioning commands, "
        "change kernel/VFIO settings, or print connection keys. Inspect only its exit code and report. "
        "Return status ready only on exit zero, or prerequisites_ready for check-only. "
        "No invented readiness, endpoints, keys or scores. Command: "
        + shlex.join(command)
        + "\nReceipt location: "
        + str(directory / "connection.json")
    )
    codex = [
        "codex",
        "--ask-for-approval",
        "never",
        "exec",
        "--sandbox",
        "danger-full-access",
        "--ephemeral",
        "--skip-git-repo-check",
        "--cd",
        str(directory),
        "--output-schema",
        str(HERE / "agent-result.schema.json"),
        "--output-last-message",
        str(agent_result),
        "-",
    ]
    log_path = directory / "agent.log"
    with log_path.open("w") as log:
        os.chmod(log_path, 0o600)
        try:
            result = subprocess.run(
                codex,
                input=prompt,
                text=True,
                stdout=log,
                stderr=log,
                timeout=args.agent_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise Blocked("codex_agent_unavailable_or_timed_out") from exc
    if result.returncode:
        raise Blocked("codex_agent_failed: see private local agent.log")
    # Agent prose is never proof. The parent checks the actual daemon and GPU boot receipt.
    if args.check_only:
        report = read_private(directory / "result.json")
        if report.get("status") != "prerequisites_ready":
            raise Blocked("agent_prerequisites_not_verified")
        return report
    receipt = read_private(directory / "connection.json")
    if receipt.get("target") != target or receipt.get("node_id") != target["id"]:
        raise Blocked("agent_connection_inventory_mismatch")
    if receipt.get("address") != f"http://127.0.0.1:{target['local_port']}":
        raise Blocked("agent_connection_tunnel_mismatch")
    verify_connection(receipt)
    return receipt


def register_resource(receipt, args):
    target = receipt["target"]
    health = verify_connection(receipt)
    limits = health["limits"]
    spec = {
        "id": target["resource_id"],
        "node_id": target["id"],
        "description": "Verified RTX A5000 GPU passthrough VM node",
        "enabled": True,
        "runtime": "remote_vm",
        "node_manifest": str(
            Path(args.state).resolve() / target["id"] / "connection.json"
        ),
        "image": "approved-vm:" + target["guest_image_sha256"],
        "cpus": float(limits["cpus"]),
        "memory_mb": limits["memory_mb"],
        "scratch_mb": limits["scratch_mb"],
        "max_timeout_seconds": limits["max_timeout_seconds"],
        "max_concurrent_jobs": 1,
        "allocation_mode": "on_demand",
        "cost_per_hour_usd": None,
    }
    key = (
        Path(args.resource_manager_key_file).read_text().strip()
        if args.resource_manager_key_file
        else os.environ.get("RESOURCE_MANAGER_API_KEY", "")
    )
    if (
        args.resource_manager_key_file
        and Path(args.resource_manager_key_file).stat().st_mode & 0o077
    ):
        raise Blocked("resource_manager_key_file_must_be_0600")
    registration = http(
        args.resource_manager,
        key,
        "/nodes/register",
        method="POST",
        body={
            "id": target["id"],
            "cpus": float(limits["cpus"]),
            "memory_mb": limits["memory_mb"],
            "status": "online",
        },
    )
    path = Path(args.resources_dir) / (target["resource_id"] + ".json")
    if path.exists():
        existing = json.loads(path.read_text())
        if (
            existing.get("node_id") != target["id"]
            or existing.get("runtime") != "remote_vm"
        ):
            raise Blocked("resource_catalog_entry_conflicts_with_existing_resource")
    private_json(path, spec)
    return registration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["enroll", "heartbeat", "start-job", "stop-job"]
    )
    parser.add_argument("--inventory")
    parser.add_argument("--state", default=str(Path.home() / ".crusoe-nodes"))
    parser.add_argument(
        "--agents",
        action="store_true",
        help="one local Codex exec agent per host, in parallel",
    )
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--agent-timeout-seconds", type=int, default=2400)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="read-only host preflight; does not boot a VM",
    )
    parser.add_argument(
        "--install-packages",
        action="store_true",
        help="install supported Ubuntu/Debian host packages",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write private result files, print no credentials",
    )
    parser.add_argument("--resource-manager", default="http://127.0.0.1:8002")
    parser.add_argument("--resource-manager-key-file")
    parser.add_argument(
        "--resources-dir",
        help="write verified node catalog and register it with local ResourceManager",
    )
    parser.add_argument("--node", help="node ID for start-job / stop-job")
    parser.add_argument(
        "--deployment",
        help="operator-approved inference deployment alias for start-job",
    )
    parser.add_argument("--job", help="VM job ID for stop-job")
    args = parser.parse_args(argv)
    from urllib.parse import urlsplit

    manager_url = urlsplit(args.resource_manager)
    if (
        manager_url.scheme != "http"
        or manager_url.hostname != "127.0.0.1"
        or not manager_url.port
        or manager_url.username
        or manager_url.password
        or manager_url.path
        or manager_url.query
        or manager_url.fragment
    ):
        parser.error("ResourceManager must be a local http://127.0.0.1:PORT address")
    state = Path(args.state).resolve()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    try:
        if not 1 <= args.parallel <= 32:
            raise ValueError("parallel must be between 1 and 32")
        if args.command in {"start-job", "stop-job"}:
            if not args.node or "/" in args.node or args.node in {".", ".."}:
                raise ValueError("a valid --node is required")
            receipt = read_private(state / args.node / "connection.json")
            verify_connection(receipt)
            if args.command == "start-job":
                target = receipt["target"]
                result = http(
                    receipt["address"],
                    receipt["key"],
                    "/v1/jobs",
                    method="POST",
                    body={
                        "resource_id": target["resource_id"],
                        "cpus": target["guest_vcpus"],
                        "memory_mb": target["guest_memory_mb"],
                        "scratch_mb": target["scratch_mb"],
                        "timeout_seconds": target["max_timeout_seconds"],
                    },
                    idempotency_key=str(__import__("uuid").uuid4()),
                    deployment_alias=args.deployment,
                )
                result.update(
                    inference_base_url=receipt["address"] + "/v1",
                    key_file=str(state / args.node / "connection.json"),
                )
            else:
                if not args.job:
                    raise ValueError("--job is required")
                import uuid

                job_id = str(uuid.UUID(args.job))
                result = http(
                    receipt["address"],
                    receipt["key"],
                    "/v1/jobs/" + job_id + "/cancel",
                    method="POST",
                    body={},
                )
            print(json.dumps(result))
            return 0
        if args.command == "heartbeat":
            while True:
                for path in state.glob("*/connection.json"):
                    try:
                        receipt = read_private(path)
                        verify_connection(receipt)
                        register_resource(receipt, args) if args.resources_dir else None
                    except (Blocked, ValueError, OSError):
                        # Do not refresh a node whose real endpoint cannot be verified.
                        pass
                time.sleep(30)
        if not args.inventory:
            raise ValueError("--inventory is required")
        targets = load_inventory(args.inventory)
        results = []

        def task(target):
            try:
                report = (
                    agent_one(target, args)
                    if args.agents
                    else enroll_one(
                        target,
                        state,
                        check_only=args.check_only,
                        install_packages=args.install_packages,
                    )
                )
                if args.resources_dir and report["status"] == "ready":
                    register_resource(report, args)
            except Exception as exc:  # noqa: BLE001 - each node gets a truthful isolated failure receipt.
                report = {
                    "node_id": target["id"],
                    "status": "blocked",
                    "address": None,
                    "key": None,
                    "error": str(exc)
                    if isinstance(exc, (Blocked, ValueError))
                    else "enrollment_failed",
                }
            private_json(state / target["id"] / "result.json", report)
            return report

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = list(pool.map(task, targets))
        report = {
            "status": "ready"
            if all(x["status"] == "ready" for x in results)
            else "prerequisites_ready"
            if all(x["status"] == "prerequisites_ready" for x in results)
            else "blocked",
            "nodes": results,
        }
        private_json(state / "results.json", report)
        if not args.quiet:
            print(json.dumps(report))
        return 0 if report["status"] in {"ready", "prerequisites_ready"} else 2
    except (Blocked, ValueError, OSError) as exc:
        report = {"status": "blocked", "nodes": [], "error": str(exc)}
        private_json(state / "results.json", report)
        if not args.quiet:
            print(json.dumps(report))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
