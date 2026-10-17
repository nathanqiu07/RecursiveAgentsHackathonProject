#!/usr/bin/env python3
"""Install the explicitly authorized Docker alternative on this Linux/WSL host."""
import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import time
from pathlib import Path

from common import Blocked, private_json, read_private
from controller import http, verify_connection
from docker_host import preflight, validate_target


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--ssh-hostname", required=True)
    parser.add_argument("--ssh-port", type=int, default=2222)
    parser.add_argument("--ssh-user", default="gpu-node")
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise Blocked("local_install_requires_root")
    target = validate_target(json.loads(Path(args.target).read_text()))
    preflight({"target": target})
    state = Path(args.state).resolve() / target["id"]
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    install_lock = (state / "install.lock").open("w")
    fcntl.flock(install_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt_path = state / "server-connection.json"
    if receipt_path.exists():
        existing = read_private(receipt_path)
        if existing["target"] != target:
            raise Blocked("existing_configuration_differs_do_not_rotate_key")
        key = existing["key"]
    else:
        key = "rn_" + secrets.token_urlsafe(32)
    source = Path(__file__).resolve().parent
    installed = Path("/opt/crusoe-docker-nodes") / target["id"]
    installed.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("host.py", "common.py", "docker_host.py"):
        shutil.copyfile(source / name, installed / name)
        os.chmod(installed / name, 0o600)
    source_patch = source.parent.parent / "docker-support.patch"
    if source_patch.is_file():
        shutil.copyfile(source_patch, installed / "docker-support.patch")
        os.chmod(installed / "docker-support.patch", 0o600)
    source_commit = subprocess.check_output(["git", "-c", "safe.directory=" + str(source.parent), "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip()
    private_json(installed / "config.json", {"target": target,
                 "state_dir": "/var/lib/crusoe-docker-nodes/" + target["id"],
                 "key_sha256": hashlib.sha256(key.encode()).hexdigest(), "source_commit": source_commit})
    host_key = Path("/etc/ssh/ssh_host_ed25519_key.pub")
    receipt = {"node_id": target["id"], "resource_id": target["resource_id"],
               "status": "provisioning", "runtime": "docker-wsl2-nvidia", "transport": "server-local",
               "address": f"http://127.0.0.1:{target['remote_port']}", "key": key,
               "gpu_uuid": target["gpu_uuid"], "target": target, "verification": None,
               "source_commit": subprocess.check_output(["git", "-c", "safe.directory=" + str(source.parent), "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip(),
               "local_modifications": "Docker backend; apply nodes/docker-support.patch on controller",
               "inference_ready": False,
               "inference_blocker": "Generic CUDA runner only; pinned reference/candidate W4A16 runtimes and model are not installed or verified.",
               "ssh_connection": {"hostname": args.ssh_hostname, "user": args.ssh_user, "port": args.ssh_port,
                   "host_public_key": host_key.read_text().strip(),
                   "reachability": "controller_must_verify_route"}}
    private_json(receipt_path, receipt)
    service = "crusoe-docker-node-" + target["id"]
    unit = f"""[Unit]
Description=Private Docker GPU resource node {target['id']}
After=docker.service
Requires=docker.service
[Service]
Type=simple
User=root
UMask=0077
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/usr/bin/python3 {installed}/docker_host.py --config {installed}/config.json
Restart=on-failure
RestartSec=5
TimeoutStopSec=45
[Install]
WantedBy=multi-user.target
"""
    Path("/etc/systemd/system", service + ".service").write_text(unit)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", service], check=True)
    subprocess.run(["systemctl", "restart", service], check=True)
    receipt["status"] = "ready"
    deadline = time.monotonic() + 120
    while True:
        try:
            receipt["verification"] = verify_connection(receipt)["verification"]
            break
        except Blocked:
            if time.monotonic() >= deadline:
                raise Blocked("docker_node_startup_or_real_gpu_probe_failed")
            time.sleep(2)
    private_json(receipt_path, receipt)
    print(json.dumps({"status": "ready", "runtime": receipt["runtime"], "node_id": target["id"],
                      "receipt_path": str(receipt_path), "verification": receipt["verification"]}, indent=2))


if __name__ == "__main__":
    main()
