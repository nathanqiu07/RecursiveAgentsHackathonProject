#!/usr/bin/env python3
"""Run as root on an explicitly trusted existing Linux GPU VM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import secrets
import shutil
import subprocess

from direct_host import gpu_probe
from common import private_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--memory-mb", type=int, default=8192)
    parser.add_argument("--cpus", type=int, default=4)
    args = parser.parse_args()
    import re
    if os.geteuid() != 0 or not re.fullmatch(r"[a-z0-9-]{1,48}", args.node_id) or not 1024 <= args.port <= 65535:
        parser.error("requires root, a lowercase node ID, and unprivileged port")
    if args.memory_mb < 1024 or args.cpus < 1 or args.cpus > (os.cpu_count() or 1):
        parser.error("invalid capacity")
    gpu = gpu_probe()
    name = "resourcejob"
    try:
        pwd.getpwnam(name)
    except KeyError:
        subprocess.run(["useradd", "--system", "--no-create-home", "--shell", "/usr/sbin/nologin", name], check=True)
    source = Path(__file__).resolve().parent
    destination = Path("/opt/crusoe-direct-node")
    destination.mkdir(mode=0o755, exist_ok=True)
    for filename in ("direct_host.py", "guest_runner.py", "host.py", "common.py"):
        shutil.copyfile(source / filename, destination / filename)
        (destination / filename).chmod(0o644)
    state = Path("/var/lib/crusoe-direct-node")
    state.mkdir(mode=0o700, exist_ok=True)
    workspaces = Path("/var/lib/crusoe-direct-workspaces")
    workspaces.mkdir(mode=0o711, exist_ok=True)
    config_file = state / "config.json"
    if config_file.exists():
        raise SystemExit("Existing configuration preserved; restart the existing service or explicitly migrate it.")
    key = "rn_" + secrets.token_urlsafe(32)
    target = {"id": args.node_id, "resource_id": "gpu-" + args.node_id, "gpu_uuid": gpu["uuid"], "remote_port": args.port,
              "guest_vcpus": args.cpus, "guest_memory_mb": args.memory_mb, "scratch_mb": 16384, "max_timeout_seconds": 21600}
    config = {"target": target, "state_dir": str(state / "jobs"), "workspace_root": str(workspaces), "job_user": name, "key_sha256": hashlib.sha256(key.encode()).hexdigest()}
    private_json(config_file, config)
    private_json(state / "server-connection.json", {"status": "installed_unverified", "node_id": args.node_id, "resource_id": target["resource_id"], "address": "http://127.0.0.1:" + str(args.port), "runtime": "direct-host-systemd", "key": key, "gpu": gpu, "limits": {"cpus": args.cpus, "memory_mb": args.memory_mb, "scratch_mb": 16384, "max_timeout_seconds": 21600}})
    unit = """[Unit]
Description=Direct GPU resource node
After=network.target
[Service]
ExecStart=/usr/bin/python3 /opt/crusoe-direct-node/direct_host.py --config /var/lib/crusoe-direct-node/config.json
Restart=on-failure
RestartSec=3
TimeoutStopSec=40
UMask=0077
[Install]
WantedBy=multi-user.target
"""
    Path("/etc/systemd/system/crusoe-direct-node.service").write_text(unit)
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "--now", "crusoe-direct-node.service"], check=True)
    print(json.dumps({"installed": True, "receipt": str(state / "server-connection.json"), "runtime": "direct-host-systemd", "note": "Verify authenticated health and a real job over SSH before registering ready."}))


if __name__ == "__main__":
    main()
