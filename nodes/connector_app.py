#!/usr/bin/env python3
"""Small outbound SSH connector for a GPU node; Python standard library only."""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable

RELAY_HOST = "172.96.160.149"
RELAY_USER = "shardul"
RELAY_PORT = 22
RELAY_FINGERPRINT = "SHA256:cPm6ER/B86pNFzx4O76UgwJsGgzwMrs7i9g0CcHmFHo"
REMOTE_PORT = 18200
LOCAL_PORT = 18090


def verified_host_key(scan: str) -> str:
    """Accept only the pinned ED25519 key; never trust a scan by itself."""
    for line in scan.splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[1] != "ssh-ed25519":
            continue
        try:
            decoded = base64.b64decode(parts[2], validate=True)
        except ValueError:
            continue
        fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(decoded).digest()).decode().rstrip("=")
        if fingerprint == RELAY_FINGERPRINT:
            return f"{RELAY_HOST} ssh-ed25519 {parts[2]}\n"
    raise RuntimeError("Relay identity could not be verified. Connection stopped.")


def ssh_command(ssh: str, key: Path, known_hosts: Path) -> list[str]:
    return [
        ssh, "-F", "NUL" if os.name == "nt" else "/dev/null",
        "-N", "-T", "-p", str(RELAY_PORT), "-i", str(key),
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "GlobalKnownHostsFile=NUL" if os.name == "nt" else "GlobalKnownHostsFile=/dev/null",
        "-o", "HostKeyAlgorithms=ssh-ed25519",
        "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        "-o", "ForwardAgent=no", "-o", "LogLevel=ERROR",
        "-R", f"127.0.0.1:{REMOTE_PORT}:127.0.0.1:{LOCAL_PORT}",
        f"{RELAY_USER}@{RELAY_HOST}",
    ]


class Connector:
    def __init__(self, status: Callable[[str], None]):
        self.status = status
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen | None = None

    def start(self, key: str):
        if self.thread and self.thread.is_alive():
            return
        path = Path(key).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Choose the relay private key file supplied by the controller.")
        if not shutil.which("ssh") or not shutil.which("ssh-keyscan"):
            raise ValueError("OpenSSH Client is required. Enable it in Windows Optional Features.")
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, args=(path,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        process = self.process
        if process and process.poll() is None:
            process.terminate()
        self.status("Disconnected")

    def _run(self, key: Path):
        # Host keys are public, but this private temporary directory prevents substitution.
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        with tempfile.TemporaryDirectory(prefix="gpu-connector-") as folder:
            known_hosts = Path(folder) / "known_hosts"
            delay = 2
            while not self.stop_event.is_set():
                self.status("Verifying relay identity…")
                try:
                    scan = subprocess.run(
                        [shutil.which("ssh-keyscan"), "-T", "10", "-p", str(RELAY_PORT), "-t", "ed25519", RELAY_HOST],
                        capture_output=True, text=True, timeout=15, **options,
                    )
                    if self.stop_event.is_set():
                        break
                    if not scan.stdout.strip():
                        self.status("Relay unreachable. Retrying…")
                        self.stop_event.wait(delay)
                        delay = min(delay * 2, 30)
                        continue
                    known_hosts.write_text(verified_host_key(scan.stdout), encoding="utf-8")
                    os.chmod(known_hosts, 0o600)
                    self.status("Connecting…")
                    process = subprocess.Popen(
                        ssh_command(shutil.which("ssh"), key, known_hosts),
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options,
                    )
                    self.process = process
                    # A surviving SSH process is a tunnel indicator, not GPU readiness proof.
                    started = time.monotonic()
                    reported = False
                    while not self.stop_event.wait(0.2) and process.poll() is None:
                        if not reported and time.monotonic() - started >= 12:
                            self.status("Tunnel running · controller will verify the GPU")
                            reported = True
                            delay = 2
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    self.process = None
                except RuntimeError as error:
                    self.status(str(error))
                    return  # Pin mismatch is never silently retried or accepted.
                except (OSError, subprocess.TimeoutExpired):
                    self.status("Connection unavailable. Retrying…")
                if not self.stop_event.is_set():
                    self.status("Connection closed. Check the relay key and node; retrying…")
                    self.stop_event.wait(delay)
                    delay = min(delay * 2, 30)
        self.status("Disconnected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default=str(Path(__file__).resolve().with_name("relay_ed25519")),
                        help="Dedicated relay private key file (defaults to sibling relay_ed25519)")
    parser.add_argument("--headless", action="store_true", help="Run without a window")
    args = parser.parse_args()
    if args.headless:
        if not args.key:
            parser.error("--headless requires --key")
        connector = Connector(print)
        try:
            connector.start(args.key)
            while connector.thread and connector.thread.is_alive():
                connector.thread.join(0.5)
        except KeyboardInterrupt:
            connector.stop()
            if connector.thread:
                connector.thread.join(20)
        except ValueError as error:
            parser.error(str(error))
        return

    import queue
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Connect GPU node")
    root.geometry("580x245")
    events: queue.Queue[str] = queue.Queue()
    connector = Connector(events.put)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Connect this GPU to your controller", font=("Arial", 15)).pack(anchor="w")
    ttk.Label(frame, text="Keep this window open while sharing the GPU.").pack(anchor="w", pady=(6, 12))
    key = tk.StringVar(value=args.key or "")
    row = ttk.Frame(frame)
    row.pack(fill="x")
    ttk.Entry(row, textvariable=key).pack(side="left", fill="x", expand=True)
    def browse():
        selected = filedialog.askopenfilename(title="Choose the relay private key")
        if selected:
            key.set(selected)
    ttk.Button(row, text="Choose key…", command=browse).pack(side="left", padx=(8, 0))
    buttons = ttk.Frame(frame)
    buttons.pack(anchor="w", pady=12)
    def connect():
        try:
            connector.start(key.get())
        except ValueError as error:
            messagebox.showerror("Cannot connect", str(error))
    ttk.Button(buttons, text="Connect", command=connect).pack(side="left")
    ttk.Button(buttons, text="Disconnect", command=connector.stop).pack(side="left", padx=8)
    status = tk.StringVar(value="Disconnected")
    ttk.Label(frame, textvariable=status, wraplength=530).pack(anchor="w")
    def poll():
        while not events.empty():
            status.set(events.get_nowait())
        root.after(100, poll)
    def close():
        connector.stop()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(100, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
