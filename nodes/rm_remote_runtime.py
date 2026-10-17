"""ResourceManager bridge; installed as resource_manager.remote_vm_runtime.

Control plane and durable queue remain local. Only VM job execution reaches the node.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .queue import JobCancelled
from .runtime import RuntimeFailure


class RemoteVMRuntime:
    def __init__(self, resource):
        self.resource = resource
        self.cleanup_confirmed = True
        self.receipt = None

    def request(self, path, method="GET", body=None, request_id=None, timeout=3):
        headers = {
            "Authorization": "Bearer " + self.receipt["key"],
            "Content-Type": "application/json",
        }
        if request_id:
            headers["Idempotency-Key"] = request_id
        request = urllib.request.Request(
            self.receipt["address"] + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read(20 * 1024 * 1024 + 1)
            if len(data) > 20 * 1024 * 1024:
                raise RuntimeFailure("remote_vm_reply_limit")
            return json.loads(data)
        except urllib.error.HTTPError as exc:
            try:
                error = json.loads(exc.read(8192)).get("error")
            except (ValueError, AttributeError):
                error = None
            if exc.code == 409 and error == "node_not_ready_or_busy":
                raise RuntimeFailure("remote_vm_busy") from None
            raise RuntimeFailure("remote_vm_control_failed") from None
        except (urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
            # No raw HTTP/SSH/guest diagnostics or scoped keys in generic queue errors.
            raise RuntimeFailure("remote_vm_control_failed") from exc

    def check_ready(self):
        path = Path(self.resource.node_manifest)
        if path.stat().st_mode & 0o077:
            raise RuntimeFailure("remote_vm_manifest_must_be_private")
        receipt = json.loads(path.read_text())
        url = urllib.parse.urlsplit(receipt["address"])
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or not url.port
            or url.path
            or url.username
            or url.password
            or url.query
            or url.fragment
            or receipt.get("node_id") != self.resource.node_id
            or receipt.get("status") != "ready"
            or not isinstance(receipt.get("key"), str)
            or not receipt["key"].startswith("rn_")
        ):
            raise RuntimeFailure("remote_vm_manifest_invalid")
        self.receipt = receipt
        health = self.request("/v1/health")
        if (
            health.get("node_id") != self.resource.node_id
            or health.get("status") != "ready"
            or health.get("runtime") != "libvirt-kvm-vfio"
            or not health.get("verification", {}).get("boot_verified")
        ):
            raise RuntimeFailure("remote_vm_not_ready")

    def run(self, job, token, queue):
        deadline = time.monotonic() + job["timeout_seconds"]
        node_id = job["id"]  # Idempotent creation uses the local UUID verbatim.
        prefix = "/v1/jobs/" + node_id
        attempted = False
        cursor = 0
        eof_sent = False
        running = False
        outcome = None
        try:
            self.check_ready()
            if not queue.heartbeat(job["id"], token):
                raise JobCancelled
            body = {
                key: job[key]
                for key in (
                    "resource_id",
                    "cpus",
                    "memory_mb",
                    "scratch_mb",
                    "timeout_seconds",
                )
            }
            while True:
                if not queue.heartbeat(job["id"], token):
                    raise JobCancelled
                if time.monotonic() >= deadline:
                    raise RuntimeFailure("compute_timeout")
                attempted = True
                self.cleanup_confirmed = False
                try:
                    self.request("/v1/jobs", "POST", body, request_id=node_id)
                    break
                except RuntimeFailure as exc:
                    if str(exc) != "remote_vm_busy":
                        raise
                    # Another local client/verifier owns the one GPU; nothing was created for us.
                    attempted = False
                    self.cleanup_confirmed = True
                    time.sleep(0.5)
            while time.monotonic() < deadline:
                if not queue.heartbeat(job["id"], token):
                    raise JobCancelled
                state = self.request(prefix)
                if state["status"] == "running" and not running:
                    queue.mark_running(job["id"], token)
                    running = True
                terminal = state["status"] in {"completed", "failed", "cancelled"}
                message, closed = queue.pending_stdin(job["id"])
                if message is not None and not terminal:
                    self.request(
                        prefix + "/stdin",
                        "POST",
                        {"id": str(message["sequence"]), "line": message["line"]},
                    )
                    queue.ack_stdin(job["id"], message["sequence"])
                elif closed and not eof_sent and not terminal:
                    self.request(prefix + "/stdin/close", "POST", {})
                    eof_sent = True
                output = self.request(prefix + "/stdout?after=" + str(cursor))
                for line in output["lines"]:
                    if (
                        type(line.get("sequence")) is not int
                        or line["sequence"] != cursor + 1
                    ):
                        raise RuntimeFailure("remote_vm_output_sequence_invalid")
                    queue.append_stdout(job["id"], token, line["line"])
                    cursor = line["sequence"]
                terminal = state["status"] in {"completed", "failed", "cancelled"}
                if (
                    terminal
                    and not output["lines"]
                    and state.get("cleanup_confirmed") is True
                ):
                    self.cleanup_confirmed = True
                    if state["status"] != "completed":
                        raise RuntimeFailure("remote_vm_job_failed")
                    outcome = state["exit_code"]
                    if type(outcome) is not int:
                        raise RuntimeFailure("remote_vm_exit_unconfirmed")
                    return outcome
                if terminal and state.get("error") == "vm_cleanup_unconfirmed":
                    raise RuntimeFailure("remote_vm_cleanup_unconfirmed")
                time.sleep(0.1)
            raise RuntimeFailure("compute_timeout")
        finally:
            if attempted and not self.cleanup_confirmed:
                # On a lost create acknowledgement the UUID still identifies the remote job.
                try:
                    self.request(prefix + "/cancel", "POST", {})
                    until = time.monotonic() + 10
                    while time.monotonic() < until:
                        state = self.request(prefix)
                        if state.get("cleanup_confirmed") is True:
                            self.cleanup_confirmed = True
                            break
                        time.sleep(0.1)
                except RuntimeFailure:
                    pass
            # Worker deliberately retains capacity when remote cleanup cannot be verified.
