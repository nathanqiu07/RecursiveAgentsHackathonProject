#!/usr/bin/env python3
"""Bounded VM job executor; install as /usr/local/bin/gpu-job-runner.

This standard-library process runner speaks the local harness execution protocol.
It is not an agent harness or verifier: orchestration stays on the local machine.
The per-job KVM guest provides process, filesystem and GPU isolation.
"""

import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

MAX_LINE_BYTES = 128 * 1024
MAX_COMMAND_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 64_000
MAX_COMMANDS = 128
DEFAULT_WORKSPACE = Path("/tmp/workspace")
ID_PATTERN = re.compile(r"[\x21-\x7e]{1,200}\Z")


def result(request_id, exit_code, output, truncated=False):
    return {
        "type": "execution_result",
        "id": request_id,
        "exit_code": exit_code,
        "output": output,
        "truncated": truncated,
    }


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can report EPERM for a process group containing only a reaped
        # or exiting child. Do not mask errors while our own child stays alive.
        process.wait(timeout=0.1)


def execute(command, timeout_seconds, workspace=DEFAULT_WORKSPACE):
    """Execute exclusively in the sandbox; bound output and stop descendants."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["/bin/sh", "-c", command],
        cwd=workspace,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(workspace),
            "TMPDIR": "/tmp",
            "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    output = bytearray()
    deadline = time.monotonic() + timeout_seconds
    stop_at = None
    forced_exit = None
    truncated = False
    try:
        os.set_blocking(process.stdout.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                now = time.monotonic()
                if stop_at is None:
                    if process.poll() is not None:
                        # Background children must not survive a finished shell
                        # or keep its pipe open for the next command.
                        _kill_group(process)
                        stop_at = now + 0.5
                    elif now >= deadline:
                        forced_exit = 124
                        _kill_group(process)
                        stop_at = now + 0.5
                if stop_at is not None and now >= stop_at:
                    break
                for key, _ in selector.select(0.02):
                    try:
                        chunk = os.read(key.fileobj.fileno(), 8192)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = MAX_OUTPUT_BYTES - len(output)
                    output.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        truncated = True
                        if forced_exit is None:
                            forced_exit = 125
                        _kill_group(process)
                        stop_at = time.monotonic() + 0.5
            # A shell may close stdout before it exits. It still has the same
            # command deadline, and any background descendants still need cleanup.
            if process.poll() is None and forced_exit is None:
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    forced_exit = 124
            _kill_group(process)
            process.wait(timeout=1)
    finally:
        _kill_group(process)
        try:
            process.wait(timeout=1)
        finally:
            process.stdout.close()
    return {
        "exit_code": forced_exit if forced_exit is not None else process.returncode,
        "output": output.decode("utf-8", errors="replace"),
        "truncated": truncated,
    }


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _constant(_value):
    raise ValueError("Non-finite JSON number")


def _request(line):
    value = json.loads(line, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict) or set(value) != {
        "type",
        "id",
        "command",
        "timeout_seconds",
    }:
        raise ValueError("Expected an execution request")
    request_id = value["id"]
    if not isinstance(request_id, str) or ID_PATTERN.fullmatch(request_id) is None:
        raise ValueError("Invalid command ID")
    command, timeout = value["command"], value["timeout_seconds"]
    if (
        value["type"] != "execute"
        or not isinstance(command, str)
        or not command.strip()
        or len(command.encode("utf-8")) > MAX_COMMAND_BYTES
        or isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= 300
    ):
        raise ValueError("Invalid execution request")
    return value


def serve(input_stream=None, output_stream=None, workspace=DEFAULT_WORKSPACE):
    """Process NDJSON commands, caching each ID to prevent repeated execution."""
    input_stream = input_stream if input_stream is not None else sys.stdin.buffer
    output_stream = output_stream if output_stream is not None else sys.stdout
    cache = {}
    while True:
        line = input_stream.readline(MAX_LINE_BYTES + 1)
        if not line:
            return
        newline = b"\n" if isinstance(line, bytes) else "\n"
        raw = (
            line if isinstance(line, bytes) else line.encode("utf-8", errors="replace")
        )
        if len(raw) > MAX_LINE_BYTES:
            # Consume this request only; a later, valid line remains usable.
            while not line.endswith(newline):
                line = input_stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
            reply = result("", 64, "Execution request exceeds the input limit")
        else:
            request_id = ""
            try:
                request = _request(raw)
                request_id = request["id"]
                signature = (request["command"], request["timeout_seconds"])
                if request_id in cache:
                    previous, saved = cache[request_id]
                    if previous != signature:
                        reply = result(
                            request_id, 64, "Command ID was reused with different input"
                        )
                    else:
                        reply = saved
                elif len(cache) >= MAX_COMMANDS:
                    reply = result(request_id, 64, "Sandbox command limit reached")
                else:
                    saved = execute(
                        request["command"], request["timeout_seconds"], workspace
                    )
                    reply = result(request_id, **saved)
                    cache[request_id] = (signature, reply)
            except (ValueError, TypeError, UnicodeError, RecursionError):
                reply = result(request_id, 64, "Invalid execution request")
            except OSError:
                reply = result(request_id, 127, "Could not start the sandbox command")
        output_stream.write(
            json.dumps(reply, ensure_ascii=True, allow_nan=False) + "\n"
        )
        output_stream.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        measured = execute("printf node-runner-ready", 5)
        if measured["exit_code"] != 0 or measured["output"] != "node-runner-ready":
            raise SystemExit(2)
        print(json.dumps({"ready": True, "protocol": "execution_result-jsonl-v1"}))
    else:
        serve()
