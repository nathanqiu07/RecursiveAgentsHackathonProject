"""Ownership and loopback safety checks using temporary, unprivileged services."""

import contextlib
import json
import os
import signal
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_stack as stack


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"status": "ok", "database": "sqlite"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@contextlib.contextmanager
def external_api():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class StackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_local_control_plane_links_reject_remote_targets_and_conflicting_ports(
        self,
    ):
        with patch.dict(os.environ, {}, clear=True):
            env, definitions = stack.environment(stack.ROOT, self.root / "absent.env")
            self.assertTrue(
                all(
                    value.startswith("http://127.0.0.1:")
                    for key, value in env.items()
                    if key in stack.LOCAL_LINKS
                )
            )
            self.assertTrue(
                all(
                    "127.0.0.1" in service.command
                    for service in definitions
                    if service.port and service.title
                )
            )
        for value in (
            "https://example.com:8003",
            "http://0.0.0.0:8003",
            "http://localhost:8003",
            "http://127.0.0.1:9009",
            "http://user:secret@127.0.0.1:8003",
        ):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"AGENT_HARNESS_URL": value}, clear=True),
                self.assertRaises(stack.StackError),
            ):
                stack.environment(stack.ROOT, self.root / "absent.env")
        with self.assertRaises(stack.StackError):
            stack.definitions(stack.ROOT, {"DASHBOARD_API_PORT": "8003"})

    def test_shared_verifier_key_stable_and_private(self):
        path = self.root / "private/local.env"
        stack.initialize(path)
        original = path.read_bytes()
        stack.initialize(path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        env = stack.read_env(path)
        self.assertGreaterEqual(len(env["NEMOTRON_VERIFIER_API_KEY"]), 40)
        self.assertEqual(
            env["NEMOTRON_VERIFIER_API_KEY"], env["BENCHMARK_VERIFIER_NEMOTRON_KEY"]
        )
        path.chmod(0o644)
        with self.assertRaises(stack.StackError):
            stack.read_env(path)

    def test_state_permissions_and_pid_reuse_fails_ownership(self):
        path = self.root / "data/stack.json"
        stack.atomic_json(path, {"version": 1, "processes": {}})
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
        # A real PID is not sufficient to own a process; its identity and session
        # must also agree. Never signal a process using a stale PID record.
        self.assertFalse(stack.owned({"pid": os.getpid(), "identity": "stale-process"}))
        self.assertFalse(
            stack.owned({"pid": os.getpid(), "identity": stack.identity(os.getpid())})
        )

    def test_reuse_healthy_external_api_and_leave_it_running_on_down(self):
        with external_api() as port:
            service = stack.Service(
                "dashboard-api", self.root, (sys.executable,), port, "/api/health"
            )
            manager = stack.Stack(
                self.root / "data", dict(os.environ), [service], timeout=2
            )
            self.assertEqual(manager.launch(service), ("reused", None))
            self.assertEqual(manager.state["processes"], {})
            self.assertEqual(manager.down({service.name}), [])
            self.assertTrue(stack.healthy(service))

    def test_different_service_on_port_is_not_launched_over(self):
        with external_api() as port:
            service = stack.Service(
                "resource-manager-api",
                self.root,
                (sys.executable,),
                port,
                title="Resource Manager",
            )
            manager = stack.Stack(
                self.root / "data", dict(os.environ), [service], timeout=2
            )
            with self.assertRaisesRegex(stack.StackError, "occupied"):
                manager.launch(service)
            self.assertTrue(stack.occupied(port))
            self.assertEqual(manager.state["processes"], {})

    def test_spawn_worker_idempotently_and_stop_only_owned_process_group(self):
        service = stack.Service(
            "test-worker",
            self.root,
            (sys.executable, "-c", "import time; time.sleep(120)"),
        )
        manager = stack.Stack(
            self.root / "data", dict(os.environ), [service], timeout=3
        )
        with patch.object(stack, "runtime_check", return_value=True):
            status, pid = manager.launch(service)
        try:
            self.assertEqual(status, "started")
            self.assertTrue(stack.owned(manager.state["processes"][service.name]))
            self.assertEqual(manager.launch(service), ("owned", pid))
            results = manager.down({service.name})
            self.assertEqual(results, [{"name": service.name, "status": "stopped"}])
            self.assertEqual(manager.state["processes"], {})
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)

    def test_stale_record_never_signals_external_process(self):
        manager = stack.Stack(self.root / "data", dict(os.environ), [], timeout=1)
        manager.state["processes"]["unowned"] = {"pid": os.getpid(), "identity": "old"}
        with patch.object(stack.os, "killpg") as kill:
            manager.down({"unowned"})
        kill.assert_not_called()

    def test_dry_run_never_creates_key_state_or_process(self):
        data = self.root / "not-created"
        with (
            patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(__import__("io").StringIO()),
        ):
            self.assertEqual(
                stack.main(["up", "--dry-run", "--data-dir", str(data)]), 0
            )
        self.assertFalse(data.exists())

    def test_readiness_has_no_fake_gpu_or_baseline(self):
        env = {name: "http://127.0.0.1:1" for name in stack.LOCAL_LINKS}

        def response(url, _key=None):
            if url.endswith("/resources"):
                return {"resources": []}
            return {"status": "ok", "ready": False}

        with patch.object(stack, "request_json", side_effect=response):
            ready = stack.readiness(env)
        self.assertFalse(ready["resource_capacity"]["ready"])
        self.assertFalse(ready["nemotron_setup"]["ready"])
        self.assertFalse(ready["nemotron_setup"]["baseline_ready"])


if __name__ == "__main__":
    unittest.main()
