"""Shared availability includes GPU VMs held outside the experiment queue."""

import contextlib
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rm_resource_status as discovery


class Queue:
    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""CREATE TABLE compute_nodes(id TEXT, cpus REAL, memory_mb INTEGER);
            CREATE TABLE compute_jobs(node_id TEXT, cpus REAL, memory_mb INTEGER, allocation_held INTEGER);
            INSERT INTO compute_nodes VALUES('gpu-one',8,65536);""")

    @contextlib.contextmanager
    def connection(self):
        yield self.db

    def usage(self, database, node_id):
        rows = database.execute(
            "SELECT * FROM compute_jobs WHERE node_id=? AND allocation_held=1",
            (node_id,),
        ).fetchall()
        return (
            len(rows),
            sum(row["cpus"] for row in rows),
            sum(row["memory_mb"] for row in rows),
        )

    def node_public(self, _row, _now):
        return {"status": "online"}


class ResourceStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "connection.json"
        self.resource = SimpleNamespace(
            id="a5000-one",
            node_id="gpu-one",
            runtime="remote_vm",
            node_manifest=str(self.path),
            enabled=True,
            cpus=8.0,
            memory_mb=65536,
            max_concurrent_jobs=1,
            max_timeout_seconds=21600,
            description="A5000 GPU VM",
        )
        self.catalog = SimpleNamespace(resources=[self.resource])

    def tearDown(self):
        self.temporary.cleanup()

    def test_live_verifier_reservation_hides_pool_from_new_experiments(self):
        calls = []
        health = {
            "node_id": "gpu-one",
            "status": "ready",
            "runtime": "libvirt-kvm-vfio",
            "busy": True,
            "verification": {"boot_verified": True},
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append((self.path, self.headers.get("Authorization")))
                raw = json.dumps(health).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.path.write_text(
            json.dumps(
                {
                    "node_id": "gpu-one",
                    "status": "ready",
                    "key": "rn_test-scoped-key",
                    "address": f"http://127.0.0.1:{server.server_address[1]}",
                }
            )
        )
        self.path.chmod(0o600)
        queue = Queue()
        try:
            status = discovery.resource_status(queue, self.catalog)
            self.assertEqual(status["resources"][0]["status"], "busy")
            self.assertEqual(status["resources"][0]["held_jobs"], 0)
            self.assertEqual(status["resources"][0]["available_slots"], 0)
            self.assertEqual(
                discovery.filter_available([{"id": "a5000-one"}], self.catalog), []
            )
            health["busy"] = False
            self.assertEqual(
                discovery.filter_available([{"id": "a5000-one"}], self.catalog),
                [{"id": "a5000-one"}],
            )
            self.assertTrue(
                all(
                    path == "/v1/health" and key == "Bearer rn_test-scoped-key"
                    for path, key in calls
                )
            )
            self.assertNotIn("rn_test-scoped-key", json.dumps(status))
        finally:
            queue.db.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_container_discovery_requires_matching_gpu_image_and_complete_verification(self):
        import copy
        import io

        receipt = {
            "node_id": "gpu-one", "status": "ready", "key": "rn_scoped",
            "address": "http://127.0.0.1:18100", "runtime": "docker-wsl2-nvidia",
            "gpu_uuid": "GPU-enrolled", "target": {"container_image": "sha256:" + "a" * 64},
        }
        health = {
            "node_id": "gpu-one", "status": "ready", "busy": False,
            "runtime": "docker-wsl2-nvidia", "verification": {
                "container_verified": True, "runner_verified": True,
                "gpu_compute_verified": True, "cleanup_confirmed": True,
                "boot_verified": False, "image_sha256": "a" * 64,
                "gpu": {"uuid": "GPU-enrolled"},
            },
        }

        def state(enrolled, live):
            self.path.write_text(json.dumps(enrolled))
            self.path.chmod(0o600)
            response = io.BytesIO(json.dumps(live).encode())
            opener = SimpleNamespace(open=lambda *_args, **_kwargs: response)
            with patch.object(discovery.urllib.request, "build_opener", return_value=opener):
                return discovery.remote_state(self.resource)

        self.assertEqual(state(receipt, health), {"status": "available"})
        health["busy"] = True
        self.assertEqual(state(receipt, health), {"status": "busy"})
        for field in ("container_verified", "runner_verified", "gpu_compute_verified", "cleanup_confirmed"):
            for invalid in (False, 1, "true", None):
                broken = copy.deepcopy(health)
                broken["verification"][field] = invalid
                with self.subTest(field=field, value=invalid):
                    self.assertEqual(state(receipt, broken), {"status": "unavailable"})
        for mutate in (
            lambda r, h: r.pop("runtime"),
            lambda r, h: h.update(runtime="libvirt-kvm-vfio", verification={"boot_verified": True}),
            lambda r, h: h["verification"].update(image_sha256="b" * 64),
            lambda r, h: h["verification"]["gpu"].update(uuid="GPU-other"),
            lambda r, h: h["verification"].update(boot_verified=True),
        ):
            changed_receipt, changed_health = copy.deepcopy(receipt), copy.deepcopy(health)
            mutate(changed_receipt, changed_health)
            self.assertEqual(state(changed_receipt, changed_health), {"status": "unavailable"})

    def test_held_cleanup_reservations_remain_busy_even_when_daemon_is_idle(self):
        queue = Queue()
        queue.db.execute("INSERT INTO compute_jobs VALUES('gpu-one',2,2048,1)")
        with patch.object(
            discovery, "remote_state", return_value={"status": "available"}
        ):
            result = discovery.resource_status(queue, self.catalog)["resources"][0]
        self.assertEqual(result["status"], "busy")
        self.assertEqual(result["held_jobs"], 1)
        self.assertEqual(result["available_slots"], 0)
        queue.db.close()

    def test_private_manifest_required_and_unreachable_nodes_not_available(self):
        self.path.write_text('{"address":"http://192.0.2.1:9090"}')
        self.path.chmod(0o600)
        self.assertEqual(
            discovery.remote_state(self.resource), {"status": "unavailable"}
        )
        self.path.chmod(0o644)
        self.assertEqual(
            discovery.remote_state(self.resource), {"status": "unavailable"}
        )
        with patch.object(
            discovery, "remote_state", return_value={"status": "unavailable"}
        ):
            self.assertEqual(
                discovery.filter_available([{"id": "a5000-one"}], self.catalog), []
            )


if __name__ == "__main__":
    unittest.main()
