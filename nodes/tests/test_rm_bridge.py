"""Run with ResourceManager/.venv/bin/python: exercises real HTTP and durable local queue."""

import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ResourceManager"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from resource_manager.remote_vm_runtime import RemoteVMRuntime
    from resource_manager.runtime import RuntimeFailure
except ImportError:
    RemoteVMRuntime = None

import common
import test_nodes
from test_nodes import allocation


@unittest.skipIf(
    RemoteVMRuntime is None,
    "use ResourceManager/.venv/bin/python for bridge integration",
)
class BridgeTests(unittest.TestCase):
    setUp = test_nodes.NodeProtocolTests.setUp
    tearDown = test_nodes.NodeProtocolTests.tearDown
    call = test_nodes.NodeProtocolTests.call
    wait_status = test_nodes.NodeProtocolTests.wait_status
    submit = test_nodes.NodeProtocolTests.submit

    def test_local_durable_queue_roundtrip_through_node_http(self):
        from resource_manager.models import (
            JobSubmission,
            NodeRegistration,
            StdinMessage,
        )
        from resource_manager.queue import ComputeQueue
        from resource_manager.resource_catalog import ResourceCatalog
        from resource_manager.worker import run_once

        receipt_path = Path(self.directory.name) / "connection.json"
        common.private_json(
            receipt_path,
            {
                "node_id": "test-node",
                "status": "ready",
                "address": self.address,
                "key": self.key,
                "gpu_uuid": "GPU-12345678-1234",
            },
        )
        catalog = ResourceCatalog(
            [
                {
                    "id": "gpu-test",
                    "node_id": "test-node",
                    "enabled": True,
                    "runtime": "remote_vm",
                    "node_manifest": str(receipt_path),
                    "image": "approved-vm:test",
                    "cpus": 4.0,
                    "memory_mb": 8192,
                    "scratch_mb": 65536,
                    "max_timeout_seconds": 20,
                    "max_concurrent_jobs": 1,
                }
            ]
        )
        queue = ComputeQueue(Path(self.directory.name) / "queue.sqlite3")
        queue.register_node(NodeRegistration(id="test-node", cpus=4.0, memory_mb=8192))
        data = allocation()
        data.update(node_id="test-node", cpus=2.0)
        job = queue.submit(JobSubmission(**data), "vm-roundtrip", catalog)
        queue.put_stdin(job["id"], StdinMessage(id="line", line="actual-remote-jsonl"))
        queue.close_stdin(job["id"])
        self.assertTrue(run_once(queue, catalog, "test-local-worker"))
        result = queue.get(job["id"])
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["cleanup_required"])
        self.assertEqual(
            queue.stdout(job["id"])["lines"][0]["line"], "actual-remote-jsonl"
        )
        self.assertEqual(queue.resources(catalog)[0]["cpus"], 4.0)

    def test_bad_manifest_never_creates_remote_job(self):
        from types import SimpleNamespace

        path = Path(self.directory.name) / "connection.json"
        common.private_json(
            path,
            {
                "node_id": "test-node",
                "status": "ready",
                "address": "https://external.example",
                "key": self.key,
            },
        )
        runtime = RemoteVMRuntime(
            SimpleNamespace(node_manifest=str(path), node_id="test-node")
        )
        with self.assertRaisesRegex(RuntimeFailure, "remote_vm_manifest_invalid"):
            runtime.check_ready()
        self.assertTrue(runtime.cleanup_confirmed)
        self.assertIsNone(self.node.active)

    def test_lost_create_acknowledgement_requires_confirmed_remote_cleanup(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        runtime = RemoteVMRuntime(SimpleNamespace())
        job = {**allocation(), "id": str(uuid.uuid4())}
        queue = SimpleNamespace(heartbeat=lambda *_: True)

        def request(path, method="GET", *args, **kwargs):
            if path == "/v1/jobs":
                raise RuntimeFailure("remote_vm_control_failed")
            if path.endswith("/cancel"):
                return {}
            return {"cleanup_confirmed": True}

        with (
            patch.object(runtime, "check_ready"),
            patch.object(runtime, "request", side_effect=request),
            self.assertRaisesRegex(RuntimeFailure, "remote_vm_control_failed"),
        ):
            runtime.run(job, "token", queue)
        self.assertTrue(runtime.cleanup_confirmed)
