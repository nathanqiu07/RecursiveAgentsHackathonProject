"""Run with ResourceManager/.venv/bin/python: exercises real HTTP and durable local queue."""

import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ResourceManager"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

@unittest.skipIf(RemoteVMRuntime is None, "use ResourceManager virtualenv")
class DirectBridgeTests(unittest.TestCase):
    def test_direct_mode_requires_explicit_receipt_and_runner_probe(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connection.json"
            resource = SimpleNamespace(node_manifest=str(path), node_id="a4000-01")
            receipt = {"node_id": "a4000-01", "status": "ready", "address": "http://127.0.0.1:18100", "key": "rn_test", "runtime": "direct-host-systemd"}
            common.private_json(path, receipt)
            health = {"node_id": "a4000-01", "status": "ready", "runtime": "direct-host-systemd", "verification": {"direct_execution_verified": True, "runner_verified": True, "boot_verified": False}}
            with patch.object(RemoteVMRuntime, "request", return_value=health):
                RemoteVMRuntime(resource).check_ready()
                receipt.pop("runtime")
                common.private_json(path, receipt)
                with self.assertRaises(RuntimeFailure):
                    RemoteVMRuntime(resource).check_ready()

@unittest.skipIf(RemoteVMRuntime is None, "use ResourceManager virtualenv")
class DockerBridgeTests(unittest.TestCase):
    def test_container_requires_matching_enrollment_and_complete_probe(self):
        import copy
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "connection.json"
            resource = SimpleNamespace(node_manifest=str(path), node_id="docker-a5000")
            receipt = {
                "node_id": resource.node_id, "status": "ready",
                "address": "http://127.0.0.1:18100", "key": "rn_test",
                "runtime": "docker-wsl2-nvidia", "gpu_uuid": "GPU-enrolled",
                "target": {"container_image": "sha256:" + "a" * 64},
            }
            health = {
                "node_id": resource.node_id, "status": "ready", "runtime": receipt["runtime"],
                "verification": {
                    "container_verified": True, "runner_verified": True,
                    "gpu_compute_verified": True, "cleanup_confirmed": True,
                    "boot_verified": False, "image_sha256": "a" * 64,
                    "gpu": {"uuid": receipt["gpu_uuid"]},
                },
            }
            common.private_json(path, receipt)
            with patch.object(RemoteVMRuntime, "request", return_value=health):
                RemoteVMRuntime(resource).check_ready()
            for field in ("container_verified", "runner_verified", "gpu_compute_verified", "cleanup_confirmed"):
                for invalid in (False, 1, "true", None):
                    broken = copy.deepcopy(health)
                    broken["verification"][field] = invalid
                    with self.subTest(field=field, value=invalid), patch.object(RemoteVMRuntime, "request", return_value=broken):
                        with self.assertRaisesRegex(RuntimeFailure, "remote_vm_not_ready"):
                            RemoteVMRuntime(resource).check_ready()
            for mutate in (
                lambda r, h: r.pop("runtime"),
                lambda r, h: h.update(runtime="libvirt-kvm-vfio", verification={"boot_verified": True}),
                lambda r, h: h["verification"].update(image_sha256="b" * 64),
                lambda r, h: h["verification"]["gpu"].update(uuid="GPU-other"),
                lambda r, h: h["verification"].update(boot_verified=True),
            ):
                changed_receipt, changed_health = copy.deepcopy(receipt), copy.deepcopy(health)
                mutate(changed_receipt, changed_health)
                common.private_json(path, changed_receipt)
                with patch.object(RemoteVMRuntime, "request", return_value=changed_health):
                    with self.assertRaisesRegex(RuntimeFailure, "remote_vm_not_ready"):
                        RemoteVMRuntime(resource).check_ready()
