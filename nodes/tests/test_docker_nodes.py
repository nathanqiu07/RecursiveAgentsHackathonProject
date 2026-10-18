"""Container identity checks must never satisfy a VM-only receipt."""
import tempfile
import hashlib
import json
import unittest
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller
import docker_host
from common import Blocked


class DockerIdentityTests(unittest.TestCase):
    def test_approved_mount_detects_content_change_and_extra_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'mount'
            root.mkdir()
            (root/'weights').write_bytes(b'approved')
            manifest=Path(directory)/'manifest.json'
            manifest.write_text(json.dumps({'files':{'weights':hashlib.sha256(b'approved').hexdigest()},'symlinks':{}}))
            mount={'source':str(root),'target':'/models/test','manifest':str(manifest),
                   'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
            docker_host.verify_mount(mount)
            (root/'weights').write_bytes(b'changed')
            with self.assertRaisesRegex(Blocked,'content_changed'):
                docker_host.verify_mount(mount)
            (root/'weights').write_bytes(b'approved')
            (root/'extra').write_bytes(b'unapproved')
            with self.assertRaisesRegex(Blocked,'file_set_changed'):
                docker_host.verify_mount(mount)

    def test_approved_mount_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'mount'
            root.mkdir()
            (Path(directory)/'outside').write_bytes(b'outside')
            (root/'escape').symlink_to('../outside')
            manifest=Path(directory)/'manifest.json'
            manifest.write_text(json.dumps({'files':{},'symlinks':{'escape':'../outside'}}))
            mount={'source':str(root),'target':'/models/test','manifest':str(manifest),
                   'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(Blocked,'symlink_changed'):
                docker_host.verify_mount(mount)

    def receipt(self):
        return {"status": "ready", "key": "rn_test", "address": "http://127.0.0.1:18100",
                "node_id": "test-node", "gpu_uuid": "GPU-test1234", "runtime": docker_host.RUNTIME,
                "target": {"container_image": "sha256:" + "a" * 64}}

    def health(self):
        return {"node_id": "test-node", "status": "ready", "runtime": docker_host.RUNTIME,
                "verification": {"boot_verified": False, "container_verified": True,
                    "runner_verified": True, "gpu_compute_verified": True, "cleanup_confirmed": True,
                    "image_sha256": "a" * 64, "gpu": {"uuid": "GPU-test1234"}}}

    def test_container_requires_explicit_receipt_runtime_and_verified_compute_cleanup(self):
        receipt, health = self.receipt(), self.health()
        with patch.object(controller, "http", side_effect=[health, Blocked("invalid_node_key")]):
            self.assertEqual(controller.verify_connection(receipt), health)
        receipt.pop("runtime")
        with patch.object(controller, "http", return_value=health):
            with self.assertRaisesRegex(Blocked, "runtime_mismatch"):
                controller.verify_connection(receipt)
        for field in ("container_verified", "runner_verified", "gpu_compute_verified", "cleanup_confirmed"):
            broken = self.health()
            broken["verification"][field] = False
            with patch.object(controller, "http", return_value=broken):
                with self.assertRaisesRegex(Blocked, "readiness_not_verified"):
                    controller.verify_connection(self.receipt())

    def test_wrong_gpu_and_missing_key_enforcement_are_rejected(self):
        health = self.health()
        health["verification"]["gpu"]["uuid"] = "GPU-wrong123"
        with patch.object(controller, "http", return_value=health):
            with self.assertRaisesRegex(Blocked, "readiness_not_verified"):
                controller.verify_connection(self.receipt())
        with patch.object(controller, "http", return_value=self.health()):
            with self.assertRaisesRegex(Blocked, "does_not_enforce_key"):
                controller.verify_connection(self.receipt())

    def test_uncertain_create_never_releases_capacity_on_immediate_absence(self):
        with tempfile.TemporaryDirectory() as state:
            guest = docker_host.DockerGuest({"target": {"id": "node"}, "state_dir": state},
                         {"id": "job", "creation_pending": True})
            from types import SimpleNamespace
            with patch.object(docker_host, "run", return_value=SimpleNamespace(stdout=b"", returncode=0)):
                self.assertFalse(guest.cleanup())
    def test_create_uses_fixed_immutable_image_without_host_mounts_or_network(self):
        with tempfile.TemporaryDirectory() as state:
            target = {"id": "node", "gpu_uuid": "GPU-test1234", "container_image": "sha256:" + "a"*64,
                      "scratch_mb": 64, "guest_job_command": ["/usr/local/bin/gpu-job-runner"]}
            guest = docker_host.DockerGuest({"target": target, "state_dir": state},
                    {"id": "job", "memory_mb": 1024, "cpus": 1, "timeout_seconds": 60})
            def command(argv, **kwargs):
                if argv[1] == "exec":
                    raise Blocked("stop_after_create")
                from types import SimpleNamespace
                return SimpleNamespace(stdout=b"", returncode=0)
            with patch.object(docker_host, "run", side_effect=command) as run:
                with self.assertRaises(Blocked):
                    guest.boot()
                argv = run.call_args_list[0].args[0]
                self.assertIn("--read-only", argv)
                self.assertIn("--cap-drop", argv)
                self.assertEqual(argv[argv.index("--network") + 1], "none")
                self.assertEqual(argv[argv.index("--user") + 1], "1000:1000")
                self.assertNotIn("--privileged", argv)
                self.assertNotIn("--volume", argv)
                self.assertNotIn("--mount", argv)
                self.assertIn(target["container_image"], argv)


if __name__ == "__main__":
    unittest.main()
