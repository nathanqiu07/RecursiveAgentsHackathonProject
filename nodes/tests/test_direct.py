import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from direct_host import DirectNode, gpu_probe
from common import Blocked


class DirectTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.node = DirectNode({"state_dir": self.directory.name, "job_user": "resourcejob", "key_sha256": hashlib.sha256(b"test").hexdigest(), "target": {"id": "a4000-01", "resource_id": "gpu-a4000-01", "gpu_uuid": "GPU-real", "guest_vcpus": 4, "guest_memory_mb": 8192, "scratch_mb": 16384, "max_timeout_seconds": 21600}})

    def test_actual_a4000_identity_and_memory(self):
        result = subprocess.CompletedProcess([], 0, b"GPU-real, NVIDIA RTX A4000, 16376, 140.00, 580.95.05\n")
        with patch("direct_host.run", return_value=result):
            gpu = gpu_probe()
        self.assertEqual(gpu["name"], "NVIDIA RTX A4000")
        self.assertEqual(gpu["memory_total_mb"], 16376)
        self.assertEqual(gpu["source"], "direct_host_nvidia_smi")

    def test_probe_does_not_claim_vm_boot(self):
        with patch("direct_host.gpu_probe", return_value={"uuid": "GPU-real"}), patch("direct_host.run", return_value=subprocess.CompletedProcess([], 0, b'{"ready":true}')):
            result = self.node.probe()
        self.assertFalse(result["boot_verified"])
        self.assertIsNone(result["vm_isolation"])
        self.assertTrue(result["direct_execution_verified"])

    def test_probe_rejects_wrong_gpu(self):
        with patch("direct_host.gpu_probe", return_value={"uuid": "GPU-other"}):
            with self.assertRaises(Blocked):
                self.node.probe()

    def test_cleanup_never_confirms_active_unit(self):
        with patch("direct_host.run", return_value=subprocess.CompletedProcess([], 0, b"active\n")):
            self.assertFalse(self.node.cleanup("resource-job-test.service"))

    def test_startup_rejects_untrusted_unit_receipt(self):
        (Path(self.directory.name) / "bad.unit").write_text("ssh.service")
        with self.assertRaises(Blocked):
            self.node.startup()

    def test_rejects_deployment_alias(self):
        with self.assertRaises(ValueError):
            self.node.submit({}, "bad", "baseline")
