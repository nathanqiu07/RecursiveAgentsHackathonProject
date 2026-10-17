import base64
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("connector_app", Path(__file__).resolve().parents[1] / "connector_app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class ConnectorTests(unittest.TestCase):
    def test_scan_must_match_pin(self):
        blob = base64.b64encode(b"test-host-public-key").decode()
        pin = "SHA256:" + base64.b64encode(hashlib.sha256(b"test-host-public-key").digest()).decode().rstrip("=")
        with patch.object(app, "RELAY_FINGERPRINT", pin):
            result = app.verified_host_key(f"untrusted-host ssh-ed25519 {blob}\n")
        self.assertEqual(result, f"{app.RELAY_HOST} ssh-ed25519 {blob}\n")
        with self.assertRaises(RuntimeError):
            app.verified_host_key(f"{app.RELAY_HOST} ssh-ed25519 {blob}\n")

    def test_malformed_and_other_algorithm_rejected(self):
        for scan in ("", "# comment", "host ssh-ed25519 @@@@", "host ssh-rsa dGVzdA=="):
            with self.subTest(scan=scan), self.assertRaises(RuntimeError):
                app.verified_host_key(scan)

    def test_tunnel_is_private_and_host_pinned(self):
        command = app.ssh_command("ssh", Path("/private/key"), Path("/private/known_hosts"))
        self.assertIn("127.0.0.1:18200:127.0.0.1:18090", command)
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertIn("BatchMode=yes", command)
        self.assertIn("ExitOnForwardFailure=yes", command)
        self.assertIn("HostKeyAlgorithms=ssh-ed25519", command)
        self.assertIn("ForwardAgent=no", command)
        self.assertIn("-N", command)
        self.assertNotIn("-g", command)

    def test_missing_key_does_not_start_thread(self):
        connector = app.Connector(lambda message: None)
        with self.assertRaises(ValueError):
            connector.start("/nonexistent/gpu-connector-test-key")
        self.assertIsNone(connector.thread)


if __name__ == "__main__":
    unittest.main()
