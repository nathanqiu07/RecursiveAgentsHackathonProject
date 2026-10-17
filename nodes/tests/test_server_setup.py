import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import server_setup


def target():
    return common.validate_target(
        {
            "id": "server-01",
            "ssh_host": "server-01",
            "local_port": 18100,
            "remote_port": 18090,
            "guest_image": "/approved.qcow2",
            "guest_image_sha256": "a" * 64,
            "gpu_pci_devices": ["0000:01:00.0", "0000:01:00.1"],
            "gpu_uuid": "GPU-12345678-1234",
            "guest_memory_mb": 8192,
            "guest_vcpus": 4,
            "scratch_mb": 65536,
            "guest_job_command": ["/usr/local/bin/gpu-job-runner"],
            "libvirt_network": "gpu-jobs",
        }
    )


class ServerSetupTests(unittest.TestCase):
    def test_failed_controller_import_closes_unverified_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            exported = Path(directory) / "export.json"
            document = {
                "node_id": "server-01",
                "status": "ready",
                "transport": "server-local",
                "target": target(),
                "gpu_uuid": target()["gpu_uuid"],
                "key": "rn_private_fixture",
            }
            common.private_json(exported, document)
            with (
                patch.object(
                    server_setup, "open_tunnel", return_value="/private/socket"
                ),
                patch.object(server_setup, "close_tunnel") as close,
                patch.object(
                    server_setup,
                    "verify_connection",
                    side_effect=common.Blocked("invalid_node_key"),
                ),
                self.assertRaisesRegex(common.Blocked, "invalid_node_key"),
            ):
                server_setup.connect_receipt(
                    exported, Path(directory) / "laptop", "trusted-server"
                )
            self.assertEqual(close.call_args.args[1], "/private/socket")
            self.assertFalse(
                (Path(directory) / "laptop" / "server-01" / "connection.json").exists()
            )

    def test_local_install_reuses_real_installer_without_ssh_or_key_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []

            def run(argv, **kwargs):
                calls.append((argv, kwargs))
                return SimpleNamespace(returncode=0, stdout=b'{"status":"installed"}\n')

            verification = {
                "boot_verified": True,
                "runner_verified": True,
                "gpu": {"uuid": target()["gpu_uuid"]},
            }
            with (
                patch.object(server_setup.platform, "system", return_value="Linux"),
                patch.object(server_setup.platform, "machine", return_value="x86_64"),
                patch.object(server_setup.subprocess, "run", side_effect=run),
                patch.object(
                    server_setup, "http", return_value={"status": "unverified"}
                ),
                patch.object(
                    server_setup,
                    "verify_connection",
                    return_value={"verification": verification},
                ),
            ):
                receipt, path = server_setup.install_local(
                    target(),
                    directory,
                    {"hostname": "actual-host", "user": "ubuntu", "port": 22},
                )
                self.assertEqual(receipt["status"], "ready")
                self.assertEqual(receipt["address"], "http://127.0.0.1:18090")
                self.assertTrue(receipt["key"].startswith("rn_"))
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertTrue(receipt["verification"]["boot_verified"])
                key = receipt["key"]
                self.assertNotIn(key, json.dumps(calls[0][0]))
                self.assertNotIn("ssh", calls[0][0])
                with tarfile.open(fileobj=io.BytesIO(calls[0][1]["input"])) as bundle:
                    installed = json.load(bundle.extractfile("config.json"))
                self.assertNotIn("key", installed)
                self.assertIn("key_sha256", installed)
                again, _ = server_setup.install_local(
                    target(), directory, receipt["ssh_connection"]
                )
                self.assertEqual(again["key"], key)
                self.assertNotIn(
                    key, json.dumps(server_setup.safe_result(receipt, path))
                )

    def test_controller_import_opens_strict_existing_tunnel_and_preserves_key_privately(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            exported = Path(directory) / "export.json"
            document = {
                "node_id": "server-01",
                "status": "ready",
                "address": "http://127.0.0.1:18090",
                "transport": "server-local",
                "target": target(),
                "gpu_uuid": target()["gpu_uuid"],
                "key": "rn_private_fixture",
                "verification": {"boot_verified": True},
            }
            common.private_json(exported, document)
            with (
                patch.object(
                    server_setup, "open_tunnel", return_value="/private/socket"
                ) as tunnel,
                patch.object(
                    server_setup,
                    "verify_connection",
                    return_value={"verification": {"boot_verified": True}},
                ),
            ):
                receipt, path = server_setup.connect_receipt(
                    exported, Path(directory) / "laptop", "trusted-server"
                )
                self.assertEqual(receipt["address"], "http://127.0.0.1:18100")
                self.assertEqual(receipt["transport"], "ssh-tunnel")
                self.assertEqual(tunnel.call_args.args[0]["ssh_host"], "trusted-server")
                self.assertEqual(
                    json.loads(path.read_text())["key"], "rn_private_fixture"
                )
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                server_setup.connect_receipt(
                    exported, Path(directory) / "laptop", "trusted-server"
                )
                self.assertEqual(tunnel.call_count, 1)

    def test_failed_install_never_returns_ready_and_preserves_recoverable_private_key(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(server_setup.platform, "system", return_value="Linux"),
                patch.object(server_setup.platform, "machine", return_value="x86_64"),
                patch.object(
                    server_setup.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=2,
                        stdout=b'{"status":"blocked","error":"iommu_disabled"}\n',
                    ),
                ),
                self.assertRaisesRegex(common.Blocked, "iommu_disabled"),
            ):
                server_setup.install_local(
                    target(), directory, {"hostname": "real-host"}
                )
            receipt = json.loads(
                (Path(directory) / "server-01" / "server-connection.json").read_text()
            )
            self.assertEqual(receipt["status"], "provisioning")
            self.assertIsNone(receipt["verification"])

    def test_target_selection_and_ssh_values_must_be_real_data_not_shell_options(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.json"
            path.write_text(
                json.dumps({"nodes": [target(), dict(target(), id="server-02")]})
            )
            self.assertEqual(
                server_setup.target_from_file(path, "server-01")["id"], "server-01"
            )
            with self.assertRaises(ValueError):
                server_setup.target_from_file(path)
            path.write_text(json.dumps({"nodes": [None]}))
            with self.assertRaisesRegex(ValueError, "target_nodes_must_be_objects"):
                server_setup.target_from_file(path, "server-01")
        with self.assertRaises(ValueError):
            server_setup.ssh_metadata("-oProxyCommand=bad", "ubuntu", 22)
        with self.assertRaises(ValueError):
            server_setup.ssh_metadata("server.example", "root; bad", 22)


if __name__ == "__main__":
    unittest.main()
