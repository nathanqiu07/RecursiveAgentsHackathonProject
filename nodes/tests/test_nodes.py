import hashlib
import io
import json
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
import xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import controller
import guest_runner
import host


def target():
    return common.validate_target(
        {
            "id": "test-node",
            "resource_id": "gpu-test",
            "ssh_host": "gpu-host",
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
            "deployments": {
                "baseline": {
                    "command": ["/usr/bin/vllm", "serve", "model"],
                    "metadata": {
                        "repository": "https://github.com/org/vllm",
                        "commit": "b" * 40,
                        "model_revision": "c" * 40,
                        "execution": "W4A16",
                        "settings": {"max_model_len": 3072},
                    },
                }
            },
        }
    )


def allocation():
    return {
        "resource_id": "gpu-test",
        "cpus": 2,
        "memory_mb": 4096,
        "scratch_mb": 8192,
        "timeout_seconds": 10,
    }


class InventoryTests(unittest.TestCase):
    def test_unconfigured_inventory_is_truthful_blocked_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory) / "inventory.json"
            inventory.write_text('{"nodes":[]}')
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                code = controller.main(
                    ["enroll", "--inventory", str(inventory), "--state", directory]
                )
            report = json.loads(output.getvalue())
            self.assertEqual(code, 2)
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["nodes"], [])
            self.assertIn("no_ssh_hosts", report["error"])
            self.assertEqual(
                stat.S_IMODE((Path(directory) / "results.json").stat().st_mode), 0o600
            )

    def test_rejects_ssh_injection_nonimmutable_image_and_duplicate_hosts(self):
        for field, value in (
            ("ssh_host", "-oProxyCommand=bad"),
            ("guest_image_sha256", "latest"),
            ("guest_image", "relative.qcow2"),
            ("guest_job_command", ["sh"]),
        ):
            data = target()
            data[field] = value
            with self.assertRaises(ValueError):
                common.validate_target(data)
        data = target()
        data["deployments"]["baseline"]["metadata"]["commit"] = "main"
        with self.assertRaises(ValueError):
            common.validate_target(data)

    def test_connection_secrets_only_in_private_data_stream(self):
        key = "rn_actual_secret"
        data = controller.bundle_for(target(), key)
        self.assertNotIn(key.encode(), data)
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
            config = json.loads(bundle.extractfile("config.json").read())
        self.assertEqual(config["key_sha256"], hashlib.sha256(key.encode()).hexdigest())
        self.assertNotIn(key, controller.INSTALLER)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            common.private_json(path, {"key": key})
            self.assertEqual(common.read_private(path)["key"], key)
            path.chmod(0o644)
            with self.assertRaises(common.Blocked):
                common.read_private(path)

    def test_domain_is_actual_kvm_with_all_gpu_functions_and_no_implicit_detach(self):
        job = {"id": str(uuid.uuid4()), "memory_mb": 4096, "cpus": 1.5}
        root = ET.fromstring(
            host.domain_xml(target(), job, Path("/safe/job"), "52:54:00:00:00:01")
        )
        self.assertEqual(root.attrib["type"], "kvm")
        self.assertEqual(root.findtext("memory"), "4096")
        self.assertEqual(root.findtext("cputune/global_quota"), "150000")
        self.assertEqual(root.findtext("vcpu"), "2")
        devices = root.findall("./devices/hostdev")
        self.assertEqual(len(devices), 2)
        self.assertTrue(
            all(
                x.attrib == {"mode": "subsystem", "type": "pci", "managed": "no"}
                for x in devices
            )
        )
        self.assertEqual(
            root.find("./devices/disk/source").attrib["file"], "/safe/job/root.qcow2"
        )


class FakeVM:
    cleanup_ok = True

    def __init__(self, config, job):
        self.job = job
        self.ip = "127.0.0.1"

    def boot(self):
        return self.hardware()

    def hardware(self):
        return {
            "uuid": "GPU-12345678-1234",
            "name": "NVIDIA RTX A5000",
            "memory_total_mb": 24564,
            "power_limit_w": 230,
            "driver_version": "580.1",
            "observed_at": time.time(),
            "vm_id": self.job["id"],
        }

    def ssh_args(self):
        # Only protocol tests use this local echo process. No test claims real KVM/VFIO succeeded.
        return [
            sys.executable,
            "-u",
            "-c",
            "import sys; [(print(x.strip(),flush=True)) for x in sys.stdin]",
        ]

    def cleanup(self):
        return self.cleanup_ok


class NodeProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.key = "rn_scoped_node_secret"
        self.node = host.Node(
            {
                "target": target(),
                "state_dir": self.directory.name,
                "key_sha256": hashlib.sha256(self.key.encode()).hexdigest(),
            }
        )
        self.node.probe_result = {
            "boot_verified": True,
            "runner_verified": True,
            "gpu": {"uuid": "GPU-12345678-1234"},
        }
        self.vm_patch = patch.object(host, "VM", FakeVM)
        self.preflight_patch = patch.object(
            host, "preflight", return_value={"virtual_disk_mb": 1024}
        )
        self.vm_patch.start()
        self.preflight_patch.start()
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), host.make_handler(self.node)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.address = "http://127.0.0.1:" + str(self.server.server_port)
        FakeVM.cleanup_ok = True

    def tearDown(self):
        if isinstance(self.node.active, dict):
            self.node.active["cancel"].set()
        deadline = time.monotonic() + 2
        while self.node.active and time.monotonic() < deadline:
            time.sleep(0.01)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.vm_patch.stop()
        self.preflight_patch.stop()
        self.directory.cleanup()

    def call(self, path, **kwargs):
        return controller.http(self.address, self.key, path, **kwargs)

    def wait_status(self, job_id, statuses):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = self.call("/v1/jobs/" + job_id)
            if job["status"] in statuses and (
                statuses == {"running"} or job["cleanup_confirmed"]
            ):
                return job
            time.sleep(0.01)
        self.fail("job did not reach expected state")

    def submit(self, alias=None):
        return self.call(
            "/v1/jobs",
            method="POST",
            body=allocation(),
            idempotency_key=str(uuid.uuid4()),
            deployment_alias=alias,
        )

    def test_scoped_auth_no_arbitrary_job_fields_and_no_unregistered_deployments(self):
        with self.assertRaisesRegex(common.Blocked, "invalid_node_key"):
            controller.http(self.address, "wrong-key", "/v1/health")
        with self.assertRaisesRegex(common.Blocked, "invalid_request"):
            self.call(
                "/v1/jobs",
                method="POST",
                body={**allocation(), "command": "touch /host"},
                idempotency_key=str(uuid.uuid4()),
            )
        with self.assertRaisesRegex(common.Blocked, "invalid_request"):
            self.submit("unapproved")
        self.assertIsNone(self.node.active)
        with self.assertRaisesRegex(common.Blocked, "no_active_running_guest"):
            self.call("/v1/hardware")

    def test_shared_node_lock_idempotency_alias_and_live_hardware(self):
        job = self.submit("baseline")
        self.wait_status(job["id"], {"running"})
        duplicate = self.call(
            "/v1/jobs",
            method="POST",
            body=allocation(),
            idempotency_key=job["id"],
            deployment_alias="baseline",
        )
        self.assertEqual(duplicate["id"], job["id"])
        with self.assertRaisesRegex(common.Blocked, "invalid_request"):
            self.call(
                "/v1/jobs", method="POST", body=allocation(), idempotency_key=job["id"]
            )
        with self.assertRaisesRegex(common.Blocked, "node_not_ready_or_busy"):
            self.submit()
        hardware = self.call("/v1/hardware")
        self.assertTrue(hardware["fresh"])
        self.assertEqual(hardware["gpu"]["vm_id"], job["id"])
        self.assertEqual(hardware["deployment"]["commit"], "b" * 40)
        self.assertEqual(self.call("/v1/health")["busy"], True)
        self.call("/v1/jobs/" + job["id"] + "/stdin/close", method="POST", body={})
        done = self.wait_status(job["id"], {"completed"})
        self.assertEqual(done["exit_code"], 0)
        self.assertFalse(self.call("/v1/health")["busy"])

    def test_line_delivery_is_idempotent_and_final_partial_line_is_retained(self):
        job = self.submit()
        prefix = "/v1/jobs/" + job["id"]
        self.wait_status(job["id"], {"running"})
        for _ in range(2):
            self.call(
                prefix + "/stdin",
                method="POST",
                body={"id": "one", "line": "actual-line"},
            )
        self.call(prefix + "/stdin/close", method="POST", body={})
        done = self.wait_status(job["id"], {"completed"})
        self.assertTrue(done["cleanup_confirmed"])
        self.assertEqual(
            self.call(prefix + "/stdout?after=0")["lines"],
            [{"sequence": 1, "line": "actual-line"}],
        )
        self.assertEqual(self.call(prefix + "/stdout?after=1")["lines"], [])

    def test_failed_cleanup_retains_gpu_lock_until_operator_recovery(self):
        FakeVM.cleanup_ok = False
        job = self.submit()
        self.wait_status(job["id"], {"running"})
        self.call("/v1/jobs/" + job["id"] + "/cancel", method="POST", body={})
        until = time.monotonic() + 2
        while self.node.active and time.monotonic() < until:
            time.sleep(0.01)
        self.assertTrue(self.node.cleanup_blocked)
        state = self.call("/v1/jobs/" + job["id"])
        self.assertFalse(state["cleanup_confirmed"])
        self.assertEqual(state["error"], "vm_cleanup_unconfirmed")
        with self.assertRaisesRegex(common.Blocked, "node_not_ready_or_busy"):
            self.submit()
        self.assertEqual(self.call("/v1/health")["status"], "unverified")


class GuestRunnerTests(unittest.TestCase):
    def test_real_shell_result_repeat_id_and_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            requests = [
                {
                    "type": "execute",
                    "id": "same",
                    "command": "printf actual-runner-result",
                    "timeout_seconds": 2,
                },
                {
                    "type": "execute",
                    "id": "same",
                    "command": "printf actual-runner-result",
                    "timeout_seconds": 2,
                },
                {
                    "type": "execute",
                    "id": "slow",
                    "command": "sleep 3",
                    "timeout_seconds": 0.1,
                },
            ]
            output = io.StringIO()
            guest_runner.serve(
                io.BytesIO(("\n".join(map(json.dumps, requests)) + "\n").encode()),
                output,
                Path(directory),
            )
            rows = [json.loads(x) for x in output.getvalue().splitlines()]
        self.assertEqual(rows[0]["output"], "actual-runner-result")
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[2]["exit_code"], 124)

    def test_gpu_document_rejects_wrong_physical_device(self):
        config = {"target": target(), "state_dir": "/not-used"}
        vm = host.VM(config, {"id": str(uuid.uuid4())})
        with (
            patch.object(
                vm,
                "execute",
                return_value=subprocess.CompletedProcess(
                    [], 0, b"GPU-other, NVIDIA RTX A5000, 24564, 230, 580.1\n"
                ),
            ),
            self.assertRaisesRegex(common.Blocked, "guest_gpu_identity_mismatch"),
        ):
            vm.hardware()


class UncertainCreateTests(unittest.TestCase):
    def test_lost_create_ack_preserves_disks_and_gpu_reservation_across_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            job = {
                "id": str(uuid.uuid4()),
                "cpus": 1,
                "memory_mb": 4096,
                "timeout_seconds": 30,
                "creation_pending": True,
            }
            vm = host.VM({"target": target(), "state_dir": directory}, job)
            vm.directory.mkdir(parents=True)
            vm.persist()
            (vm.directory / "root.qcow2").write_bytes(b"must-not-delete")
            with patch.object(
                host, "virsh", return_value=subprocess.CompletedProcess([], 0, b"", b"")
            ):
                self.assertFalse(vm.cleanup())
            self.assertTrue((vm.directory / "root.qcow2").exists())
            restored = host.VM(
                vm.config, json.loads((vm.directory / "job.json").read_text())
            )
            self.assertTrue(restored.create_uncertain)
            self.assertFalse(restored.cleanup_confirmed)


class PreflightTests(unittest.TestCase):
    def test_complete_vfio_group_checked_and_no_automatic_driver_detach(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sysfs = root / "sys"
            group = sysfs / "kernel/iommu_groups/9/devices"
            group.mkdir(parents=True)
            driver = sysfs / "bus/pci/drivers/vfio-pci"
            driver.mkdir(parents=True)
            data = target()
            image = root / "approved.qcow2"
            image.write_bytes(b"approved-test-image")
            data["guest_image"] = str(image)
            for index, pci in enumerate(data["gpu_pci_devices"]):
                device = sysfs / "bus/pci/devices" / pci
                device.mkdir(parents=True)
                (device / "iommu_group").symlink_to(group.parent)
                (device / "driver").symlink_to(driver)
                (device / "vendor").write_text("0x10de")
                (device / "class").write_text("0x030000" if index == 0 else "0x040300")
                (group / pci).symlink_to(device)

            def virsh(*args, **kwargs):
                replies = {
                    "net-dumpxml": b"<network><name>gpu-jobs</name></network>",
                    "net-info": b"Active: yes\n",
                    "list": b"",
                }
                return subprocess.CompletedProcess(
                    args, 0, replies.get(args[0], b""), b""
                )

            original_exists = Path.exists
            original_read = Path.read_text

            def exists(path):
                return True if str(path) == "/dev/kvm" else original_exists(path)

            def read(path, *args, **kwargs):
                if str(path) == "/proc/meminfo":
                    return "MemTotal: 33554432 kB\nMemAvailable: 30000000 kB\n"
                return original_read(path, *args, **kwargs)

            with (
                patch.object(Path, "exists", exists),
                patch.object(Path, "read_text", read),
                patch.object(host.shutil, "which", lambda x: "/usr/bin/" + x),
                patch.object(host, "virsh", side_effect=virsh) as controls,
                patch.object(
                    host,
                    "run",
                    return_value=subprocess.CompletedProcess(
                        [],
                        0,
                        json.dumps(
                            {"format": "qcow2", "virtual-size": 1024**3}
                        ).encode(),
                    ),
                ),
                patch.object(host, "sha256_file", return_value="a" * 64),
                patch.object(host.os, "cpu_count", return_value=16),
            ):
                checked = host.preflight({"target": data}, sysfs=sysfs)
                self.assertTrue(checked["iommu"])
                self.assertTrue(checked["vfio"])
                self.assertFalse(
                    any(
                        "nodedev-detach" in call.args
                        for call in controls.call_args_list
                    )
                )
                (group / "0000:02:00.0").symlink_to(root / "other-device")
                with self.assertRaisesRegex(common.Blocked, "unsafe_iommu_group"):
                    host.preflight({"target": data}, sysfs=sysfs)


class AgentLauncherTests(unittest.TestCase):
    def test_codex_uses_supported_structured_flags_and_parent_rechecks_receipt(self):
        import argparse

        with tempfile.TemporaryDirectory() as directory:
            data = target()
            receipt = {
                "node_id": data["id"],
                "status": "ready",
                "key": "rn_very_secret",
                "address": "http://127.0.0.1:18100",
                "target": data,
            }
            common.private_json(
                Path(directory) / data["id"] / "connection.json", receipt
            )
            args = argparse.Namespace(
                state=directory,
                check_only=False,
                install_packages=False,
                agent_timeout_seconds=600,
            )
            with (
                patch.object(
                    controller.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 0),
                ) as launch,
                patch.object(
                    controller, "verify_connection", return_value={}
                ) as verify,
            ):
                result = controller.agent_one(data, args)
            command = launch.call_args.args[0]
            self.assertIn("--ephemeral", command)
            self.assertIn("--output-schema", command)
            self.assertIn("--output-last-message", command)
            self.assertEqual(
                command[command.index("--sandbox") + 1], "danger-full-access"
            )
            self.assertNotIn("rn_very_secret", " ".join(command))
            self.assertNotIn("rn_very_secret", launch.call_args.kwargs["input"])
            verify.assert_called_once_with(receipt)
            self.assertEqual(result, receipt)
            self.assertEqual(
                stat.S_IMODE(
                    (Path(directory) / data["id"] / "agent.log").stat().st_mode
                ),
                0o600,
            )


if __name__ == "__main__":
    unittest.main()
