#!/usr/bin/env python3
"""Private resource-node service: one real KVM/VFIO VM per job, no harness/verifier."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from common import MAX_BODY, MAX_OUTPUT, Blocked, private_json, validate_target


def run(argv, *, timeout=30, input=None, check=True):
    try:
        result = subprocess.run(
            argv, input=input, capture_output=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Blocked(
            "command_unavailable_or_timed_out: " + Path(argv[0]).name
        ) from exc
    if check and result.returncode:
        # Never include arbitrary stdout/stderr in receipts or API errors.
        raise Blocked("command_failed: " + Path(argv[0]).name)
    return result


def virsh(*args, **kwargs):
    return run(["virsh", "-c", "qemu:///system", *args], **kwargs)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def preflight(config, *, sysfs=Path("/sys"), verify_image=True):
    """No device rebinding, reboots, package installs, or guessed GPU assignment."""
    target = validate_target(config["target"])
    if os.name != "posix" or not Path("/dev/kvm").exists():
        raise Blocked(
            "kvm_unavailable: enable CPU virtualization on a Linux bare-metal host"
        )
    for tool in ("virsh", "qemu-img", "cloud-localds", "ssh", "ssh-keygen"):
        if shutil.which(tool) is None:
            raise Blocked("missing_host_tool: " + tool)
    virsh("version")
    network = ET.fromstring(virsh("net-dumpxml", target["libvirt_network"]).stdout)
    if network.find("forward") is not None:
        raise Blocked(
            "guest_network_not_isolated: use an operator-approved libvirt network without forwarding"
        )
    info = virsh("net-info", target["libvirt_network"]).stdout.decode()
    if not re.search(r"Active:\s+yes", info):
        raise Blocked("libvirt_network_inactive")
    devices = set(target["gpu_pci_devices"])
    group_devices = set()
    nvidia_gpu = False
    for address in devices:
        device = sysfs / "bus/pci/devices" / address
        if not device.is_dir() or not (device / "iommu_group").exists():
            raise Blocked("iommu_group_missing: " + address)
        group_devices.update(x.name for x in (device / "iommu_group/devices").iterdir())
        if (device / "driver").resolve().name != "vfio-pci":
            raise Blocked(
                "gpu_not_vfio_bound: operator must bind approved GPU functions before enrollment"
            )
        vendor = (device / "vendor").read_text().strip()
        pci_class = (device / "class").read_text().strip()
        if vendor != "0x10de" or pci_class[:6] not in {
            "0x0300",
            "0x0302",
            "0x0403",
            "0x0c03",
            "0x0c80",
        }:
            raise Blocked("iommu_group_contains_non_gpu_functions")
        nvidia_gpu |= vendor == "0x10de" and pci_class.startswith("0x03")
    if devices != group_devices or not nvidia_gpu:
        raise Blocked(
            "unsafe_iommu_group: list every function of the dedicated NVIDIA GPU group"
        )
    # Refuse assignment already held by any domain, including a stopped persistent domain.
    for name in virsh("list", "--all", "--name").stdout.decode().splitlines():
        if not name:
            continue
        xml = ET.fromstring(virsh("dumpxml", name).stdout)
        for device in xml.findall("./devices/hostdev/source/address"):
            attrs = device.attrib
            if not {"domain", "bus", "slot", "function"} <= attrs.keys():
                continue
            address = f"{int(attrs['domain'], 16):04x}:{int(attrs['bus'], 16):02x}:{int(attrs['slot'], 16):02x}.{int(attrs['function'], 16)}"
            if address in devices:
                raise Blocked("gpu_already_assigned_to_domain")
    image = Path(target["guest_image"])
    if not image.is_file():
        raise Blocked("approved_guest_image_missing")
    image_info = json.loads(
        run(["qemu-img", "info", "--output=json", str(image)]).stdout
    )
    if image_info.get("format") != "qcow2" or image_info.get("backing-filename"):
        raise Blocked("guest_image_must_be_standalone_qcow2")
    if verify_image and sha256_file(image) != target["guest_image_sha256"]:
        raise Blocked("guest_image_digest_mismatch")
    if image_info["virtual-size"] > target["scratch_mb"] * 1024 * 1024:
        raise Blocked("guest_image_exceeds_scratch_virtual_disk_quota")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        memory[parts[0].rstrip(":")] = int(parts[1])
    if memory["MemAvailable"] // 1024 < target["guest_memory_mb"] + 2048:
        raise Blocked(
            "insufficient_available_host_ram: leave at least 2 GiB for the host"
        )
    if target["guest_vcpus"] > (os.cpu_count() or 1):
        raise Blocked("insufficient_host_cpus")
    return {
        "kvm": True,
        "iommu": True,
        "vfio": True,
        "guest_image_sha256": target["guest_image_sha256"],
        "host_cpus": os.cpu_count(),
        "host_memory_mb": memory["MemTotal"] // 1024,
        "gpu_pci_devices": sorted(devices),
        "virtual_disk_mb": math.ceil(image_info["virtual-size"] / 1024**2),
    }


def domain_xml(target, job, directory, mac):
    root = ET.Element("domain", type="kvm")
    ET.SubElement(root, "name").text = "crusoe-" + target["id"] + "-" + job["id"]
    ET.SubElement(root, "uuid").text = job["id"]
    ET.SubElement(root, "memory", unit="MiB").text = str(job["memory_mb"])
    ET.SubElement(root, "vcpu", placement="static").text = str(math.ceil(job["cpus"]))
    tune = ET.SubElement(root, "cputune")
    ET.SubElement(tune, "global_period").text = "100000"
    ET.SubElement(tune, "global_quota").text = str(int(job["cpus"] * 100000))
    osxml = ET.SubElement(root, "os")
    ET.SubElement(osxml, "type", arch="x86_64", machine="q35").text = "hvm"
    ET.SubElement(osxml, "boot", dev="hd")
    features = ET.SubElement(root, "features")
    ET.SubElement(features, "acpi")
    ET.SubElement(features, "apic")
    ET.SubElement(root, "cpu", mode="host-passthrough", check="none")
    ET.SubElement(root, "on_poweroff").text = "destroy"
    ET.SubElement(root, "on_reboot").text = "destroy"
    ET.SubElement(root, "on_crash").text = "destroy"
    devices = ET.SubElement(root, "devices")
    for filename, kind, dev in (
        ("root.qcow2", "disk", "vda"),
        ("seed.img", "disk", "vdb"),
    ):
        disk = ET.SubElement(devices, "disk", type="file", device=kind)
        ET.SubElement(
            disk,
            "driver",
            name="qemu",
            type="qcow2" if filename.endswith("qcow2") else "raw",
        )
        ET.SubElement(disk, "source", file=str(directory / filename))
        ET.SubElement(disk, "target", dev=dev, bus="virtio")
        if filename == "seed.img":
            ET.SubElement(disk, "readonly")
    interface = ET.SubElement(devices, "interface", type="network")
    ET.SubElement(interface, "mac", address=mac)
    ET.SubElement(interface, "source", network=target["libvirt_network"])
    ET.SubElement(interface, "model", type="virtio")
    ET.SubElement(devices, "serial", type="pty")
    ET.SubElement(devices, "console", type="pty")
    ET.SubElement(devices, "memballoon", model="none")
    # managed=no is deliberate: never detach a GPU in use by the host.
    for pci in target["gpu_pci_devices"]:
        domain, bus, slot_function = pci.split(":")
        slot, function = slot_function.split(".")
        hostdev = ET.SubElement(
            devices, "hostdev", mode="subsystem", type="pci", managed="no"
        )
        source = ET.SubElement(hostdev, "source")
        ET.SubElement(
            source,
            "address",
            domain="0x" + domain,
            bus="0x" + bus,
            slot="0x" + slot,
            function="0x" + function,
        )
    return ET.tostring(root, encoding="unicode")


class VM:
    def __init__(self, config, job):
        self.config = config
        self.target = config["target"]
        self.job = job
        self.name = "crusoe-" + self.target["id"] + "-" + job["id"]
        self.directory = Path(config["state_dir"]) / "jobs" / job["id"]
        self.ip = None
        self.cleanup_confirmed = False
        self.create_uncertain = job.get("creation_pending", False)

    def persist(self):
        values = {
            key: self.job[key] for key in ("id", "memory_mb", "cpus", "timeout_seconds")
        }
        values["creation_pending"] = self.create_uncertain
        private_json(self.directory / "job.json", values)

    def ssh_args(self):
        if self.ip is None:
            raise Blocked("guest_not_ready")
        return [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "UserKnownHostsFile=" + str(self.directory / "known_hosts"),
            "-o",
            "HostKeyAlias=" + self.name,
            "-i",
            str(self.directory / "client_key"),
            self.target["guest_user"] + "@" + self.ip,
        ]

    def execute(self, command, timeout=15):
        return run([*self.ssh_args(), shlex.join(command)], timeout=timeout)

    def boot(self):
        self.directory.mkdir(parents=True, mode=0o711)
        os.chmod(
            self.directory, 0o711
        )  # QEMU can traverse only the explicit disk paths.
        self.persist()
        for name in ("client_key", "host_key"):
            run(
                [
                    "ssh-keygen",
                    "-q",
                    "-t",
                    "ed25519",
                    "-N",
                    "",
                    "-f",
                    str(self.directory / name),
                ]
            )
        client_pub = (self.directory / "client_key.pub").read_text().strip()
        host_pub = (self.directory / "host_key.pub").read_text().strip()
        # JSON is valid YAML, after the cloud-config header. No untrusted shell interpolation.
        user_data = {
            "users": [
                {
                    "name": self.target["guest_user"],
                    "lock_passwd": True,
                    "shell": "/bin/bash",
                    "ssh_authorized_keys": [client_pub],
                }
            ],
            "ssh_pwauth": False,
            "disable_root": True,
            "ssh_deletekeys": True,
            "ssh_keys": {
                "ed25519_private": (self.directory / "host_key").read_text(),
                "ed25519_public": host_pub,
            },
        }
        seed_config = self.directory / "user-data"
        seed_config.write_text("#cloud-config\n" + json.dumps(user_data))
        os.chmod(seed_config, 0o600)
        meta = self.directory / "meta-data"
        meta.write_text(
            json.dumps({"instance-id": self.job["id"], "local-hostname": self.name})
        )
        (self.directory / "known_hosts").write_text(self.name + " " + host_pub + "\n")
        run(
            [
                "qemu-img",
                "create",
                "-f",
                "qcow2",
                "-F",
                "qcow2",
                "-b",
                self.target["guest_image"],
                str(self.directory / "root.qcow2"),
            ]
        )
        run(
            [
                "cloud-localds",
                str(self.directory / "seed.img"),
                str(seed_config),
                str(meta),
            ]
        )
        for name in ("root.qcow2", "seed.img"):
            os.chmod(self.directory / name, 0o660)
            # Standard libvirt Ubuntu/Debian owner; operator may configure QEMU differently.
            try:
                shutil.chown(self.directory / name, user="libvirt-qemu", group="kvm")
            except LookupError as exc:
                raise Blocked(
                    "libvirt_qemu_user_missing: supported host is Ubuntu/Debian libvirt"
                ) from exc
        mac = "52:54:00:" + ":".join(
            f"{byte:02x}" for byte in uuid.UUID(self.job["id"]).bytes[-3:]
        )
        xml = self.directory / "domain.xml"
        xml.write_text(domain_xml(self.target, self.job, self.directory, mac))
        # Persist before sending create: a lost RPC acknowledgement must retain the GPU slot.
        self.create_uncertain = True
        self.persist()
        # Transient VM + managed=no PCI + qcow2 backing file, never a container.
        virsh("create", str(xml))
        self.create_uncertain = False
        self.persist()
        deadline = time.monotonic() + min(
            self.target["boot_timeout_seconds"], self.job["timeout_seconds"]
        )
        while time.monotonic() < deadline:
            if self.job.get("cancel") is not None and self.job["cancel"].is_set():
                raise Blocked("job_cancelled")
            addresses = virsh(
                "domifaddr", self.name, "--source", "lease", check=False
            ).stdout.decode()
            for text in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}/\d+", addresses):
                address = ipaddress.ip_interface(text).ip
                if address.version == 4 and address.is_private:
                    self.ip = str(address)
            if self.ip:
                try:
                    self.execute(
                        ["cloud-init", "status", "--wait"],
                        timeout=min(10, max(1, deadline - time.monotonic())),
                    )
                    # Fail enrollment before accepting jobs if the immutable entrypoint is absent.
                    self.execute(["test", "-x", self.target["guest_job_command"][0]])
                    runner_test = json.loads(
                        self.execute(
                            [*self.target["guest_job_command"], "--self-test"]
                        ).stdout
                    )
                    if runner_test != {
                        "ready": True,
                        "protocol": "execution_result-jsonl-v1",
                    }:
                        raise Blocked("guest_job_runner_self_test_failed")
                    return self.hardware()
                except Blocked:
                    pass
            time.sleep(2)
        raise Blocked("guest_boot_or_cloud_init_timeout")

    def hardware(self):
        output = self.execute(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,power.limit,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ).stdout.decode()
        rows = [x.strip().split(", ") for x in output.strip().splitlines()]
        if len(rows) != 1 or len(rows[0]) != 5 or rows[0][0] != self.target["gpu_uuid"]:
            raise Blocked("guest_gpu_identity_mismatch")
        gpu_uuid, name, memory, power, driver = rows[0]
        if "A5000" not in name or int(float(memory)) < 23000:
            raise Blocked("guest_gpu_is_not_A5000")
        return {
            "uuid": gpu_uuid,
            "name": name,
            "memory_total_mb": int(float(memory)),
            "power_limit_w": float(power),
            "driver_version": driver,
            "observed_at": time.time(),
            "source": "active_kvm_guest_nvidia_smi",
            "vm_id": self.job["id"],
        }

    def cleanup(self):
        try:
            # A failed create may have completed after its control command timed out.
            # Confirm absence after an explicit destroy; never delete backing files first.
            virsh("destroy", self.name, check=False)
            names = virsh("list", "--all", "--name").stdout.decode().splitlines()
            if self.name in names or self.create_uncertain:
                # The daemon could finish a timed-out create after this immediate absence check.
                # Preserve the disk/state and quarantine capacity for operator reconciliation.
                return False
            if self.directory.exists():
                shutil.rmtree(self.directory)
            self.cleanup_confirmed = True
            return True
        except (Blocked, OSError):
            return False


def validate_job(target, data, disk_mb=0):
    allowed = {"resource_id", "cpus", "memory_mb", "scratch_mb", "timeout_seconds"}
    if (
        not isinstance(data, dict)
        or set(data) != allowed
        or data.get("resource_id") != target["resource_id"]
    ):
        raise ValueError("unknown_resource_or_job_fields")
    if (
        type(data["cpus"]) not in (int, float)
        or not math.isfinite(data["cpus"])
        or not 0.01 <= data["cpus"] <= target["guest_vcpus"]
    ):
        raise ValueError("invalid_cpus")
    for field, low, high in (
        ("memory_mb", 1024, target["guest_memory_mb"]),
        ("scratch_mb", max(16, disk_mb), target["scratch_mb"]),
        ("timeout_seconds", 1, target["max_timeout_seconds"]),
    ):
        if type(data[field]) is not int or not low <= data[field] <= high:
            raise ValueError("invalid_" + field)
    return dict(data)


class Node:
    def __init__(self, config):
        self.config = config
        self.target = validate_target(config["target"])
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None
        self.probe_result = None
        self.disk_mb = 0
        self.cleanup_blocked = False

    def startup(self):
        # Recover only VMs created by this node, recorded under its private state dir.
        jobs = Path(self.config["state_dir"]) / "jobs"
        if jobs.exists():
            for directory in jobs.iterdir():
                try:
                    job = json.loads((directory / "job.json").read_text())
                    if str(uuid.UUID(job["id"])) != directory.name:
                        raise ValueError
                    vm = VM(self.config, job)
                    if not vm.cleanup():
                        raise Blocked("orphan_vm_cleanup_unconfirmed")
                except (OSError, ValueError, KeyError) as exc:
                    raise Blocked(
                        "invalid_orphan_job_state_requires_operator_review"
                    ) from exc
        result = preflight(self.config)
        self.disk_mb = result["virtual_disk_mb"]

    def probe(self):
        with self.lock:
            if self.active or self.cleanup_blocked:
                raise Blocked("node_busy_or_cleanup_required")
            self.active = "probe"
            self.probe_result = None
        job = {
            "id": str(uuid.uuid4()),
            "memory_mb": self.target["guest_memory_mb"],
            "cpus": self.target["guest_vcpus"],
            "timeout_seconds": self.target["boot_timeout_seconds"],
        }
        vm = VM(self.config, job)
        try:
            preflight(self.config)
            gpu = vm.boot()
            result = {
                "gpu": gpu,
                "vm_isolation": "libvirt-kvm-vfio",
                "boot_verified": True,
                "runner_verified": True,
                "verified_at": time.time(),
                "image_sha256": self.target["guest_image_sha256"],
            }
        finally:
            cleaned = vm.cleanup()
            with self.lock:
                self.active = None
                self.cleanup_blocked = not cleaned
        if not cleaned:
            raise Blocked("probe_vm_cleanup_unconfirmed")
        self.probe_result = result
        return result

    def submit(self, data, request_id=None, deployment_alias=None):
        if request_id is None:
            raise ValueError("Idempotency-Key UUID is required")
        request_id = str(uuid.UUID(request_id))
        request = validate_job(self.target, data, self.disk_mb)
        if deployment_alias is not None and deployment_alias not in self.target.get(
            "deployments", {}
        ):
            raise ValueError("unknown_approved_deployment")
        request["deployment_alias"] = deployment_alias
        with self.lock:
            if request_id in self.jobs:
                existing = self.jobs[request_id]
                if any(existing[key] != value for key, value in request.items()):
                    raise ValueError("job_idempotency_conflict")
                return self.public_job(existing)
            if self.active or self.cleanup_blocked or not self.probe_result:
                raise Blocked("node_not_ready_or_busy")
            if len(self.jobs) >= 100:
                # Completed receipts are bounded. Live client should retain results locally.
                for key, value in list(self.jobs.items()):
                    if value["status"] in {"completed", "failed", "cancelled"}:
                        del self.jobs[key]
                        break
            job = {
                **request,
                "id": request_id,
                "status": "provisioning",
                "exit_code": None,
                "error": None,
                "cleanup_confirmed": False,
                "created_at": time.time(),
                "stdin": [],
                "messages": {},
                "stdin_closed": False,
                "stdout": [],
                "bytes": 0,
                "stdin_bytes": 0,
                "cancel": threading.Event(),
                "vm": None,
            }
            self.jobs[job["id"]] = job
            self.active = job
            threading.Thread(target=self.work, args=(job,), daemon=True).start()
            return self.public_job(job)

    @staticmethod
    def public_job(job):
        return {
            key: job.get(key)
            for key in (
                "id",
                "resource_id",
                "status",
                "exit_code",
                "error",
                "cleanup_confirmed",
                "created_at",
                "deployment_alias",
            )
        }

    def work(self, job):
        vm = VM(self.config, job)
        job["vm"] = vm
        process = None
        deadline = time.monotonic() + job["timeout_seconds"]
        try:
            preflight(self.config)
            vm.boot()
            if job["cancel"].is_set():
                raise Blocked("job_cancelled")
            command = self.target["guest_job_command"]
            if job.get("deployment_alias") is not None:
                command = self.target["deployments"][job["deployment_alias"]]["command"]
            process = subprocess.Popen(
                [*vm.ssh_args(), shlex.join(command)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            job["status"] = "running"
            self.relay(process, job, deadline)
            job["exit_code"] = process.wait(
                timeout=max(0.01, deadline - time.monotonic())
            )
            job["status"] = "completed" if job["exit_code"] == 0 else "failed"
            if job["exit_code"]:
                job["error"] = "guest_job_exited_nonzero"
        except (Blocked, ValueError, subprocess.TimeoutExpired) as exc:
            job["status"] = "cancelled" if job["cancel"].is_set() else "failed"
            job["error"] = str(exc) if isinstance(exc, Blocked) else "guest_job_failed"
        except Exception:  # noqa: BLE001 - failures must still destroy the VM.
            job["status"], job["error"] = "failed", "guest_job_execution_failed"
        finally:
            if process is not None:
                try:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass  # VM teardown below remains authoritative even if its SSH client is stuck.
                finally:
                    for pipe in (process.stdin, process.stdout, process.stderr):
                        if not pipe.closed:
                            pipe.close()
            cleaned = vm.cleanup()
            with self.lock:
                job["cleanup_confirmed"] = cleaned
                self.cleanup_blocked = not cleaned
                if not cleaned:
                    job["status"], job["error"] = "failed", "vm_cleanup_unconfirmed"
                self.active = None

    def relay(self, process, job, deadline):
        buffer = bytearray()
        outgoing = memoryview(b"")
        stderr_bytes = 0
        with selectors.DefaultSelector() as selector:
            for stream in (process.stdin, process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ)
            selector.register(process.stderr, selectors.EVENT_READ)
            while True:
                if time.monotonic() >= deadline or job["cancel"].is_set():
                    raise Blocked(
                        "job_cancelled" if job["cancel"].is_set() else "job_timeout"
                    )
                if not process.stdin.closed and not outgoing:
                    with self.lock:
                        if job["stdin"]:
                            outgoing = memoryview((job["stdin"].pop(0) + "\n").encode())
                            selector.register(process.stdin, selectors.EVENT_WRITE)
                        elif job["stdin_closed"]:
                            process.stdin.close()
                for key, _ in selector.select(0.05):
                    if key.fileobj is process.stdin:
                        try:
                            outgoing = outgoing[
                                os.write(process.stdin.fileno(), outgoing[:65536]) :
                            ]
                        except BrokenPipeError:
                            outgoing = memoryview(b"")
                            selector.unregister(process.stdin)
                            process.stdin.close()
                            continue
                        except BlockingIOError:
                            continue
                        if not outgoing:
                            selector.unregister(process.stdin)
                        continue
                    try:
                        data = os.read(key.fileobj.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    if key.fileobj is process.stderr:
                        stderr_bytes += len(data)
                        if stderr_bytes > MAX_OUTPUT:
                            raise Blocked("stderr_output_limit")
                        continue
                    job["bytes"] += len(data)
                    buffer.extend(data)
                    if job["bytes"] > MAX_OUTPUT:
                        raise Blocked("stdout_output_limit")
                    lines = buffer.split(b"\n")
                    buffer = lines.pop()
                    for line in lines:
                        self.append_line(job, line)
                    if len(buffer) > MAX_BODY:
                        raise Blocked("stdout_line_limit")
                if process.poll() is not None and not any(
                    x.fileobj is not process.stdin for x in selector.get_map().values()
                ):
                    if buffer:
                        self.append_line(job, buffer)
                    return

    def append_line(self, job, line):
        if len(line) > MAX_BODY:
            raise Blocked("stdout_line_limit")
        try:
            text = line.decode("utf-8")
        except UnicodeError as exc:
            raise Blocked("stdout_not_utf8") from exc
        with self.lock:
            job["stdout"].append({"sequence": len(job["stdout"]) + 1, "line": text})

    def hardware(self):
        with self.lock:
            job = self.active
            if not isinstance(job, dict) or job["status"] != "running":
                raise Blocked("no_active_running_guest")
            return job["vm"].hardware()


def make_handler(node):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass  # HTTP logs must never capture authorization headers or opaque job I/O.

        def send_json(self, value, status=200):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def body(self):
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= MAX_BODY:
                raise ValueError("request_body_limit")
            value = json.loads(self.rfile.read(size) or b"{}")
            return value

        def do_GET(self):
            self.handle_request()

        def do_POST(self):
            self.handle_request()

        def handle_request(self):
            supplied = self.headers.get("Authorization", "")
            if not supplied.startswith("Bearer ") or not hmac.compare_digest(
                hashlib.sha256(supplied[7:].encode()).hexdigest(),
                node.config["key_sha256"],
            ):
                self.send_json({"error": "invalid_node_key"}, 401)
                self.close_connection = True
                return
            try:
                url = urllib.parse.urlsplit(self.path)
                path = url.path
                if path in {"/v1/models", "/v1/completions", "/v1/chat/completions"}:
                    return self.proxy(path)
                if self.command == "GET" and path == "/v1/health":
                    return self.send_json(
                        {
                            "node_id": node.target["id"],
                            "status": "ready"
                            if node.probe_result and not node.cleanup_blocked
                            else "unverified",
                            "busy": node.active is not None,
                            "verification": node.probe_result,
                            "limits": {
                                "cpus": node.target["guest_vcpus"],
                                "memory_mb": node.target["guest_memory_mb"],
                                "scratch_mb": node.target["scratch_mb"],
                                "max_timeout_seconds": node.target[
                                    "max_timeout_seconds"
                                ],
                            },
                            "runtime": "libvirt-kvm-vfio",
                        }
                    )
                if self.command == "POST" and path == "/v1/probe":
                    self.body()
                    return self.send_json(node.probe())
                if self.command == "GET" and path == "/v1/hardware":
                    gpu = node.hardware()
                    with node.lock:
                        alias = (
                            node.active.get("deployment_alias")
                            if isinstance(node.active, dict)
                            else None
                        )
                        deployment = node.target.get("deployments", {}).get(alias)
                    return self.send_json(
                        {
                            "fresh": True,
                            "gpu": gpu,
                            "image_sha256": node.target["guest_image_sha256"],
                            "deployment": deployment["metadata"]
                            if deployment
                            else None,
                        }
                    )
                if self.command == "POST" and path == "/v1/jobs":
                    return self.send_json(
                        node.submit(
                            self.body(),
                            self.headers.get("Idempotency-Key"),
                            self.headers.get("X-Deployment-Alias"),
                        ),
                        202,
                    )
                match = re.fullmatch(
                    r"/v1/jobs/([a-f0-9-]{36})(?:/(stdin/close|stdin|stdout|cancel))?",
                    path,
                )
                if match is None:
                    return self.send_json({"error": "not_found"}, 404)
                job_id, action = match.groups()
                with node.lock:
                    job = node.jobs.get(job_id)
                    if job is None:
                        return self.send_json({"error": "unknown_job"}, 404)
                    if self.command == "GET" and action is None:
                        return self.send_json(node.public_job(job))
                    if self.command == "GET" and action == "stdout":
                        after = int(
                            urllib.parse.parse_qs(url.query).get("after", ["0"])[0]
                        )
                        if after < 0:
                            raise ValueError("invalid_output_cursor")
                        # Bounded pages keep control-plane replies under the body cap.
                        rows, size = [], 0
                        for row in job["stdout"][after:]:
                            size += len(row["line"].encode()) + 100
                            if rows and size > MAX_BODY:
                                break
                            rows.append(row)
                        return self.send_json(
                            {
                                "lines": rows,
                                "next_after": rows[-1]["sequence"] if rows else after,
                                "closed": job["status"]
                                in {"completed", "failed", "cancelled"},
                            }
                        )
                    if self.command != "POST":
                        return self.send_json({"error": "method_not_allowed"}, 405)
                    data = self.body()
                    if action == "cancel":
                        job["cancel"].set()
                        return self.send_json(node.public_job(job), 202)
                    if action == "stdin/close":
                        job["stdin_closed"] = True
                        return self.send_json({"closed": True}, 202)
                    if action == "stdin":
                        if not isinstance(data, dict) or set(data) != {"id", "line"}:
                            raise ValueError("invalid_stdin_fields")
                        mid, line = data["id"], data["line"]
                        if (
                            not isinstance(mid, str)
                            or not 1 <= len(mid) <= 200
                            or not isinstance(line, str)
                            or "\n" in line
                            or len(line.encode()) > MAX_BODY
                        ):
                            raise ValueError("invalid_stdin_message")
                        if mid in job["messages"]:
                            if job["messages"][mid] != line:
                                raise ValueError("stdin_idempotency_conflict")
                            return self.send_json({"accepted": True}, 202)
                        if job["stdin_closed"] or job["status"] not in {
                            "provisioning",
                            "running",
                        }:
                            raise Blocked("stdin_closed")
                        if job["stdin_bytes"] + len(line.encode()) > MAX_OUTPUT:
                            raise Blocked("stdin_total_limit")
                        job["stdin_bytes"] += len(line.encode())
                        job["messages"][mid] = line
                        job["stdin"].append(line)
                        return self.send_json({"accepted": True}, 202)
                self.send_json({"error": "not_found"}, 404)
            except Blocked as exc:
                self.send_json({"error": str(exc)}, 409)
            except (ValueError, TypeError, KeyError, UnicodeError):
                self.send_json({"error": "invalid_request"}, 422)
            except Exception:  # noqa: BLE001 - never expose host diagnostics or secrets.
                self.send_json({"error": "node_request_failed"}, 503)

        def proxy(self, path):
            if (self.command, path) not in {
                ("GET", "/v1/models"),
                ("POST", "/v1/completions"),
                ("POST", "/v1/chat/completions"),
            }:
                return self.send_json({"error": "method_not_allowed"}, 405)
            with node.lock:
                job = node.active
                if not isinstance(job, dict) or job["status"] != "running":
                    raise Blocked("no_active_running_guest")
                guest_ip = job["vm"].ip
            body = self.body() if self.command == "POST" else None
            if body is not None and body.get("stream"):
                raise ValueError("streaming_not_supported_by_bounded_node_proxy")
            request = urllib.request.Request(
                f"http://{guest_ip}:{node.target['inference_port']}{path}",
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type": "application/json"},
                method=self.command,
            )
            try:
                with urllib.request.urlopen(request, timeout=node.target["max_timeout_seconds"]) as response:
                    payload = response.read(MAX_OUTPUT + 1)
                if len(payload) > MAX_OUTPUT:
                    raise Blocked("inference_reply_limit")
                self.send_json(json.loads(payload))
            except (urllib.error.URLError, TimeoutError):
                raise Blocked("guest_inference_not_ready_or_failed") from None

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "serve"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    try:
        if args.command == "check":
            print(
                json.dumps(
                    {"status": "prerequisites_ready", "checks": preflight(config)}
                )
            )
        else:
            import fcntl

            state = Path(config["state_dir"])
            state.mkdir(parents=True, exist_ok=True, mode=0o711)
            os.chmod(state, 0o711)
            lock = (state / "node.lock").open("w")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            node = Node(config)
            node.startup()
            server = ThreadingHTTPServer(
                ("127.0.0.1", config["target"]["remote_port"]), make_handler(node)
            )
            signal.signal(
                signal.SIGTERM,
                lambda *_: threading.Thread(
                    target=server.shutdown, daemon=True
                ).start(),
            )
            try:
                server.serve_forever()
            finally:
                if isinstance(node.active, dict):
                    node.active["cancel"].set()
                    vm = node.active.get("vm")
                    if vm is not None:
                        vm.cleanup()
    except (Blocked, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
