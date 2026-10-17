import json
import sys

import pytest

from verification_api import setup_node
from verification_api.main import app


def options(tmp_path, *extra):
    return setup_node.parser().parse_args(
        [
            "--server",
            "http://testserver",
            "--state-dir",
            str(tmp_path / "node"),
            "--runner-python",
            sys.executable,
            "--min-free-disk-gb",
            "0",
            "--min-ram-gb",
            "0",
            "--enroll-only",
            *extra,
        ]
    )


def test_real_machine_checks_enroll_and_preserve_id(client, tmp_path):
    args = options(tmp_path, "--label", "pool=development")
    setup_node.enroll(args, client)
    config_path = args.state_dir / "node.json"
    saved = json.loads(config_path.read_text())
    node = client.get(f"/nodes/{saved['id']}").json()
    assert node["config"] == saved["config"]
    assert node["config"]["labels"]["pool"] == "development"
    assert node["config"]["software"]["python_version"].startswith("3.")
    setup_node.enroll(args, client)
    assert json.loads(config_path.read_text())["id"] == saved["id"]
    report = json.loads((args.state_dir / "preflight.json").read_text())
    assert report["registered"] is True
    assert all(check["status"] == "passed" for check in report["checks"])


def test_check_only_does_not_register(client, tmp_path):
    args = options(tmp_path)
    args.enroll_only = False
    args.check_only = True
    setup_node.enroll(args, client)
    assert client.get("/nodes").json()["nodes"] == []
    assert not (args.state_dir / "node.json").exists()
    assert json.loads((args.state_dir / "preflight.json").read_text())["registered"] is False


def test_failed_preflight_does_not_register(client, tmp_path, monkeypatch):
    def failed_git(cache):
        raise setup_node.SetupError("Git checkout failed")

    monkeypatch.setattr(setup_node, "git_probe", failed_git)
    args = options(tmp_path)
    with pytest.raises(setup_node.SetupError, match="not registered"):
        setup_node.enroll(args, client)
    assert client.get("/nodes").json()["nodes"] == []
    report = json.loads((args.state_dir / "preflight.json").read_text())
    assert any(
        check["name"] == "git-checkout" and check["status"] == "failed"
        for check in report["checks"]
    )


def test_required_gpu_failure_blocks_registration(client, tmp_path, monkeypatch):
    def failed_gpu(python):
        raise setup_node.SetupError("GPU compute failed")

    monkeypatch.setattr(setup_node, "gpu_probe", failed_gpu)
    with pytest.raises(setup_node.SetupError):
        setup_node.enroll(options(tmp_path, "--require-gpu"), client)
    assert client.get("/nodes").json()["nodes"] == []


def test_server_auth_failure_is_reported_without_secret(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app.state, "api_key", "private-server-key")
    args = options(tmp_path)
    with pytest.raises(setup_node.SetupError, match="HTTP 401"):
        setup_node.enroll(args, client)
    assert "private-server-key" not in (args.state_dir / "preflight.json").read_text()
    assert not (args.state_dir / "node.json").exists()


def test_another_machine_cannot_reuse_node_id(client, tmp_path, monkeypatch):
    first = options(tmp_path, "--node-id", "same-node")
    monkeypatch.setattr(setup_node, "machine_id", lambda: "machine-a")
    setup_node.enroll(first, client)
    monkeypatch.setattr(setup_node, "machine_id", lambda: "machine-b")
    second = options(tmp_path / "other", "--node-id", "same-node")
    with pytest.raises(setup_node.SetupError, match="another machine"):
        setup_node.enroll(second, client)
    assert client.get("/nodes/same-node").json()["config"]["machine_id"] == "machine-a"


def test_state_lock_prevents_duplicate_node_processes(tmp_path):
    state = tmp_path / "node"
    with setup_node.state_lock(state):
        with pytest.raises(setup_node.SetupError, match="already owns"):
            with setup_node.state_lock(state):
                pytest.fail("Duplicate process obtained the same state lock")


def test_gpu_inventory_and_compute_response_are_combined(monkeypatch):
    monkeypatch.setattr(setup_node.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup_node.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def command(argv, timeout=20):
        if "--format=csv,noheader,nounits" in argv:
            return "GPU-fixture, NVIDIA fixture, 81920, 580.0"
        return json.dumps(
            {
                "driver_cuda_version": 13000,
                "gpus": [
                    {
                        "name": "NVIDIA fixture",
                        "compute_test": "passed",
                        "memory_bytes": 80 * setup_node.GIB,
                    },
                ],
            }
        )

    monkeypatch.setattr(setup_node, "command", command)
    detected = setup_node.gpu_probe(sys.executable)
    assert len(detected["gpus"]) == 1
    assert detected["nvidia_smi_inventory"][0]["driver_version"] == "580.0"
