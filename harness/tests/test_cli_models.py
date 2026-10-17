import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_harness import cli_models
from agent_harness.cli_models import ClaudeCodeModel, NemoClawModel
from agent_harness.clients import MAX_REQUEST_BYTES, ServiceError

KEY = "dummy-native-cli-profile-key"
MODEL = "vendor/test-model"
MESSAGES = [{"role": "user", "content": "Investigate the supplied benchmark"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
PLAN = {
    "content": None,
    "tool_calls": [
        {"id": "one", "name": "search_library", "arguments": '{"query":"optimization"}'}
    ],
}


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_NEMOCLAW_STATE_DIR", str(tmp_path / "nemo-state"))
    for name in (
        "HARNESS_CLAUDE_CODE_BIN",
        "HARNESS_NEMOCLAW_BIN",
        "HARNESS_NEMOCLAW_ENABLED",
        "HARNESS_CLAUDE_API_KEY",
        "ANTHROPIC_API_KEY",
        "HARNESS_NEMOCLAW_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class ClaudeRunner:
    def __init__(self, plan=None):
        self.plan = PLAN if plan is None else plan
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        return json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": self.plan,
            }
        )


class NemoRunner:
    def __init__(self, plan=None, denied=True, fail=None):
        self.plan = PLAN if plan is None else plan
        self.denied, self.fail = denied, fail
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.fail and self.fail in argv:
            raise ServiceError("cli_runtime_failed")
        if "config" in argv:
            return json.dumps([{"id": "main", "tools": {"deny": ["*"] if self.denied else []}}])
        if "exec" in argv:
            return json.dumps(
                {
                    "status": "ok",
                    "result": {
                        "payloads": [
                            {"text": json.dumps(self.plan)},
                        ]
                    },
                }
            )
        return "{}"


def test_claude_native_planner_without_ambient_tools_or_credentials(monkeypatch):
    monkeypatch.setenv("OTHER_SECRET", "do-not-inherit")
    monkeypatch.setenv("HTTPS_PROXY", "https://do-not-inherit.test")
    runner = ClaudeRunner()
    with ClaudeCodeModel(MODEL, "https://anthropic-compatible.test", KEY, runner=runner) as model:
        workdir = model.workdir
        result = model.complete(MESSAGES, TOOLS)
        argv, options = runner.calls[0]
        assert "--print" in argv and "--bare" in argv
        assert argv[argv.index("--tools") + 1] == ""
        assert "--strict-mcp-config" in argv
        assert json.loads(argv[argv.index("--mcp-config") + 1]) == {"mcpServers": {}}
        assert argv[argv.index("--setting-sources") + 1] == ""
        assert json.loads(argv[argv.index("--settings") + 1])["disableAllHooks"]
        assert "--disable-slash-commands" in argv and "--no-session-persistence" in argv
        assert argv[argv.index("--model") + 1] == MODEL
        assert KEY not in json.dumps(argv)
        assert "Investigate the supplied benchmark" not in json.dumps(argv)
        assert json.loads(options["input"])["messages"] == MESSAGES
        assert options["env"]["ANTHROPIC_API_KEY"] == KEY
        assert options["env"]["ANTHROPIC_BASE_URL"] == "https://anthropic-compatible.test"
        assert "OTHER_SECRET" not in options["env"] and "HTTPS_PROXY" not in options["env"]
        assert str(workdir) in options["env"]["HOME"]
        assert result["tool_calls"][0]["function"]["name"] == "search_library"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"query":"optimization"}'
        assert KEY not in json.dumps(result)
    assert not workdir.exists()


def test_cli_tool_ids_are_unique_across_native_turns():
    with ClaudeCodeModel(MODEL, api_key=KEY, runner=ClaudeRunner()) as model:
        first = model.complete(MESSAGES, TOOLS)["tool_calls"][0]["id"]
        second = model.complete(MESSAGES, TOOLS)["tool_calls"][0]["id"]
        assert first != second


@pytest.mark.parametrize(
    "plan",
    [
        {"content": None, "tool_calls": []},
        {"content": None, "tool_calls": [{"id": "one", "name": "Bash", "arguments": "{}"}]},
        {
            "content": None,
            "tool_calls": [{"id": "one", "name": "search_library", "arguments": "[]"}],
        },
        {"content": KEY, "tool_calls": []},
        {"content": "Finished", "tool_calls": [], "unknown": True},
    ],
)
def test_native_plans_reject_unknown_tools_malformed_arguments_and_key_leaks(plan):
    with ClaudeCodeModel(MODEL, api_key=KEY, runner=ClaudeRunner(plan)) as model:
        with pytest.raises(ServiceError, match="invalid_model_response"):
            model.complete(MESSAGES, TOOLS)


def test_complete_json_preserves_independent_reviewer_contract():
    runner = ClaudeRunner(
        {"content": '{"approved":true,"reason":"Evidence holds"}', "tool_calls": []}
    )
    with ClaudeCodeModel(MODEL, api_key=KEY, runner=runner) as model:
        assert model.complete_json("Review the evidence", {"observations": [1, 2]}) == {
            "approved": True,
            "reason": "Evidence holds",
        }
        prompt = json.loads(runner.calls[0][1]["input"])
        assert prompt["tools"] == []
        assert json.loads(prompt["messages"][1]["content"]) == {"observations": [1, 2]}


def test_cli_configuration_rejects_unsafe_endpoints_and_model_ids():
    for changes in (
        {"base_url": "http://public.test"},
        {"base_url": "https://user:pass@example.test"},
        {"base_url": "https://example.test?key=x"},
        {"model": "--dangerous"},
        {"api_key": "bad\nkey"},
    ):
        with pytest.raises(ServiceError):
            ClaudeCodeModel(**{"model": MODEL, "api_key": KEY, **changes}, runner=ClaudeRunner())


def test_unavailable_claude_binary_does_not_use_generic_http(monkeypatch):
    monkeypatch.setattr(cli_models, "_binary", lambda *_: None)
    assert not cli_models.claude_code_status()["available"]
    with pytest.raises(ServiceError, match="claude_code_unavailable"):
        ClaudeCodeModel(MODEL, api_key=KEY)


def test_nemoclaw_creates_independent_managed_sandbox_and_destroys_it(monkeypatch):
    monkeypatch.setenv("UNRELATED_KEY", "not-for-nemo")
    runner = NemoRunner()
    with NemoClawModel(
        MODEL, "https://api.inference.crusoecloud.com/v1", KEY, runner=runner
    ) as model:
        workdir, name = model.workdir, model.name
        result = model.complete(MESSAGES, TOOLS)
        model.complete(MESSAGES, TOOLS)
        onboard = [call for call in runner.calls if "onboard" in call[0]]
        assert len(onboard) == 1
        argv, options = onboard[0]
        assert argv[argv.index("--agent") + 1] == "openclaw"
        assert argv[argv.index("--name") + 1] == name
        assert "--non-interactive" in argv and "--fresh" in argv
        assert options["env"]["NEMOCLAW_PROVIDER"] == "custom"
        assert options["env"]["NEMOCLAW_MODEL"] == MODEL
        assert options["env"]["COMPATIBLE_API_KEY"] == KEY
        assert "UNRELATED_KEY" not in options["env"]
        assert "COMPATIBLE_API_KEY" not in model.env
        manifest = json.loads(Path(argv[argv.index("--agents") + 1]).read_text())
        assert manifest["main"]["tools"]["deny"] == ["*"]
        assert any("--probe-only" in call[0] for call in runner.calls)
        assert any("config" in call[0] for call in runner.calls)
        execute = next(call for call in runner.calls if "exec" in call[0])
        assert "--stdin" in execute[0] and "--no-tty" in execute[0]
        assert execute[0][-3:] == ["python3", "-c", cli_models._NEMO_TURN]
        assert KEY not in json.dumps(execute[0])
        assert "Investigate the supplied benchmark" not in json.dumps(execute[0])
        assert "COMPATIBLE_API_KEY" not in execute[1]["env"]
        assert json.loads(json.loads(execute[1]["input"])["prompt"])["messages"] == MESSAGES
        assert result["tool_calls"][0]["function"]["name"] == "search_library"
    assert runner.calls[-1][0][-3:] == ["destroy", "--yes", "--cleanup-gateway"]
    assert not workdir.exists()


@pytest.mark.parametrize(
    "endpoint,model_id,api_key",
    [
        ("https://api.openai.com/v1", "example-openai-model", "sk-dummy-openai-api-key"),
        (
            "https://api.inference.crusoecloud.com/v1",
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B",
            "dummy-crusoe-inference-key",
        ),
    ],
    ids=["openai", "crusoe-nemotron"],
)
def test_nemoclaw_forwards_compatible_provider_endpoint_model_and_key(endpoint, model_id, api_key):
    runner = NemoRunner()
    with NemoClawModel(model_id, endpoint, api_key, runner=runner) as model:
        result = model.complete(MESSAGES, TOOLS)
        onboard = [call for call in runner.calls if "onboard" in call[0]]
        assert len(onboard) == 1
        argv, options = onboard[0]
        assert options["env"]["NEMOCLAW_PROVIDER"] == "custom"
        assert options["env"]["NEMOCLAW_ENDPOINT_URL"] == endpoint
        assert options["env"]["NEMOCLAW_MODEL"] == model_id
        assert options["env"]["NEMOCLAW_PREFERRED_API"] == "openai-completions"
        assert options["env"]["COMPATIBLE_API_KEY"] == api_key
        assert api_key not in json.dumps(argv)
        assert "COMPATIBLE_API_KEY" not in model.env
        assert api_key not in json.dumps(result)
        for arguments, settings in runner.calls:
            if "onboard" not in arguments:
                assert "COMPATIBLE_API_KEY" not in settings["env"]
    assert "destroy" in runner.calls[-1][0]


def test_nemoclaw_never_reuses_another_runs_sandbox_or_inference_gateway():
    left = NemoClawModel(MODEL, api_key=KEY, runner=NemoRunner())
    right = NemoClawModel(MODEL, api_key=KEY, runner=NemoRunner())
    try:
        assert left.name != right.name and left.workdir != right.workdir
        assert left.env["HOME"] != right.env["HOME"]
        assert left.env["NEMOCLAW_GATEWAY_PORT"] != right.env["NEMOCLAW_GATEWAY_PORT"]
    finally:
        left.close()
        right.close()


def test_nemoclaw_refuses_planning_if_tool_policy_is_not_verified():
    runner = NemoRunner(denied=False)
    with NemoClawModel(MODEL, api_key=KEY, runner=runner) as model:
        with pytest.raises(ServiceError, match="nemoclaw_tool_policy_unverified"):
            model.complete(MESSAGES, TOOLS)
        assert not any("exec" in call[0] for call in runner.calls)
    assert "destroy" in runner.calls[-1][0]


def test_nemoclaw_keeps_recovery_authority_when_destroy_cannot_be_confirmed():
    runner = NemoRunner(fail="destroy")
    model = NemoClawModel(MODEL, api_key=KEY, runner=runner)
    model.complete(MESSAGES, TOOLS)
    with pytest.raises(ServiceError, match="nemoclaw_cleanup_required"):
        model.close()
    assert model.workdir.exists()
    runner.fail = None
    model.close()
    assert not model.workdir.exists()


def test_nemoclaw_failed_onboarding_does_not_retry_or_use_a_host_runner():
    runner = NemoRunner(fail="onboard")
    with NemoClawModel(MODEL, api_key=KEY, runner=runner) as model:
        with pytest.raises(ServiceError, match="cli_runtime_failed"):
            model.complete(MESSAGES, TOOLS)
        with pytest.raises(ServiceError, match="nemoclaw_creation_uncertain"):
            model.complete(MESSAGES, TOOLS)
    assert sum("onboard" in call[0] for call in runner.calls) == 1
    assert not any("exec" in call[0] for call in runner.calls)


def test_nemoclaw_status_reports_missing_runtime_opt_in_and_docker(monkeypatch):
    monkeypatch.setattr(cli_models, "_binary", lambda *_: None)
    assert not cli_models.nemoclaw_status()["available"]
    monkeypatch.setattr(cli_models, "_binary", lambda *_: "/usr/bin/nemoclaw")
    assert not cli_models.nemoclaw_status()["available"]
    monkeypatch.setenv("HARNESS_NEMOCLAW_ENABLED", "1")
    monkeypatch.setattr(cli_models.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        cli_models.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            1,
        ),
    )
    assert not cli_models.nemoclaw_status()["available"]
    monkeypatch.setattr(
        cli_models.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
        ),
    )
    assert cli_models.nemoclaw_status()["available"]


def test_nemoclaw_missing_runtime_never_falls_back_to_openclaw(monkeypatch):
    monkeypatch.setattr(cli_models, "nemoclaw_status", lambda: {"available": False})
    with pytest.raises(ServiceError, match="nemoclaw_unavailable"):
        NemoClawModel(MODEL, api_key=KEY)


def test_native_process_broker_bounds_output_and_timeout_without_a_shell(tmp_path):
    planner = cli_models._CLIPlanner()
    planner._initialize(None)
    planner.workdir = tmp_path
    planner.env = {"PATH": os.defpath}
    assert planner._execute([sys.executable, "-c", "print('ok')"]) == "ok\n"
    with pytest.raises(ServiceError, match="response_too_large"):
        planner._execute([sys.executable, "-c", f"print('x'*{MAX_REQUEST_BYTES + 1})"])
    with pytest.raises(ServiceError, match="timeout"):
        planner._execute([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05)
    planner._stop()
    with pytest.raises(ServiceError, match="runtime_closed"):
        planner._execute([sys.executable, "-c", "print('must not run')"])


def test_environment_auth_profiles_honor_explicit_endpoint_variables(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic-env.test")
    with ClaudeCodeModel(MODEL, runner=ClaudeRunner()) as model:
        assert model.base_url == "https://anthropic-env.test"
    monkeypatch.setenv("HARNESS_CLAUDE_BASE_URL", "https://harness-claude.test")
    with ClaudeCodeModel(MODEL, runner=ClaudeRunner()) as model:
        assert model.base_url == "https://harness-claude.test"
    monkeypatch.setenv("HARNESS_NEMOCLAW_API_KEY", KEY)
    monkeypatch.setenv("HARNESS_NEMOCLAW_BASE_URL", "https://nemo-env.test/v1")
    with NemoClawModel(MODEL, runner=NemoRunner()) as model:
        assert model.base_url == "https://nemo-env.test/v1"


def test_runtime_reference_is_available_before_first_turn_and_contains_no_credentials():
    runner = NemoRunner()
    model = NemoClawModel(MODEL, api_key=KEY, runner=runner)
    reference = model.cleanup_reference()
    assert reference == {
        "kind": "nemoclaw",
        "workdir": str(model.workdir),
        "name": model.name,
        "gateway_port": model.env["NEMOCLAW_GATEWAY_PORT"],
    }
    assert KEY not in json.dumps(reference)
    metadata = json.loads((model.workdir / "runtime.json").read_text())
    assert metadata["creation_started"] is False
    assert runner.calls == []
    assert cli_models.reconcile_runtime_cleanup(reference, runner=runner)
    assert not model.workdir.exists()
    assert runner.calls == []
    # A durable receipt closes the crash window before Engine clears its DB flag.
    assert cli_models.reconcile_runtime_cleanup(reference, runner=runner)
    model.close()


def test_reconcile_runtime_cleanup_repeats_only_the_owned_nemo_destroy():
    runner = NemoRunner(fail="destroy")
    model = NemoClawModel(MODEL, api_key=KEY, runner=runner)
    model.complete(MESSAGES, TOOLS)
    reference = model.cleanup_reference()
    with pytest.raises(ServiceError):
        cli_models.reconcile_runtime_cleanup(reference, runner=runner)
    assert model.workdir.exists()
    runner.fail = None
    assert cli_models.reconcile_runtime_cleanup(reference, runner=runner)
    argv, options = runner.calls[-1]
    assert argv == [model.binary, model.name, "destroy", "--yes", "--cleanup-gateway"]
    assert options["env"]["NEMOCLAW_GATEWAY_PORT"] == reference["gateway_port"]
    assert "COMPATIBLE_API_KEY" not in options["env"]
    assert options["env"]["HOME"] == str(model.workdir / "home")
    assert not model.workdir.exists()
    model.close()


def test_cleanup_rejects_foreign_paths_names_and_symlinks(tmp_path):
    runner = NemoRunner()
    model = NemoClawModel(MODEL, api_key=KEY, runner=runner)
    reference = model.cleanup_reference()
    for changed in (
        {"workdir": str(tmp_path)},
        {"name": "someone-elses-sandbox"},
        {"gateway_port": "--dangerous"},
        {"kind": "shell"},
    ):
        with pytest.raises(ServiceError, match="invalid_runtime_cleanup_reference"):
            cli_models.reconcile_runtime_cleanup({**reference, **changed}, runner=runner)
    linked = model.workdir.parent / "run-symlink"
    linked.symlink_to(model.workdir, target_is_directory=True)
    with pytest.raises(ServiceError, match="invalid_runtime_cleanup_reference"):
        cli_models.reconcile_runtime_cleanup({**reference, "workdir": str(linked)}, runner=runner)
    linked.unlink()
    assert not runner.calls
    model.close()


def test_cleanup_refuses_missing_authority_and_preserves_a_mismatched_runtime_record():
    runner = NemoRunner()
    model = NemoClawModel(MODEL, api_key=KEY, runner=runner)
    reference = model.cleanup_reference()
    metadata = model.workdir / "runtime.json"
    original = metadata.read_text()
    metadata.write_text(
        json.dumps({"reference": {**reference, "name": "foreign"}, "creation_started": False})
    )
    with pytest.raises(ServiceError, match="invalid_runtime_cleanup_reference"):
        cli_models.reconcile_runtime_cleanup(reference, runner=runner)
    assert model.workdir.exists() and not runner.calls
    metadata.write_text(original)
    model.close()


def test_controlled_cli_path_preserves_only_discovered_runtime_binary_directories(
    monkeypatch, tmp_path
):
    commands = {
        "node": "/opt/nvm/node/bin/node",
        "npm": "/opt/nvm/node/bin/npm",
        "openshell": "/opt/openshell/bin/openshell",
        "docker": "/usr/local/bin/docker",
    }
    monkeypatch.setattr(cli_models.shutil, "which", commands.get)
    monkeypatch.setenv("PATH", "/untrusted-extra-dir")
    environment = cli_models._environment(tmp_path / "home", "/app/cli/nemoclaw")
    assert "/opt/nvm/node/bin" in environment["PATH"]
    assert "/opt/openshell/bin" in environment["PATH"]
    assert "/usr/local/bin" in environment["PATH"]
    assert "/untrusted-extra-dir" not in environment["PATH"]
