import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from agent_harness import codex_model as cm
from agent_harness.clients import ServiceError

MODEL = "gpt-6.1-sol"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sandbox_execute",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
]


@pytest.fixture
def planner(monkeypatch):
    monkeypatch.setattr(cm, "_binary", lambda: sys.executable)
    monkeypatch.setattr(
        cm,
        "_catalog",
        lambda _: [
            {
                "slug": MODEL,
                "visibility": "list",
                "shell_type": "unified_exec",
                "apply_patch_tool_type": "freeform",
                "base_instructions": "native instructions",
            }
        ],
    )
    cm._AUDITED.clear()
    with cm.CodexModel() as model:
        yield model
    cm._AUDITED.clear()


def events(document, *, extra=None, completed=True):
    values = [] if extra is None else [extra]
    values.append(
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": json.dumps(document),
            },
        }
    )
    if completed:
        values.append({"type": "turn.completed"})
    return "\n".join(json.dumps(item) for item in values).encode()


def fake_completion(monkeypatch, planner, result):
    monkeypatch.setattr(planner, "_audit", lambda _deadline=None: None)
    monkeypatch.setattr(planner, "_invoke", lambda *_args, **_kwargs: result)


def test_planner_only_proposes_engine_tool_calls(monkeypatch, planner):
    fake_completion(
        monkeypatch,
        planner,
        events(
            {
                "content": None,
                "tool_calls": [
                    {
                        "name": "sandbox_execute",
                        "arguments": '{"command":"python experiment.py"}',
                    }
                ],
            }
        ),
    )
    answer = planner.complete([{"role": "user", "content": "Test the idea"}], TOOLS)
    assert answer["role"] == "assistant"
    call = answer["tool_calls"][0]
    assert call["id"].startswith("codex_call_")
    assert call["function"] == {
        "name": "sandbox_execute",
        "arguments": '{"command":"python experiment.py"}',
    }


@pytest.mark.parametrize(
    "call",
    [
        {"name": "exec_command", "arguments": "{}"},
        {"name": "sandbox_execute", "arguments": "[]"},
        {"name": "sandbox_execute", "arguments": '{"a":1,"a":2}'},
        {"name": "sandbox_execute", "arguments": {"command": "uname"}},
    ],
)
def test_native_or_malformed_tool_calls_are_rejected(monkeypatch, planner, call):
    fake_completion(monkeypatch, planner, events({"content": None, "tool_calls": [call]}))
    with pytest.raises(ServiceError, match="invalid_model_response"):
        planner.complete([{"role": "user", "content": "test"}], TOOLS)


def test_json_review_unwraps_only_a_valid_object(monkeypatch, planner):
    fake_completion(monkeypatch, planner, events({"json": '{"novel":true,"evidence":[1,2]}'}))
    assert planner.complete_json("Review novelty", {"paper": "Evidence"}) == {
        "novel": True,
        "evidence": [1, 2],
    }


@pytest.mark.parametrize("document", [{"json": "[]"}, {"json": '{"a":1,"a":2}'}, {}])
def test_invalid_json_review_fails_safely(monkeypatch, planner, document):
    fake_completion(monkeypatch, planner, events(document))
    with pytest.raises(ServiceError, match="invalid_model_response"):
        planner.complete_json("Review", "paper")


@pytest.mark.parametrize(
    "kind", ["command_execution", "file_change", "mcp_tool_call", "web_search"]
)
def test_native_execution_events_never_count_as_planner_success(monkeypatch, planner, kind):
    fake_completion(
        monkeypatch,
        planner,
        events(
            {"content": "done", "tool_calls": []},
            extra={"type": "item.completed", "item": {"type": kind}},
        ),
    )
    with pytest.raises(ServiceError, match="invalid_model_response"):
        planner.complete([{"role": "user", "content": "test"}])


def test_incomplete_cli_turn_is_not_a_success(monkeypatch, planner):
    fake_completion(monkeypatch, planner, events({"json": "{}"}, completed=False))
    with pytest.raises(ServiceError, match="invalid_model_response"):
        planner.complete_json("Review", "test")


def test_profile_disables_tools_and_ignores_user_configuration(tmp_path, planner):
    arguments = cm._argv(planner.binary, str(tmp_path), MODEL, {}, planner.entries)
    for required in ["--no-daemon", "--ignore-user-config", "--ignore-rules", "--ephemeral"]:
        assert required in arguments
    assert arguments[arguments.index("--sandbox") + 1] == "read-only"
    assert 'forced_login_method="chatgpt"' in arguments
    for feature in cm._DISABLED:
        assert "features." + feature + "=false" in arguments
    catalog = json.loads((tmp_path / "models.json").read_text())["models"][0]
    assert catalog["apply_patch_tool_type"] is None
    assert catalog["shell_type"] == "disabled"
    assert catalog["experimental_supported_tools"] == []
    assert "sqlite_home" not in " ".join(arguments)


@pytest.mark.parametrize("unsafe", ["tools", "credentials", "path"])
def test_loopback_safety_audit_fails_closed(monkeypatch, planner, unsafe):
    def invoke(arguments, _prompt, **_kwargs):
        option = next(
            value
            for value in arguments
            if value.startswith("model_providers.harness_audit.base_url=")
        )
        base = json.loads(option.split("=", 1)[1])
        headers = {"Content-Type": "application/json"}
        if unsafe == "credentials":
            headers["Authorization"] = "Bearer dummy-secret"
        data = {
            "tools": [{"type": "function", "name": "exec_command"}] if unsafe == "tools" else []
        }
        path = "/wrong" if unsafe == "path" else "/responses"
        request = urllib.request.Request(base + path, json.dumps(data).encode(), headers)
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                response.read()
        except urllib.error.HTTPError:
            pass
        return b""

    monkeypatch.setattr(planner, "_invoke", invoke)
    with pytest.raises(ServiceError, match="codex_isolation_unavailable"):
        planner._audit()


def test_loopback_safety_audit_accepts_omitted_tools_and_caches_by_binary(monkeypatch, planner):
    calls = []

    def invoke(arguments, _prompt, **_kwargs):
        calls.append(arguments)
        option = next(
            value
            for value in arguments
            if value.startswith("model_providers.harness_audit.base_url=")
        )
        base = json.loads(option.split("=", 1)[1])
        request = urllib.request.Request(
            base + "/responses", b"{}", {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            assert b"response.completed" in response.read()
        return b""

    monkeypatch.setattr(planner, "_invoke", invoke)
    planner._audit()
    planner._audit()
    assert len(calls) == 1


def test_subprocess_timeout_terminates_the_planner_without_raw_output(tmp_path, planner):
    arguments = [sys.executable, "-c", "import time; time.sleep(10)", "--cd", str(tmp_path)]
    started = time.monotonic()
    with pytest.raises(ServiceError, match="codex_timeout"):
        planner._invoke(arguments, "test", timeout=0.15)
    assert time.monotonic() - started < 2
    assert not planner._processes


def test_close_interrupts_an_active_planner(tmp_path, planner):
    arguments = [sys.executable, "-c", "import time; time.sleep(10)", "--cd", str(tmp_path)]
    errors = []

    def run():
        try:
            planner._invoke(arguments, "test", timeout=10)
        except ServiceError as error:
            errors.append(error.code)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 2
    while not planner._processes and time.monotonic() < deadline:
        time.sleep(0.01)
    planner.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == ["codex_cancelled"]
    assert not planner._processes


def test_secret_environment_does_not_reach_codex_subprocess(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-api-key")
    monkeypatch.setenv("CRUSOE_API_KEY", "dummy-crusoe-key")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "dummy-token")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example")
    env = cm._environment()
    assert not any(
        name in env
        for name in [
            "OPENAI_API_KEY",
            "CRUSOE_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "HTTPS_PROXY",
        ]
    )
    assert env.get("HOME") == cm.os.environ.get("HOME")


def test_status_never_returns_authentication_command_output(monkeypatch):
    monkeypatch.setattr(cm, "_binary", lambda: sys.executable)

    def command(_binary, arguments):
        if arguments[0] == "exec":
            return subprocess.CompletedProcess(
                [], 0, b"--ignore-user-config --ignore-rules --ephemeral --output-schema", b""
            )
        return subprocess.CompletedProcess([], 0, b"Logged in using API key dummy-secret", b"")

    monkeypatch.setattr(cm, "_small_command", command)
    status = cm.codex_status()
    assert status["available"] is False
    assert status["auth_mode"] == "api_key"
    assert "dummy-secret" not in json.dumps(status)


def test_catalog_labels_are_sanitized_and_no_entitlement_is_inferred(monkeypatch):
    monkeypatch.setattr(cm, "_binary", lambda: sys.executable)
    monkeypatch.setattr(
        cm,
        "_catalog",
        lambda _: [
            {
                "slug": MODEL,
                "display_name": "Model",
                "visibility": "list",
                "secret": "dummy-secret",
            },
            {"slug": "hidden-model", "visibility": "hide"},
        ],
    )
    assert cm.codex_models() == [{"id": MODEL, "label": "Model"}]
