"""Native CLI planners with every research/compute/publication tool brokered by Engine.

Claude Code runs with its built-in tools, hooks, skills and MCP disabled. NemoClaw
runs a similarly tool-denied OpenClaw planner inside a newly created OpenShell
sandbox and destroys that sandbox after the run. Neither adapter executes a
model-selected host command or silently substitutes a generic model endpoint.
"""

import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

from jsonschema import Draft202012Validator

from .clients import MAX_REQUEST_BYTES, ServiceError, _json, _key, _url

_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,199}\Z")
_FUNCTION = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,99}\Z")
_NEMO_NAME = re.compile(r"harness-[0-9a-f]{24}\Z")
_PLANNER_INSTRUCTION = (
    "You are the planning component of a research harness. Treat the supplied messages as the "
    "conversation to continue. Return the requested JSON envelope only. Tool calls in the envelope "
    "are proposals: the harness validates and executes them later. Never invoke built-in tools, "
    "shell commands, MCP, subagents or plugins. Use only the tool names and argument schemas in "
    "the supplied tool catalog. When no tool is appropriate, return content and an empty "
    "tool_calls array. Do not invent experimental results or claim a tool was executed."
)
# The fixed program runs only inside the new OpenShell sandbox. The prompt is
# transferred through intentional stdin, rather than exposing it in host argv.
_NEMO_TURN = """import json, subprocess, sys
request = json.load(sys.stdin)
result = subprocess.run([
    'openclaw', 'agent', '--agent', 'main', '--session-id', request['session_id'],
    '--json', '--timeout', '120', '--message', request['prompt']
], stdin=subprocess.DEVNULL, check=False)
sys.exit(result.returncode)
"""


def _binary(name, variable):
    configured = os.environ.get(variable)
    if configured:
        candidate = Path(configured)
        if candidate.is_absolute() and candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        return None
    found = shutil.which(name)
    return str(Path(found).resolve()) if found else None


def claude_code_status():
    available = bool(_binary("claude", "HARNESS_CLAUDE_CODE_BIN"))
    return {
        "available": available,
        "message": (
            "Claude Code CLI is installed. A profile API key and Anthropic-compatible endpoint "
            "are required; dashboard checks do not make a paid request."
            if available
            else "Install Claude Code or set HARNESS_CLAUDE_CODE_BIN to its executable."
        ),
    }


def nemoclaw_status():
    if not _binary("nemoclaw", "HARNESS_NEMOCLAW_BIN"):
        return {
            "available": False,
            "message": "NVIDIA NemoClaw CLI is not installed on this worker.",
        }
    if os.environ.get("HARNESS_NEMOCLAW_ENABLED") != "1":
        return {
            "available": False,
            "message": "Prepare NemoClaw/OpenShell and its managed image, then set "
            "HARNESS_NEMOCLAW_ENABLED=1 on the worker.",
        }
    if not shutil.which("openshell") or not shutil.which("docker"):
        return {
            "available": False,
            "message": "NemoClaw requires OpenShell and Docker on the worker.",
        }
    # Read-only daemon probe; no Docker credential file or inherited CLI config.
    try:
        result = subprocess.run(
            [shutil.which("docker"), "info", "--format", "{{.ServerVersion}}"],
            env={"PATH": os.defpath},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
        if result.returncode:
            raise OSError
    except (OSError, subprocess.TimeoutExpired):
        return {
            "available": False,
            "message": "Docker daemon is unavailable for NemoClaw sandboxes.",
        }
    return {
        "available": True,
        "message": "NemoClaw is enabled. Each run must successfully provision its own OpenShell "
        "sandbox before planning; image and endpoint compatibility are checked then.",
    }


def _configuration(model, base_url, api_key):
    if not isinstance(model, str) or not _MODEL.fullmatch(model):
        raise ServiceError("invalid_configuration")
    url = _url(base_url)
    if "?" in base_url or ".." in url.path.split("/"):
        raise ServiceError("invalid_url")
    if url.scheme == "http" and url.host not in {"localhost", "127.0.0.1", "::1"}:
        raise ServiceError("invalid_url")
    key = _key(api_key)
    if not key or not key.strip() or len(key) > 4096 or key in model:
        raise ServiceError("invalid_configuration")
    return model, str(url).rstrip("/"), key


def _environment(home, binary):
    home = Path(home)
    for directory in (home, home / ".config", home / ".cache", home / ".tmp"):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Deliberately do not inherit proxy, cloud, Git, SSH, auth, plugin or project variables.
    directories = [str(Path(binary).parent)]
    # Discover exact executable directories before replacing HOME and PATH; Node
    # installations in NVM/Homebrew and Docker's CLI otherwise become invisible.
    for command in ("node", "npm", "openshell", "docker"):
        executable = shutil.which(command)
        if executable:
            directories.append(str(Path(executable).parent))
    path = os.pathsep.join(dict.fromkeys([*directories, *os.defpath.split(os.pathsep)]))
    return {
        "PATH": path,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "TMPDIR": str(home / ".tmp"),
        "LANG": "C.UTF-8",
        "NO_COLOR": "1",
        "CI": "1",
    }


def _state_parent():
    return Path(
        os.environ.get("HARNESS_NEMOCLAW_STATE_DIR")
        or Path(__file__).resolve().parent.parent / "data" / "nemoclaw"
    ).resolve()


def _private_directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ServiceError("invalid_runtime_cleanup_reference")


def _write_private_json(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".runtime-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output, allow_nan=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_private_json(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 4096:
            raise ServiceError("invalid_runtime_cleanup_reference")
        return _json(os.read(descriptor, 4097).decode())
    finally:
        os.close(descriptor)


def reconcile_runtime_cleanup(reference, *, runner=None):
    """Retry only an owned run's NemoClaw destroy; preserve uncertain recovery state.

    Completion receipts contain no credential or prompt content. They make a
    worker crash after successful destruction, before clearing its DB flag,
    recoverable even after the private runtime home has already been removed.
    """
    try:
        if not isinstance(reference, dict) or set(reference) != {
            "kind",
            "workdir",
            "name",
            "gateway_port",
        }:
            raise ValueError
        if reference["kind"] != "nemoclaw" or not _NEMO_NAME.fullmatch(reference["name"]):
            raise ValueError
        port = reference["gateway_port"]
        if not isinstance(port, str) or not port.isascii() or not port.isdigit():
            raise ValueError
        if not 1024 <= int(port) <= 65535:
            raise ValueError
        parent = _state_parent()
        workdir = Path(reference["workdir"])
        if (
            not workdir.is_absolute()
            or workdir.parent != parent
            or not re.fullmatch(r"run-[A-Za-z0-9_-]+", workdir.name)
        ):
            raise ValueError
        _private_directory(parent)
        receipts = parent / ".completed"
        if receipts.exists() or receipts.is_symlink():
            _private_directory(receipts)
        receipt = receipts / (workdir.name + ".json")
        completed = False
        if receipt.exists() or receipt.is_symlink():
            if _read_private_json(receipt) != reference:
                raise ValueError
            completed = True
        if not completed or workdir.exists() or workdir.is_symlink():
            _private_directory(workdir)
            metadata = _read_private_json(workdir / "runtime.json")
            if (
                metadata.get("reference") != reference
                or type(metadata.get("creation_started")) is not bool
            ):
                raise ValueError
            if not completed and metadata["creation_started"]:
                binary = _binary("nemoclaw", "HARNESS_NEMOCLAW_BIN")
                if binary is None and runner is None:
                    raise ServiceError("nemoclaw_cleanup_required")
                home = workdir / "home"
                _private_directory(home)
                for child in (home / ".config", home / ".cache", home / ".tmp"):
                    if child.exists() or child.is_symlink():
                        _private_directory(child)
                planner = _CLIPlanner()
                planner._initialize(runner)
                planner.workdir = workdir
                planner.env = _environment(home, binary or "nemoclaw")
                planner.env.update({"NEMOCLAW_GATEWAY_PORT": port, "NEMOCLAW_NON_INTERACTIVE": "1"})
                try:
                    planner._execute(
                        [
                            binary or "nemoclaw",
                            reference["name"],
                            "destroy",
                            "--yes",
                            "--cleanup-gateway",
                        ],
                        timeout=180,
                    )
                finally:
                    planner._stop()
            if not completed:
                receipts.mkdir(mode=0o700, exist_ok=True)
                _private_directory(receipts)
                _write_private_json(receipt, reference)
            shutil.rmtree(workdir)
        return True
    except (ValueError, TypeError, KeyError, AttributeError, OSError, UnicodeError):
        raise ServiceError("invalid_runtime_cleanup_reference") from None


def _schema(tools):
    if tools is None:
        tools = []
    if not isinstance(tools, list) or len(tools) > 100:
        raise ServiceError("invalid_tools")
    names = []
    try:
        for tool in tools:
            name = tool["function"]["name"]
            if (
                tool["type"] != "function"
                or not isinstance(name, str)
                or not _FUNCTION.fullmatch(name)
            ):
                raise ValueError
            names.append(name)
    except (TypeError, KeyError, ValueError):
        raise ServiceError("invalid_tools") from None
    if len(names) != len(set(names)):
        raise ServiceError("invalid_tools")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["content", "tool_calls"],
        "properties": {
            "content": {"type": ["string", "null"], "maxLength": 220000},
            "tool_calls": {
                "type": "array",
                "maxItems": 8 if names else 0,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "name", "arguments"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1, "maxLength": 200},
                        "name": {"type": "string", **({"enum": names} if names else {})},
                        "arguments": {"type": "string", "maxLength": 140000},
                    },
                },
            },
        },
    }


def _prompt(messages, tools, schema):
    if not isinstance(messages, list) or not messages or len(messages) > 300:
        raise ServiceError("invalid_messages")
    try:
        value = json.dumps(
            {
                "instruction": _PLANNER_INSTRUCTION,
                "messages": messages,
                "tools": tools or [],
                "output_schema": schema,
            },
            allow_nan=False,
            ensure_ascii=False,
        )
        if len(value.encode()) > MAX_REQUEST_BYTES:
            raise ValueError
        return value
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ServiceError("invalid_messages") from None


def _decode_plan(value, schema, key):
    try:
        if not isinstance(value, dict) or not Draft202012Validator(schema).is_valid(value):
            raise ValueError
        if key in json.dumps(value):
            raise ValueError
        if not value["content"] and not value["tool_calls"]:
            raise ValueError
        calls = []
        for call in value["tool_calls"]:
            arguments = _json(call["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError
            # IDs must remain unique across independently launched CLI turns.
            calls.append(
                {
                    "id": "call_" + uuid4().hex,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
            )
        result = {"role": "assistant", "content": value["content"]}
        if calls:
            result["tool_calls"] = calls
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise ServiceError("invalid_model_response") from None


class _CLIPlanner:
    def _initialize(self, runner):
        self._runner = runner
        self._lock = threading.Lock()
        self._processes = set()
        self._closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    @staticmethod
    def _kill(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def _execute(self, argv, *, prompt="", timeout=180, env=None, allow_closed=False):
        if self._closed.is_set() and not allow_closed:
            raise ServiceError("runtime_closed")
        if self._runner is not None:
            return self._runner(
                argv, input=prompt, env=env or self.env, cwd=str(self.workdir), timeout=timeout
            )
        with self._lock:
            if self._closed.is_set() and not allow_closed:
                raise ServiceError("runtime_closed")
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=str(self.workdir),
                    env=env or self.env,
                    start_new_session=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                raise ServiceError("runtime_unavailable") from None
            self._processes.add(process)
        oversized = threading.Event()
        chunks = []

        def read():
            size = 0
            while chunk := process.stdout.read(8192):
                size += len(chunk)
                if size > MAX_REQUEST_BYTES:
                    oversized.set()
                    self._kill(process)
                    break
                chunks.append(chunk)

        def write():
            try:
                process.stdin.write(prompt.encode())
                process.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

        reader, writer = (
            threading.Thread(target=read, daemon=True),
            threading.Thread(target=write, daemon=True),
        )
        reader.start()
        writer.start()
        try:
            process.wait(timeout=timeout)
            reader.join(timeout=2)
            writer.join(timeout=2)
            if oversized.is_set():
                raise ServiceError("response_too_large")
            if process.returncode:
                raise ServiceError("cli_runtime_failed")
            return b"".join(chunks).decode("utf-8")
        except subprocess.TimeoutExpired:
            self._kill(process)
            process.wait(timeout=5)
            raise ServiceError("timeout") from None
        except UnicodeError:
            raise ServiceError("invalid_model_response") from None
        finally:
            self._kill(process)
            with self._lock:
                self._processes.discard(process)
            for stream in (process.stdin, process.stdout):
                try:
                    stream.close()
                except OSError:
                    pass

    def _stop(self):
        self._closed.set()
        with self._lock:
            for process in self._processes:
                self._kill(process)

    def complete_json(self, system, user):
        if not isinstance(system, str) or not isinstance(user, (str, dict, list)):
            raise ServiceError("invalid_messages")
        try:
            content = user if isinstance(user, str) else json.dumps(user, allow_nan=False)
            result = self.complete(
                [
                    {
                        "role": "system",
                        "content": system + " Return the answer as a JSON object in "
                        "the content string of the planner envelope; tool_calls must be empty.",
                    },
                    {"role": "user", "content": content},
                ]
            )
            value = _json(result["content"])
            if not isinstance(value, dict) or result.get("tool_calls"):
                raise ValueError
            return value
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise ServiceError("invalid_model_response") from None


class ClaudeCodeModel(_CLIPlanner):
    def __init__(self, model, base_url=None, api_key=None, sandbox=None, *, runner=None):
        self.model, self.base_url, self.api_key = _configuration(
            model,
            base_url
            or os.environ.get("HARNESS_CLAUDE_BASE_URL")
            or os.environ.get("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com",
            api_key
            or os.environ.get("HARNESS_CLAUDE_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self.binary = _binary("claude", "HARNESS_CLAUDE_CODE_BIN")
        if not self.binary and runner is None:
            raise ServiceError("claude_code_unavailable")
        self.binary = self.binary or "claude"
        self._temporary = tempfile.TemporaryDirectory(prefix="harness-claude-")
        self.workdir = Path(self._temporary.name)
        self.env = _environment(self.workdir / "home", self.binary)
        self.env.update(
            {
                "ANTHROPIC_API_KEY": self.api_key,
                "ANTHROPIC_BASE_URL": self.base_url,
                "CLAUDE_CONFIG_DIR": str(self.workdir / "home" / ".claude"),
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_AUTOUPDATER": "1",
                "DISABLE_ERROR_REPORTING": "1",
                "DISABLE_TELEMETRY": "1",
            }
        )
        self._initialize(runner)

    def complete(self, messages, tools=None):
        schema = _schema(tools)
        prompt = _prompt(messages, tools, schema)
        argv = [
            self.binary,
            "--print",
            "--bare",
            "--output-format",
            "json",
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--setting-sources",
            "",
            "--settings",
            '{"disableAllHooks":true}',
            "--disable-slash-commands",
            "--no-session-persistence",
            "--no-chrome",
            "--model",
            self.model,
            "--system-prompt",
            _PLANNER_INSTRUCTION,
            "--json-schema",
            json.dumps(schema),
        ]
        output = self._execute(argv, prompt=prompt)
        try:
            envelope = _json(output)
            if not isinstance(envelope, dict) or envelope.get("is_error"):
                raise ValueError
            if envelope.get("mcp_servers") or envelope.get("mcp_server_errors"):
                raise ValueError
            value = envelope.get("structured_output")
            return _decode_plan(value, schema, self.api_key)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise ServiceError("invalid_model_response") from None

    def close(self):
        self._stop()
        self.env.pop("ANTHROPIC_API_KEY", None)
        self._temporary.cleanup()


class NemoClawModel(_CLIPlanner):
    def __init__(self, model, base_url=None, api_key=None, sandbox=None, *, runner=None):
        self.model, self.base_url, self.api_key = _configuration(
            model,
            base_url
            or os.environ.get("HARNESS_NEMOCLAW_BASE_URL")
            or "https://integrate.api.nvidia.com/v1",
            api_key or os.environ.get("HARNESS_NEMOCLAW_API_KEY"),
        )
        self.binary = _binary("nemoclaw", "HARNESS_NEMOCLAW_BIN")
        if runner is None and not nemoclaw_status()["available"]:
            raise ServiceError("nemoclaw_unavailable")
        self.binary = self.binary or "nemoclaw"
        parent = _state_parent()
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.workdir = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
        self.name = "harness-" + uuid4().hex[:24]
        self.env = _environment(self.workdir / "home", self.binary)
        # Each run owns a gateway because NemoClaw gateways have a shared inference route.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.env.update(
            {
                "NEMOCLAW_GATEWAY_PORT": str(port),
                "NEMOCLAW_NON_INTERACTIVE": "1",
                "NEMOCLAW_WEB_SEARCH_PROVIDER": "none",
                "NEMOCLAW_MINIMAL_BOOTSTRAP": "1",
            }
        )
        self.manifest = self.workdir / "agents.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "main": {
                        "tools": {
                            "profile": "minimal",
                            "deny": ["*"],
                        }
                    }
                }
            )
        )
        self.manifest.chmod(0o600)
        self._initialize(runner)
        self._created = False
        self._creation_started = False
        self._lifecycle = threading.Lock()
        _write_private_json(
            self.workdir / "runtime.json",
            {
                "reference": self.cleanup_reference(),
                "creation_started": False,
            },
        )

    def cleanup_reference(self):
        return {
            "kind": "nemoclaw",
            "workdir": str(self.workdir),
            "name": self.name,
            "gateway_port": self.env["NEMOCLAW_GATEWAY_PORT"],
        }

    def _ensure_sandbox(self):
        with self._lifecycle:
            if self._created:
                return
            if self._creation_started:
                raise ServiceError("nemoclaw_creation_uncertain")
            _write_private_json(
                self.workdir / "runtime.json",
                {
                    "reference": self.cleanup_reference(),
                    "creation_started": True,
                },
            )
            self._creation_started = True
            environment = {
                **self.env,
                "NEMOCLAW_PROVIDER": "custom",
                "NEMOCLAW_MODEL": self.model,
                "NEMOCLAW_ENDPOINT_URL": self.base_url,
                "NEMOCLAW_PREFERRED_API": "openai-completions",
                "COMPATIBLE_API_KEY": self.api_key,
            }
            self._execute(
                [
                    self.binary,
                    "onboard",
                    "--non-interactive",
                    "--yes",
                    "--fresh",
                    "--name",
                    self.name,
                    "--agent",
                    "openclaw",
                    "--agents",
                    str(self.manifest),
                    "--no-gpu",
                    "--no-sandbox-gpu",
                    "--tool-disclosure",
                    "direct",
                    "--yes-i-accept-third-party-software",
                ],
                env=environment,
                timeout=600,
            )
            # Require the actual managed sandbox to pass NemoClaw's readiness boundary.
            self._execute([self.binary, self.name, "connect", "--probe-only"], timeout=180)
            policy = self._execute(
                [
                    self.binary,
                    self.name,
                    "config",
                    "get",
                    "--key",
                    "agents.list",
                    "--format",
                    "json",
                ],
                timeout=30,
            )
            try:
                agents = _json(policy)
                primary = next(item for item in agents if item.get("id") == "main")
                if primary.get("tools", {}).get("deny") != ["*"]:
                    raise ValueError
            except (TypeError, ValueError, StopIteration, AttributeError):
                raise ServiceError("nemoclaw_tool_policy_unverified") from None
            self._created = True

    def complete(self, messages, tools=None):
        schema = _schema(tools)
        prompt = _prompt(messages, tools, schema)
        self._ensure_sandbox()
        output = self._execute(
            [
                self.binary,
                self.name,
                "exec",
                "--workdir",
                "/sandbox",
                "--no-tty",
                "--stdin",
                "--timeout",
                "150",
                "--",
                "python3",
                "-c",
                _NEMO_TURN,
            ],
            prompt=json.dumps({"session_id": str(uuid4()), "prompt": prompt}),
            timeout=180,
        )
        try:
            envelope = _json(output)
            if not isinstance(envelope, dict) or envelope.get("status") != "ok":
                raise ValueError
            payloads = envelope["result"]["payloads"]
            if not isinstance(payloads, list) or len(payloads) != 1:
                raise ValueError
            value = _json(payloads[0]["text"])
            return _decode_plan(value, schema, self.api_key)
        except (KeyError, TypeError, ValueError, UnicodeError, RecursionError):
            raise ServiceError("invalid_model_response") from None

    def close(self):
        self._stop()
        if not self._lifecycle.acquire(timeout=10):
            raise ServiceError("nemoclaw_cleanup_required")
        try:
            if not reconcile_runtime_cleanup(self.cleanup_reference(), runner=self._runner):
                raise ServiceError("nemoclaw_cleanup_required")
            self._creation_started = False
        except ServiceError:
            # Preserve exact gateway state and recovery authority for an operator retry.
            raise ServiceError("nemoclaw_cleanup_required") from None
        finally:
            self._lifecycle.release()
