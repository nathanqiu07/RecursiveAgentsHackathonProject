import io
import json
import shlex
import sys
import time

import pytest

from agent_harness import sandbox


def command(request_id="one", code="printf hello", timeout=2, **changes):
    return {
        "type": "execute",
        "id": request_id,
        "command": code,
        "timeout_seconds": timeout,
        **changes,
    }


def exchange(tmp_path, *requests):
    inputs = io.BytesIO(
        b"".join(
            (json.dumps(item) if not isinstance(item, str) else item).encode() + b"\n"
            for item in requests
        )
    )
    output = io.StringIO()
    sandbox.serve(inputs, output, tmp_path)
    return [json.loads(line) for line in output.getvalue().splitlines()]


def python(code):
    # execute deliberately exposes a fixed PATH, so use this interpreter's path
    # explicitly when tests run in a virtual environment.
    return shlex.quote(sys.executable) + " -c " + shlex.quote(code)


def test_commands_share_only_their_sandbox_workspace(tmp_path):
    results = exchange(
        tmp_path,
        command(code="printf evidence > result.txt"),
        command("two", "cat result.txt; printf warning >&2"),
    )
    assert [item["exit_code"] for item in results] == [0, 0]
    assert results[1]["output"] == "evidencewarning"
    assert results[1]["type"] == "execution_result"
    assert (tmp_path / "result.txt").read_text() == "evidence"


def test_process_gets_no_host_credentials_or_protocol_stdin(tmp_path, monkeypatch):
    monkeypatch.setenv("VERY_SECRET_TEST_KEY", "do-not-leak")
    reply = exchange(
        tmp_path,
        command(code=python("import os,sys; print(os.getenv('VERY_SECRET_TEST_KEY')); "
                            "print(sys.stdin.read())")),
    )[0]
    assert reply["exit_code"] == 0
    assert reply["output"] == "None\n\n"


def test_runtime_preserves_shell_failures(tmp_path):
    reply = exchange(tmp_path, command(code="printf failed; exit 7"))[0]
    assert reply["exit_code"] == 7
    assert reply["output"] == "failed"


def test_timed_out_process_group_is_killed_and_following_command_works(tmp_path):
    marker = tmp_path / "leaked.txt"
    replies = exchange(
        tmp_path,
        command(code=f"(sleep .3; touch {shlex.quote(str(marker))}) & wait", timeout=0.05),
        command("two"),
    )
    assert replies[0]["exit_code"] == 124
    assert replies[1]["output"] == "hello"
    time.sleep(0.35)
    assert not marker.exists()


def test_normal_exit_kills_background_descendants(tmp_path):
    marker = tmp_path / "background.txt"
    reply = exchange(
        tmp_path,
        command(code=f"(sleep .3; touch {shlex.quote(str(marker))}) >/dev/null 2>&1 &"),
    )[0]
    assert reply["exit_code"] == 0
    time.sleep(0.35)
    assert not marker.exists()


def test_stdout_closure_does_not_bypass_command_deadline(tmp_path):
    started = time.monotonic()
    reply = exchange(tmp_path, command(code="exec 1>&- 2>&-; sleep 5", timeout=0.05))[0]
    assert reply["exit_code"] == 124
    assert time.monotonic() - started < 2


def test_excess_output_is_bounded_and_stops_command(tmp_path):
    reply = exchange(tmp_path, command(code=python("print('a' * 1000000)")))[0]
    assert reply["exit_code"] == 125
    assert reply["truncated"] is True
    assert len(reply["output"].encode()) == sandbox.MAX_OUTPUT_BYTES


def test_same_command_id_replays_saved_result_without_repeating_work(tmp_path):
    request = command(code="printf x >> result.txt; cat result.txt")
    replies = exchange(tmp_path, request, request)
    assert replies[0] == replies[1]
    assert (tmp_path / "result.txt").read_text() == "x"


def test_reusing_command_id_for_different_input_is_rejected(tmp_path):
    replies = exchange(tmp_path, command(), command(code="touch forbidden.txt"))
    assert replies[1]["exit_code"] == 64
    assert not (tmp_path / "forbidden.txt").exists()


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "[]",
        '{"type":"execute","id":"x","command":"true",'
        '"command":"false","timeout_seconds":1}',
        command(timeout=0),
        command(timeout=301),
        command(timeout=True),
        command(timeout=float("nan")),
        command(code=""),
        command(code="x" * (sandbox.MAX_COMMAND_BYTES + 1)),
        command(request_id="invalid id"),
        command(env={"SECRET": "value"}),
    ],
)
def test_invalid_requests_are_rejected_and_stream_recovers(tmp_path, payload):
    replies = exchange(tmp_path, payload, command("valid"))
    assert replies[0]["exit_code"] == 64
    assert replies[1]["exit_code"] == 0


def test_oversized_line_is_consumed_without_eating_following_request(tmp_path):
    replies = exchange(tmp_path, "x" * (sandbox.MAX_LINE_BYTES * 2), command())
    assert len(replies) == 2
    assert replies[0]["exit_code"] == 64
    assert replies[1]["output"] == "hello"


def test_unique_command_budget_cannot_be_bypassed_by_replays(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "MAX_COMMANDS", 1)
    first = command()
    replies = exchange(tmp_path, first, first, command("two", "touch forbidden.txt"))
    assert replies[0] == replies[1]
    assert replies[2]["exit_code"] == 64
    assert not (tmp_path / "forbidden.txt").exists()


def test_eof_has_no_protocol_output(tmp_path):
    assert exchange(tmp_path) == []
