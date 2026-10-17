"""Offline tests for `team.crusoe`: no network, no live key. A fake OpenAI client stands in
for the server so retry, JSON-mode fallback, usage and streaming paths are exercised.
Slot files live in temporary directories, never the shared /tmp/crusoe-slots."""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from team.crusoe import (  # noqa: E402
    NANO,
    SUPER,
    CrusoeLLM,
    MissingKeyError,
    SlotSemaphore,
    TokenBucket,
    cost_usd,
    extract_json,
    strip_think,
)
from team.types import Reply  # noqa: E402

FAKE_KEY = "test-key-not-real"
_SLOT_TMP = tempfile.TemporaryDirectory(prefix="crusoe-slots-test-")


def completion(content: str | None, reasoning: str | None = None, prompt: int = 100, out: int = 10, cached: int | None = None) -> Any:
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=out, prompt_tokens_details=details)
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def chunk(content: str | None = None, reasoning: str | None = None, usage: Any = None) -> Any:
    choices = [] if usage is not None else [SimpleNamespace(delta=SimpleNamespace(content=content, reasoning_content=reasoning))]
    return SimpleNamespace(choices=choices, usage=usage)


def api_error(status: int, headers: dict[str, str] | None = None) -> Exception:
    import openai

    try:
        import httpx2 as httpx
    except ImportError:
        import httpx  # type: ignore[no-redef]
    classes = {400: openai.BadRequestError, 401: openai.AuthenticationError, 429: openai.RateLimitError, 503: openai.InternalServerError}
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {})
    return classes[status](f"status {status}", response=response, body=None)


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_llm(
    outcomes: list[Any],
    sleep: Callable[[float], None] = lambda s: None,
    completions: Any = None,
    **env: str,
) -> tuple[CrusoeLLM, Any]:
    """`env` overrides CRUSOE_* settings read at construction; slots default to a temp dir."""
    completions = completions if completions is not None else FakeCompletions(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    settings = {"CRUSOE_SLOT_DIR": _SLOT_TMP.name, "CRUSOE_MAX_CONCURRENT": "3", "CRUSOE_MAX_RETRIES": "8", **env}
    with mock.patch.dict(os.environ, settings):
        llm = CrusoeLLM(api_key=FAKE_KEY, rpm=6000, dotenv_path=None, client=client, sleep=sleep)
    return llm, completions


class SlowCompletions:
    """Fake endpoint that takes `delay_s` per request and records the peak number in flight."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        with self.lock:
            self.in_flight += 1
            self.calls += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            time.sleep(self.delay_s)
            return completion("ok")
        finally:
            with self.lock:
                self.in_flight -= 1


def slot_files_free(root: str | Path) -> tuple[int, int]:
    """(slot files nobody holds, slot files in total) under `root`."""
    paths = sorted(Path(root).rglob("*.lock"))
    free = 0
    for path in paths:
        fd = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            free += 1
        except BlockingIOError:
            pass
        finally:
            os.close(fd)
    return free, len(paths)


class ExtractJsonTest(unittest.TestCase):
    def test_plain_object(self) -> None:
        self.assertEqual(extract_json('{"ok": true}'), {"ok": True})

    def test_fenced_with_language(self) -> None:
        self.assertEqual(extract_json('Here you go:\n```json\n{"a": [1, 2]}\n```\nDone.'), {"a": [1, 2]})

    def test_fenced_without_language(self) -> None:
        self.assertEqual(extract_json("```\n[1, 2, 3]\n```"), [1, 2, 3])

    def test_leading_and_trailing_prose(self) -> None:
        text = 'Sure! The answer is {"model": "nano", "note": "a } in a string"} hope that helps.'
        self.assertEqual(extract_json(text), {"model": "nano", "note": "a } in a string"})

    def test_skips_brace_that_is_not_json(self) -> None:
        self.assertEqual(extract_json('set {x} then {"y": 1}'), {"y": 1})

    def test_ignores_json_inside_think(self) -> None:
        self.assertEqual(extract_json('<think>maybe {"draft": 1}</think>{"final": 2}'), {"final": 2})

    def test_raises_when_no_json(self) -> None:
        with self.assertRaises(ValueError):
            extract_json("no json here")


class StripThinkTest(unittest.TestCase):
    def test_closed_block(self) -> None:
        self.assertEqual(strip_think("<think>\nreasoning\n</think>\n\nAnswer"), "Answer")

    def test_dangling_close_tag(self) -> None:
        self.assertEqual(strip_think("reasoning opened by the template</think>Answer"), "Answer")

    def test_unclosed_open_tag(self) -> None:
        self.assertEqual(strip_think("Answer<think>cut off mid"), "Answer")

    def test_no_tags_and_none(self) -> None:
        self.assertEqual(strip_think("  Answer  "), "Answer")
        self.assertEqual(strip_think(None), "")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TokenBucketTest(unittest.TestCase):
    def test_spacing_after_burst_is_one_interval(self) -> None:
        clock = FakeClock()
        bucket = TokenBucket(60, capacity=1, clock=clock.time, sleep=clock.sleep)
        waits = [bucket.acquire() for _ in range(4)]
        self.assertEqual(waits[0], 0.0)
        for w in waits[1:]:
            self.assertAlmostEqual(w, 1.0)
        self.assertAlmostEqual(clock.now, 3.0)

    def test_burst_capacity_then_throttle(self) -> None:
        clock = FakeClock()
        bucket = TokenBucket(30, capacity=3, clock=clock.time, sleep=clock.sleep)
        waits = [bucket.acquire() for _ in range(5)]
        self.assertEqual(waits[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(waits[3], 2.0)
        self.assertAlmostEqual(waits[4], 2.0)

    def test_at_most_capacity_plus_rate_in_first_minute(self) -> None:
        clock = FakeClock()
        bucket = TokenBucket(25, capacity=3, clock=clock.time, sleep=clock.sleep)
        sent = 0
        while True:
            bucket.acquire()
            if clock.now > 60.0:
                break
            sent += 1
        self.assertLessEqual(sent, 28)

    def test_concurrent_threads_are_spaced(self) -> None:
        bucket = TokenBucket(600, capacity=1)  # one slot per 0.1s
        stamps: list[float] = []
        lock = threading.Lock()

        def worker() -> None:
            bucket.acquire()
            with lock:
                stamps.append(time.monotonic())

        start = time.monotonic()
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(stamps), 5)
        self.assertGreaterEqual(max(stamps) - start, 0.38)


class CostTest(unittest.TestCase):
    def test_uncached(self) -> None:
        self.assertAlmostEqual(cost_usd(NANO, 1_000_000, 1_000_000), 0.05 + 0.20)
        self.assertAlmostEqual(cost_usd(SUPER, 2_000_000, 500_000), 2 * 0.30 + 0.5 * 2.40)

    def test_cached_tokens_bill_at_cached_rate(self) -> None:
        expected = (600_000 * 0.05 + 400_000 * 0.03 + 100_000 * 0.20) / 1_000_000
        self.assertAlmostEqual(cost_usd(NANO, 1_000_000, 100_000, cached_tokens=400_000), expected)

    def test_unknown_model_costs_zero(self) -> None:
        self.assertEqual(cost_usd("someone/else", 1000, 1000), 0.0)


class ConfigTest(unittest.TestCase):
    def test_missing_key_names_the_variable(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("CRUSOE_API_KEY", None)
            with self.assertRaises(MissingKeyError) as ctx:
                CrusoeLLM(dotenv_path=None)
        self.assertIn("CRUSOE_API_KEY", str(ctx.exception))

    def test_rpm_and_base_url_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"CRUSOE_RPM": "12", "CRUSOE_BASE_URL": "https://example.invalid/v1"}):
            llm = CrusoeLLM(api_key=FAKE_KEY, dotenv_path=None, client=object())
        self.assertEqual(llm.rpm, 12.0)
        self.assertEqual(llm.base_url, "https://example.invalid/v1")

    def test_defaults(self) -> None:
        with mock.patch.dict(os.environ):
            for name in ("CRUSOE_RPM", "CRUSOE_BASE_URL", "CRUSOE_MAX_RETRIES", "CRUSOE_MAX_CONCURRENT", "CRUSOE_SLOT_DIR"):
                os.environ.pop(name, None)
            llm = CrusoeLLM(api_key=FAKE_KEY, dotenv_path=None, client=object())
        self.assertEqual(llm.rpm, 25.0)
        self.assertEqual(llm.base_url, "https://api.inference.crusoecloud.com/v1")
        self.assertEqual(llm.max_retries, 8)
        self.assertEqual(llm.max_concurrent, 3)
        self.assertEqual(llm.slot_dir, Path("/tmp/crusoe-slots"))

    def test_retry_and_slot_settings_from_env(self) -> None:
        env = {"CRUSOE_MAX_RETRIES": "2", "CRUSOE_MAX_CONCURRENT": "0", "CRUSOE_SLOT_DIR": "/somewhere/else"}
        with mock.patch.dict(os.environ, env):
            llm = CrusoeLLM(api_key=FAKE_KEY, dotenv_path=None, client=object())
            explicit = CrusoeLLM(api_key=FAKE_KEY, dotenv_path=None, client=object(), max_retries=5)
        self.assertEqual((llm.max_retries, llm.max_concurrent, llm.slot_dir), (2, 0, Path("/somewhere/else")))
        self.assertEqual(explicit.max_retries, 5)

    def test_bad_integer_names_the_variable(self) -> None:
        with mock.patch.dict(os.environ, {"CRUSOE_MAX_CONCURRENT": "three"}):
            with self.assertRaises(ValueError) as ctx:
                CrusoeLLM(api_key=FAKE_KEY, dotenv_path=None, client=object())
        self.assertIn("CRUSOE_MAX_CONCURRENT", str(ctx.exception))


class ChatWithFakeClientTest(unittest.TestCase):
    def test_returns_reply_with_usage_cost_and_ledger(self) -> None:
        llm, _ = make_llm([completion("<think>hmm</think>Hello", prompt=1000, out=50, cached=600)])
        reply = llm.chat(NANO, [{"role": "user", "content": "hi"}], role="worker")
        self.assertIsInstance(reply, Reply)
        self.assertEqual(reply.text, "Hello")
        self.assertEqual((reply.prompt_tokens, reply.completion_tokens, reply.cached_tokens), (1000, 50, 600))
        self.assertAlmostEqual(reply.cost_usd, cost_usd(NANO, 1000, 50, 600))
        summary = llm.ledger.summary()
        self.assertEqual(summary["total"]["calls"], 1)
        self.assertEqual(summary["by_role"]["worker"]["cached_tokens"], 600)
        self.assertEqual(summary["by_model"][NANO]["prompt_tokens"], 1000)

    def test_thinking_kwarg_reaches_extra_body_and_none_sends_nothing(self) -> None:
        llm, completions = make_llm([completion("{}"), completion("{}"), completion("{}")])
        llm.chat(NANO, [{"role": "user", "content": "hi"}], thinking=False)
        llm.chat(NANO, [{"role": "user", "content": "hi"}], thinking=True)
        llm.chat(NANO, [{"role": "user", "content": "hi"}])
        self.assertEqual(completions.calls[0]["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(completions.calls[1]["extra_body"], {"chat_template_kwargs": {"enable_thinking": True}})
        self.assertNotIn("extra_body", completions.calls[2])

    def test_falls_back_to_reasoning_content(self) -> None:
        llm, _ = make_llm([completion("", reasoning="the answer is 4")])
        self.assertEqual(llm.chat(NANO, [{"role": "user", "content": "2+2"}]).text, "the answer is 4")

    def test_retries_429_and_503_then_succeeds(self) -> None:
        llm, completions = make_llm([api_error(429), api_error(503), completion("ok")])
        reply = llm.chat(NANO, [{"role": "user", "content": "hi"}], role="r")
        self.assertEqual(reply.text, "ok")
        self.assertEqual(len(completions.calls), 3)
        self.assertEqual(llm.ledger.summary()["total"]["retries"], 2)

    def test_does_not_retry_auth_error(self) -> None:
        import openai

        llm, completions = make_llm([api_error(401), completion("never")])
        with self.assertRaises(openai.AuthenticationError):
            llm.chat(NANO, [{"role": "user", "content": "hi"}])
        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(llm.ledger.summary()["total"]["errors"], 1)

    def test_gives_up_after_max_retries(self) -> None:
        import openai

        llm, completions = make_llm([api_error(429)] * 9)
        with self.assertRaises(openai.RateLimitError):
            llm.chat(NANO, [{"role": "user", "content": "hi"}])
        self.assertEqual(len(completions.calls), 9)

    def test_json_mode_retries_without_response_format_on_400(self) -> None:
        llm, completions = make_llm([api_error(400), completion('{"ok": true}'), completion('{"ok": 1}')])
        reply = llm.chat(NANO, [{"role": "user", "content": "json"}], json_mode=True)
        self.assertEqual(extract_json(reply.text), {"ok": True})
        self.assertIn("response_format", completions.calls[0])
        self.assertNotIn("response_format", completions.calls[1])
        llm.chat(NANO, [{"role": "user", "content": "json"}], json_mode=True)
        self.assertNotIn("response_format", completions.calls[2])

    def test_non_json_400_is_not_retried(self) -> None:
        import openai

        llm, completions = make_llm([api_error(400), completion("never")])
        with self.assertRaises(openai.BadRequestError):
            llm.chat(NANO, [{"role": "user", "content": "hi"}])
        self.assertEqual(len(completions.calls), 1)

    def test_stream_measures_ttft_and_usage(self) -> None:
        usage = SimpleNamespace(prompt_tokens=4000, completion_tokens=3, prompt_tokens_details=SimpleNamespace(cached_tokens=3968))
        stream = [chunk(reasoning="thinking"), chunk(content="</think>4"), chunk(content="2"), chunk(usage=usage)]
        llm, completions = make_llm([iter(stream)])
        reply = llm.chat_stream_ttft(NANO, [{"role": "user", "content": "q"}], role="probe")
        self.assertEqual(reply.text, "42")
        self.assertIsNotNone(reply.ttft_s)
        self.assertEqual(reply.cached_tokens, 3968)
        self.assertTrue(completions.calls[0]["stream"])
        self.assertEqual(completions.calls[0]["stream_options"], {"include_usage": True})


HI = [{"role": "user", "content": "hi"}]


class RateLimitBackoffTest(unittest.TestCase):
    def test_429_then_success_returns_and_counts_the_retry(self) -> None:
        sleeps: list[float] = []
        llm, completions = make_llm([api_error(429), completion("ok")], sleep=sleeps.append)
        reply = llm.chat(SUPER, HI, role="worker")
        self.assertEqual(reply.text, "ok")
        self.assertEqual(len(completions.calls), 2)
        summary = llm.ledger.summary()
        self.assertEqual(summary["total"]["retries"], 1)
        self.assertEqual(summary["by_role"]["worker"]["retries"], 1)
        self.assertEqual(summary["total"]["errors"], 0)
        self.assertEqual(len(sleeps), 1)
        self.assertTrue(0.0 <= sleeps[0] <= 2.0, sleeps)

    def _sleeps_with_rng_at(self, end: str, outcomes: list[Any]) -> list[float]:
        """Run one chat with every jitter draw pinned to the low or high end of its range."""
        sleeps: list[float] = []
        llm, _ = make_llm(outcomes, sleep=sleeps.append)
        pick = (lambda low, high: low) if end == "low" else (lambda low, high: high)
        with mock.patch("team.crusoe.random.uniform", side_effect=pick):
            self.assertEqual(llm.chat(SUPER, HI).text, "ok")
        return sleeps

    def test_retry_after_seconds_and_ms_are_a_floor(self) -> None:
        def outcomes() -> list[Any]:
            return [api_error(429, {"retry-after": "7"}), api_error(429, {"retry-after-ms": "2500"}), completion("ok")]

        # No jitter: exactly what the server asked for.
        self.assertEqual(self._sleeps_with_rng_at("low", outcomes()), [7.0, 2.5])
        # Max jitter: Retry-After + 1s, or the attempt's backoff (2s, then 4s) when that is longer.
        self.assertEqual(self._sleeps_with_rng_at("high", outcomes()), [8.0, 4.0])

    def test_long_retry_after_is_capped_at_30s(self) -> None:
        sleeps: list[float] = []
        llm, _ = make_llm([api_error(429, {"retry-after": "600"}), completion("ok")], sleep=sleeps.append)
        llm.chat(SUPER, HI)
        self.assertEqual(sleeps, [30.0])

    def test_backoff_doubles_from_2s_to_a_30s_cap(self) -> None:
        sleeps: list[float] = []
        llm, _ = make_llm([api_error(429)] * 8 + [completion("ok")], sleep=sleeps.append)
        with mock.patch("team.crusoe.random.uniform", side_effect=lambda low, high: high):
            llm.chat(SUPER, HI)
        self.assertEqual(sleeps, [2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0, 30.0])
        self.assertEqual(llm.ledger.summary()["total"]["retries"], 8)

    def test_retries_exhausted_raises_and_records_error(self) -> None:
        import openai

        sleeps: list[float] = []
        llm, completions = make_llm([api_error(429)] * 10, sleep=sleeps.append, CRUSOE_MAX_RETRIES="3")
        with self.assertRaises(openai.RateLimitError):
            llm.chat(SUPER, HI, role="worker")
        self.assertEqual(len(completions.calls), 4)
        self.assertEqual(len(sleeps), 3)
        self.assertEqual(llm.ledger.summary()["total"]["errors"], 1)
        self.assertEqual(llm.ledger.summary()["total"]["calls"], 0)


class SlotSemaphoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crusoe-slots-test-")
        self.slot_dir = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _hammer(self, llms: list[CrusoeLLM], threads: int = 8, calls_each: int = 2) -> None:
        errors: list[BaseException] = []

        def worker(llm: CrusoeLLM) -> None:
            try:
                for _ in range(calls_each):
                    llm.chat(SUPER, HI)
            except BaseException as exc:  # surfaced below; a thread cannot fail the test itself
                errors.append(exc)

        pool = [threading.Thread(target=worker, args=(llms[i % len(llms)],)) for i in range(threads)]
        for t in pool:
            t.start()
        for t in pool:
            t.join(timeout=30)
        self.assertFalse(any(t.is_alive() for t in pool), "a worker hung waiting for a slot")
        self.assertEqual(errors, [])

    def test_eight_threads_across_two_clients_never_exceed_three_in_flight(self) -> None:
        # Two CrusoeLLM objects stand in for two processes: they share nothing but the slot directory.
        server = SlowCompletions(delay_s=0.05)
        llms = [make_llm([], completions=server, CRUSOE_SLOT_DIR=self.slot_dir, CRUSOE_MAX_CONCURRENT="3")[0] for _ in range(2)]
        self._hammer(llms)
        self.assertEqual(server.calls, 16)
        self.assertEqual(server.peak, 3)
        self.assertEqual(slot_files_free(self.slot_dir), (3, 3))

    def test_zero_disables_the_cap(self) -> None:
        server = SlowCompletions(delay_s=0.05)
        llm, _ = make_llm([], completions=server, CRUSOE_SLOT_DIR=self.slot_dir, CRUSOE_MAX_CONCURRENT="0")
        self._hammer([llm])
        self.assertGreater(server.peak, 3)
        self.assertEqual(slot_files_free(self.slot_dir), (0, 0))

    def test_slot_released_after_exception_and_not_held_during_backoff(self) -> None:
        import openai

        free_during_sleep: list[tuple[int, int]] = []
        llm, _ = make_llm(
            [api_error(429), completion("ok"), api_error(401), completion("after")],
            sleep=lambda s: free_during_sleep.append(slot_files_free(self.slot_dir)),
            CRUSOE_SLOT_DIR=self.slot_dir,
            CRUSOE_MAX_CONCURRENT="1",
        )
        self.assertEqual(llm.chat(SUPER, HI).text, "ok")
        self.assertEqual(free_during_sleep, [(1, 1)])
        with self.assertRaises(openai.AuthenticationError):
            llm.chat(SUPER, HI)
        self.assertEqual(slot_files_free(self.slot_dir), (1, 1))
        self.assertEqual(llm.chat(SUPER, HI).text, "after")

    def test_slot_held_by_another_process_is_freed_when_it_dies(self) -> None:
        child_code = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from team.crusoe import SlotSemaphore\n"
            "hold = SlotSemaphore(Path(sys.argv[2]), 1).hold()\n"
            "hold.__enter__()\n"
            "print('held', flush=True)\n"
            "time.sleep(60)\n"
        )
        child = subprocess.Popen(
            [sys.executable, "-c", child_code, str(REPO_ROOT), self.slot_dir],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(child.stdout.readline().strip(), "held")
            self.assertEqual(slot_files_free(self.slot_dir), (0, 1))
        finally:
            child.kill()
            child.wait(timeout=10)
            child.stdout.close()
        self.assertEqual(slot_files_free(self.slot_dir), (1, 1))
        with SlotSemaphore(Path(self.slot_dir), 1).hold():
            self.assertEqual(slot_files_free(self.slot_dir), (0, 1))


if __name__ == "__main__":
    unittest.main()
