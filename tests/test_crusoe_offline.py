"""Offline tests for `team.crusoe`: no network, no live key. A fake OpenAI client stands in
for the server so retry, JSON-mode fallback, usage and streaming paths are exercised."""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from team.crusoe import (  # noqa: E402
    NANO,
    SUPER,
    CrusoeLLM,
    MissingKeyError,
    TokenBucket,
    cost_usd,
    extract_json,
    strip_think,
)
from team.types import Reply  # noqa: E402

FAKE_KEY = "test-key-not-real"


def completion(content: str | None, reasoning: str | None = None, prompt: int = 100, out: int = 10, cached: int | None = None) -> Any:
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=out, prompt_tokens_details=details)
    message = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def chunk(content: str | None = None, reasoning: str | None = None, usage: Any = None) -> Any:
    choices = [] if usage is not None else [SimpleNamespace(delta=SimpleNamespace(content=content, reasoning_content=reasoning))]
    return SimpleNamespace(choices=choices, usage=usage)


def api_error(status: int) -> Exception:
    import openai

    try:
        import httpx2 as httpx
    except ImportError:
        import httpx  # type: ignore[no-redef]
    classes = {400: openai.BadRequestError, 401: openai.AuthenticationError, 429: openai.RateLimitError, 503: openai.InternalServerError}
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return classes[status](f"status {status}", response=httpx.Response(status, request=request), body=None)


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


def make_llm(outcomes: list[Any]) -> tuple[CrusoeLLM, FakeCompletions]:
    completions = FakeCompletions(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = CrusoeLLM(api_key=FAKE_KEY, rpm=6000, dotenv_path=None, client=client, sleep=lambda s: None)
    return llm, completions


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
            os.environ.pop("CRUSOE_RPM", None)
            os.environ.pop("CRUSOE_BASE_URL", None)
            llm = CrusoeLLM(api_key=FAKE_KEY, dotenv_path=None, client=object())
        self.assertEqual(llm.rpm, 25.0)
        self.assertEqual(llm.base_url, "https://api.inference.crusoecloud.com/v1")


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

        llm, completions = make_llm([api_error(429)] * 6)
        with self.assertRaises(openai.RateLimitError):
            llm.chat(NANO, [{"role": "user", "content": "hi"}])
        self.assertEqual(len(completions.calls), 6)

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


if __name__ == "__main__":
    unittest.main()
