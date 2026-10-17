"""Crusoe Managed Inference client (OpenAI-compatible) implementing `team.types.LLM`.

One `CrusoeLLM` is shared by every agent thread. It rate-limits client-side per model
(the documented free-tier cap is 30 RPM per model, per project), retries transient
failures with backoff, strips reasoning traces, and keeps a ledger so every run can
report its token bill. The limiter is per process: teammates sharing one key from
several machines must split `CRUSOE_RPM` between them.

`openai` and `python-dotenv` are imported lazily so the pure helpers here
(`extract_json`, `strip_think`, `cost_usd`, `TokenBucket`) work under a bare Python.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from team.types import Reply

T = TypeVar("T")

NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
LIGHTNING = "nvidia/Nemotron-3.5-Lightning-30B-A3B"

DEFAULT_BASE_URL = "https://api.inference.crusoecloud.com/v1"
DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
DEFAULT_RPM = 25.0
MAX_RETRIES = 5
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
RETRY_AFTER_CAP_S = 60.0

# USD per 1M tokens (input, output), read from https://www.crusoe.ai/cloud/pricing on 2026-10-17.
PRICES: dict[str, tuple[float, float]] = {
    NANO: (0.05, 0.20),
    SUPER: (0.30, 2.40),
    LIGHTNING: (0.05, 0.20),
}
# USD per 1M cached prompt tokens, same page; these bill instead of the input rate.
CACHED_PRICES: dict[str, float] = {
    NANO: 0.03,
    SUPER: 0.15,
    LIGHTNING: 0.03,
}

RATE_LIMIT_HEADER_PREFIXES = ("x-ratelimit", "retry-after")


class MissingKeyError(RuntimeError):
    """No Crusoe API key configured; the message says how to provide one."""


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    """Price one call. Cached prompt tokens bill at the cached rate; unknown models cost 0."""
    if model not in PRICES:
        return 0.0
    in_price, out_price = PRICES[model]
    cached = max(0, min(cached_tokens, prompt_tokens))
    cached_price = CACHED_PRICES.get(model, in_price)
    return ((prompt_tokens - cached) * in_price + cached * cached_price + completion_tokens * out_price) / 1_000_000


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str | None) -> str:
    """Drop reasoning traces so callers see only the answer.

    Covers a closed block, a dangling `</think>` (the chat template opened the tag
    inside the prompt) and an unclosed `<think>` (output cut off mid-thought).
    """
    text = _THINK_BLOCK.sub("", text or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


_FENCE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | list:
    """Parse the first JSON object or array in model output.

    Models wrap JSON in code fences or lead with prose even in JSON mode, so try the
    whole text, then each fenced block, then every `{` or `[` position in turn.
    """
    text = strip_think(text) or (text or "")
    candidates = [text.strip()] + [m.group(1).strip() for m in _FENCE.finditer(text)]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text, i)
        except ValueError:
            continue
        return value
    raise ValueError(f"no JSON object or array in model output: {text[:200]!r}")


class TokenBucket:
    """Thread-safe token bucket. Callers reserve a slot under the lock and sleep outside it,
    so concurrent callers queue at even spacing instead of stampeding after a refill."""

    def __init__(
        self,
        rate_per_min: float,
        capacity: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_min <= 0:
            raise ValueError("rate_per_min must be positive")
        self.rate_per_s = rate_per_min / 60.0
        self.capacity = max(1.0, float(capacity))
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a request may be sent; returns the seconds waited."""
        with self._lock:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate_per_s)
            self._last = now
            self._tokens -= 1.0
            wait = 0.0 if self._tokens >= 0 else -self._tokens / self.rate_per_s
        if wait > 0:
            self._sleep(wait)
        return wait


@dataclass
class Tally:
    calls: int = 0
    errors: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


class UsageLedger:
    """Thread-safe running totals, overall and per role and model, so each run reports its bill."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = Tally()
        self._by_role: dict[str, Tally] = {}
        self._by_model: dict[str, Tally] = {}

    def _tallies(self, model: str, role: str) -> tuple[Tally, Tally, Tally]:
        return (
            self._total,
            self._by_role.setdefault(role or "-", Tally()),
            self._by_model.setdefault(model, Tally()),
        )

    def record(self, reply: Reply, role: str = "", retries: int = 0) -> None:
        with self._lock:
            for t in self._tallies(reply.model, role):
                t.calls += 1
                t.retries += retries
                t.prompt_tokens += reply.prompt_tokens
                t.completion_tokens += reply.completion_tokens
                t.cached_tokens += reply.cached_tokens
                t.cost_usd += reply.cost_usd
                t.latency_s += reply.latency_s

    def record_error(self, model: str, role: str = "") -> None:
        with self._lock:
            for t in self._tallies(model, role):
                t.errors += 1

    def summary(self) -> dict[str, Any]:
        def dump(t: Tally) -> dict[str, Any]:
            d = asdict(t)
            d["cost_usd"] = round(t.cost_usd, 6)
            d["latency_s"] = round(t.latency_s, 3)
            return d

        with self._lock:
            return {
                "total": dump(self._total),
                "by_role": {k: dump(v) for k, v in sorted(self._by_role.items())},
                "by_model": {k: dump(v) for k, v in sorted(self._by_model.items())},
            }


def _status_code(exc: BaseException) -> int | None:
    import openai

    return exc.status_code if isinstance(exc, openai.APIStatusError) else None


def _is_retryable(exc: BaseException) -> bool:
    """429, 408 and 5xx are transient; other 4xx mean the request itself is wrong."""
    import openai

    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return True
    status = _status_code(exc)
    if status is not None:
        return status in (408, 429) or status >= 500
    # A dropped connection mid-stream surfaces as the HTTP library's own error.
    return any(c.__name__ in ("TransportError", "TimeoutException") for c in type(exc).__mro__)


def _retry_after_s(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _reasoning_of(obj: Any) -> str:
    if obj is None:
        return ""
    return getattr(obj, "reasoning_content", None) or getattr(obj, "reasoning", None) or ""


def _usage_counts(usage: Any) -> tuple[int, int, int]:
    if usage is None:
        return 0, 0, 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) or 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0), int(cached)


class CrusoeLLM:
    """`team.types.LLM` backed by Crusoe Managed Inference. Safe to share across threads."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        rpm: float | None = None,
        burst: int = 3,
        timeout_s: float = 120.0,
        max_retries: int = MAX_RETRIES,
        extra_body: dict[str, Any] | None = None,
        dotenv_path: Path | None = DOTENV_PATH,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """`burst` 3 at 25 RPM admits at most 28 requests in any minute, under the 30 RPM cap.
        `extra_body` is passed through on every request (e.g. server-specific reasoning switches).
        `dotenv_path=None` skips the dotenv file; `client` injects a fake for offline tests."""
        if dotenv_path is not None:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path)
        key = (api_key if api_key is not None else os.environ.get("CRUSOE_API_KEY", "")).strip()
        if not key:
            raise MissingKeyError(
                "CRUSOE_API_KEY is not set. Create a .env file at the repo root "
                f"({DOTENV_PATH}) with one line: CRUSOE_API_KEY=<your key>, "
                "or run: export CRUSOE_API_KEY=<your key>"
            )
        self.base_url = base_url or os.environ.get("CRUSOE_BASE_URL") or DEFAULT_BASE_URL
        self.rpm = float(rpm if rpm is not None else os.environ.get("CRUSOE_RPM") or DEFAULT_RPM)
        self.burst = burst
        self.max_retries = max_retries
        self.extra_body = extra_body
        self._sleep = sleep
        if client is None:
            import openai

            client = openai.OpenAI(api_key=key, base_url=self.base_url, timeout=timeout_s, max_retries=0)
        self.client = client
        self.ledger = UsageLedger()
        self._lock = threading.Lock()
        self._buckets: dict[str, TokenBucket] = {}
        self._no_json_format: set[str] = set()

    def _bucket(self, model: str) -> TokenBucket:
        """Crusoe counts RPM per model, so each model gets its own bucket."""
        with self._lock:
            if model not in self._buckets:
                self._buckets[model] = TokenBucket(self.rpm, self.burst)
            return self._buckets[model]

    def accepts_json_format(self, model: str) -> bool:
        """False once the server has rejected `response_format` for this model."""
        with self._lock:
            return model not in self._no_json_format

    def _backoff_s(self, attempt: int, exc: BaseException) -> float:
        """1s, 2s, 4s ... capped at 30s, plus up to 50% jitter; never shorter than Retry-After."""
        base = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2**attempt)
        delay = base + random.uniform(0, base / 2)
        retry_after = _retry_after_s(exc)
        if retry_after is not None:
            delay = max(delay, min(retry_after, RETRY_AFTER_CAP_S))
        return delay

    def _with_retries(self, model: str, fn: Callable[[], T]) -> tuple[T, int]:
        """Run `fn` under the model's rate limiter, retrying transient failures.
        Returns the result and how many retries it took."""
        for attempt in range(self.max_retries + 1):
            self._bucket(model).acquire()
            try:
                return fn(), attempt
            except Exception as exc:
                if attempt >= self.max_retries or not _is_retryable(exc):
                    raise
                self._sleep(self._backoff_s(attempt, exc))
        raise AssertionError("unreachable")

    def _base_kwargs(self, model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs

    def _timed_create(self, kwargs: dict[str, Any]) -> tuple[Any, float]:
        start = time.monotonic()
        resp = self.client.chat.completions.create(**kwargs)
        return resp, time.monotonic() - start

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
        role: str = "",
        thinking: bool | None = None,
    ) -> Reply:
        """One completion. `latency_s` is the successful attempt only, excluding limiter waits and backoff.
        `thinking` None sends nothing; True/False sets chat_template_kwargs.enable_thinking (Nemotron)."""
        kwargs = self._base_kwargs(model, messages, temperature, max_tokens)
        if thinking is not None:
            extra = dict(kwargs.get("extra_body") or {})
            extra["chat_template_kwargs"] = {**(extra.get("chat_template_kwargs") or {}), "enable_thinking": bool(thinking)}
            kwargs["extra_body"] = extra
        want_format = json_mode and self.accepts_json_format(model)
        if want_format:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            try:
                (resp, latency), retries = self._with_retries(model, lambda: self._timed_create(kwargs))
            except Exception as exc:
                if not (want_format and _status_code(exc) in (400, 422)):
                    raise
                with self._lock:
                    self._no_json_format.add(model)
                kwargs.pop("response_format")
                (resp, latency), retries = self._with_retries(model, lambda: self._timed_create(kwargs))
                retries += 1
        except Exception:
            self.ledger.record_error(model, role)
            raise
        message = resp.choices[0].message if resp.choices else None
        text = strip_think(getattr(message, "content", None))
        if not text:
            text = _reasoning_of(message).strip()
        prompt, completion, cached = _usage_counts(resp.usage)
        reply = Reply(
            text=text,
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            latency_s=latency,
            cost_usd=cost_usd(model, prompt, completion, cached),
        )
        self.ledger.record(reply, role, retries)
        return reply

    def chat_stream_ttft(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        role: str = "",
    ) -> Reply:
        """Streamed completion that records time-to-first-token, which is prefill time:
        the number prefix caching should shrink. Reasoning deltas count as first tokens."""
        kwargs = self._base_kwargs(model, messages, temperature, max_tokens)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        def run() -> tuple[str, str, Any, float, float | None]:
            start = time.monotonic()
            ttft: float | None = None
            content: list[str] = []
            reasoning: list[str] = []
            usage = None
            for chunk in self.client.chat.completions.create(**kwargs):
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                for choice in getattr(chunk, "choices", None) or []:
                    delta = getattr(choice, "delta", None)
                    piece = getattr(delta, "content", None) or ""
                    thought = _reasoning_of(delta)
                    if ttft is None and (piece or thought):
                        ttft = time.monotonic() - start
                    content.append(piece)
                    reasoning.append(thought)
            return "".join(content), "".join(reasoning), usage, time.monotonic() - start, ttft

        try:
            (content, reasoning, usage, latency, ttft), retries = self._with_retries(model, run)
        except Exception:
            self.ledger.record_error(model, role)
            raise
        text = strip_think(content) or reasoning.strip()
        prompt, completion, cached = _usage_counts(usage)
        reply = Reply(
            text=text,
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cached_tokens=cached,
            latency_s=latency,
            ttft_s=ttft,
            cost_usd=cost_usd(model, prompt, completion, cached),
        )
        self.ledger.record(reply, role, retries)
        return reply

    def list_models(self) -> list[str]:
        """Model ids the endpoint serves, to confirm the constants above before a run."""
        models, _ = self._with_retries("__models__", lambda: list(self.client.models.list()))
        return sorted(m.id for m in models)

    def rate_limit_headers(self, model: str, role: str = "probe") -> dict[str, str]:
        """Send a tiny request and return its rate-limit headers (x-ratelimit-*, retry-after),
        so a run can see remaining quota instead of discovering it through 429s."""
        kwargs = self._base_kwargs(model, [{"role": "user", "content": "Reply with: ok"}], 0.0, 16)

        def run() -> tuple[Any, Any, float]:
            start = time.monotonic()
            raw = self.client.chat.completions.with_raw_response.create(**kwargs)
            return raw, raw.parse(), time.monotonic() - start

        try:
            (raw, resp, latency), retries = self._with_retries(model, run)
        except Exception:
            self.ledger.record_error(model, role)
            raise
        prompt, completion, cached = _usage_counts(resp.usage)
        self.ledger.record(
            Reply(
                text="",
                model=model,
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_tokens=cached,
                latency_s=latency,
                cost_usd=cost_usd(model, prompt, completion, cached),
            ),
            role,
            retries,
        )
        return {k.lower(): v for k, v in raw.headers.items() if k.lower().startswith(RATE_LIMIT_HEADER_PREFIXES)}
