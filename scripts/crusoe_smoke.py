"""Prove the Crusoe connection works and measure what the team design depends on:
which Nemotron models are served, JSON mode, per-call cost, whether a shared ~4k-token
prefix is cached (cached_tokens, TTFT), and the rate-limit headers.

Run with the venv: .venv/bin/python scripts/crusoe_smoke.py
Exit codes: 0 all steps ran, 1 a step failed, 2 no API key configured.
"""
from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from team.crusoe import (  # noqa: E402
    DOTENV_PATH,
    LIGHTNING,
    NANO,
    SUPER,
    CrusoeLLM,
    MissingKeyError,
    extract_json,
)
from team.types import Reply  # noqa: E402

CONSTANTS = {"NANO": NANO, "SUPER": SUPER, "LIGHTNING": LIGHTNING}
COLORS = ["red", "blue", "green", "amber", "violet", "silver", "black", "white"]
OBJECTS = ["crate", "lantern", "anchor", "compass", "barrel", "kettle", "ledger", "rope"]
PREFIX_FACTS = 280  # about 58 characters (~15 tokens) per fact, so about 4,000 tokens


def describe_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "") + f": {exc}"


def fmt(reply: Reply) -> str:
    ttft = f"ttft={reply.ttft_s:.3f}s " if reply.ttft_s is not None else ""
    return (
        f"{ttft}latency={reply.latency_s:.3f}s prompt={reply.prompt_tokens} "
        f"cached={reply.cached_tokens} completion={reply.completion_tokens} cost=${reply.cost_usd:.6f}"
    )


def synthetic_prefix(seed: int) -> str:
    """Deterministic fact list; a fresh nonce on top keeps each run's first call cold."""
    nonce = secrets.token_hex(8)
    lines = [f"Reference sheet {nonce}. Answer questions using only these facts."]
    for i in range(1, PREFIX_FACTS + 1):
        k = i * 7 + seed
        lines.append(
            f"Fact {i}: the {COLORS[k % len(COLORS)]} {OBJECTS[(k // 3) % len(OBJECTS)]} "
            f"in room {k % 97} weighs {(k * 13) % 89 + 1} kilograms."
        )
    return "\n".join(lines)


def step_models(llm: CrusoeLLM) -> bool:
    print("\n== b. models.list() ==")
    try:
        ids = llm.list_models()
    except Exception as exc:
        print(f"  FAILED: {describe_error(exc)}")
        return False
    nemotron = [m for m in ids if "nemotron" in m.lower()]
    print(f"  {len(ids)} models served, {len(nemotron)} Nemotron:")
    for m in nemotron:
        print(f"    {m}")
    lowered = {m.lower(): m for m in ids}
    for name, model_id in CONSTANTS.items():
        if model_id in ids:
            print(f"  {name:9s} {model_id}: present")
        elif model_id.lower() in lowered:
            print(f"  {name:9s} {model_id}: present only as {lowered[model_id.lower()]} (case differs)")
        else:
            print(f"  {name:9s} {model_id}: MISSING")
    return True


def step_json(llm: CrusoeLLM) -> bool:
    print("\n== c. JSON-mode call per model ==")
    ok = True
    for name, model_id in (("NANO", NANO), ("SUPER", SUPER)):
        messages = [
            {"role": "system", "content": "You reply with a single JSON object and nothing else."},
            {"role": "user", "content": f'Return exactly this JSON: {{"ok": true, "model": "{name}"}}'},
        ]
        try:
            reply = llm.chat(model_id, messages, temperature=0.0, max_tokens=1024, json_mode=True, role=f"json-{name}")
        except Exception as exc:
            print(f"  {name}: FAILED: {describe_error(exc)}")
            ok = False
            continue
        print(f"  {name}: {fmt(reply)}")
        try:
            print(f"  {name}: parsed {json.dumps(extract_json(reply.text))}")
        except ValueError:
            print(f"  {name}: could not parse JSON from: {reply.text[:300]!r}")
            ok = False
    with_format = [m for m in (NANO, SUPER) if llm.accepts_json_format(m)]
    print(f"  response_format accepted by: {with_format or 'neither (fell back to prompt-only JSON)'}")
    return ok


def step_prefix_cache(llm: CrusoeLLM) -> bool:
    print("\n== d. prefix-cache probe (NANO, streaming) ==")
    shared = synthetic_prefix(seed=0)
    control = synthetic_prefix(seed=5)
    runs: list[tuple[str, str, str]] = [
        ("shared prefix, question 1 (cold)", shared, "What does the object in Fact 17 weigh? Reply with the number only."),
        ("shared prefix, question 2 (warm?)", shared, "Which room is Fact 42 in? Reply with the number only."),
        ("control prefix, question 1", control, "What does the object in Fact 17 weigh? Reply with the number only."),
    ]
    results: list[Reply] = []
    for label, prefix, question in runs:
        messages = [{"role": "system", "content": prefix}, {"role": "user", "content": question}]
        try:
            reply = llm.chat_stream_ttft(NANO, messages, temperature=0.0, max_tokens=512, role="cache-probe")
        except Exception as exc:
            print(f"  {label}: FAILED: {describe_error(exc)}")
            return False
        results.append(reply)
        print(f"  {label}: {fmt(reply)} answer={reply.text[:40]!r}")
    cold, warm, ctrl = results
    reported = any(r.cached_tokens for r in results)
    print(f"  cached_tokens reported by server: {'yes' if reported else 'no (field absent or always 0)'}")
    if cold.ttft_s and warm.ttft_s and ctrl.ttft_s:
        print(
            f"  TTFT warm/cold = {warm.ttft_s / cold.ttft_s:.2f}, control/cold = {ctrl.ttft_s / cold.ttft_s:.2f} "
            "(single samples; queueing noise can dominate)"
        )
    return True


def step_headers(llm: CrusoeLLM) -> bool:
    print("\n== e. rate-limit headers (NANO) ==")
    try:
        headers = llm.rate_limit_headers(NANO)
    except Exception as exc:
        print(f"  FAILED: {describe_error(exc)}")
        return False
    if not headers:
        print("  no x-ratelimit-* or retry-after headers returned")
    for k, v in sorted(headers.items()):
        print(f"  {k}: {v}")
    return True


def main() -> int:
    print("== a. key ==")
    try:
        llm = CrusoeLLM()
    except MissingKeyError:
        print("  No Crusoe API key found. To fix, do one of:")
        print(f"    1. Create {DOTENV_PATH} containing one line:  CRUSOE_API_KEY=<your key>")
        print("    2. Or in your shell:  export CRUSOE_API_KEY=<your key>")
        print(f"  Then rerun:  {REPO_ROOT / '.venv/bin/python'} {Path(__file__).resolve()}")
        print("  Create a key in the Crusoe console: https://console.crusoecloud.com")
        return 2
    print(f"  key loaded; base_url={llm.base_url} rpm={llm.rpm:g} per model")
    results: dict[str, bool] = {
        "models": step_models(llm),
        "json": step_json(llm),
        "prefix_cache": step_prefix_cache(llm),
        "headers": step_headers(llm),
    }
    print("\n== f. ledger ==")
    summary: dict[str, Any] = llm.ledger.summary()
    print(json.dumps(summary, indent=2))
    failed = [k for k, ok in results.items() if not ok]
    print(f"\nRESULT: {'all steps ran' if not failed else 'failed: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
