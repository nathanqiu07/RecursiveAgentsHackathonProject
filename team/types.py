"""Types shared by every module in `team`. Stdlib only.

Any LLM backend is an object with a `chat` method matching `LLM` below.
`team.crusoe.CrusoeLLM` is the real one; `team.fake_llm.ScriptedLLM` is for tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Reply:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float | None = None
    cost_usd: float = 0.0


class LLM(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        json_mode: bool = False,
        role: str = "",
    ) -> Reply: ...


@dataclass
class Task:
    id: str
    split: str  # "train" | "dev" | "heldout"
    payload: dict[str, Any]
    text: str = ""  # short description, used for brain retrieval


@dataclass
class RunResult:
    task_id: str
    version: int
    score: float  # 0..1 from the independent verifier, never from an LLM
    passed: bool
    claimed_done: bool = False
    iterations: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    wall_s: float = 0.0
    failures: list[str] = field(default_factory=list)
    brain_ids_read: list[str] = field(default_factory=list)
    notes: str = ""
    log_path: str | None = None
