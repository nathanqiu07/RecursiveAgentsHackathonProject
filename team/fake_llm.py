"""Deterministic stand-in for an LLM backend, so the brain and evolver run offline in tests."""
from __future__ import annotations

import threading
from typing import Callable

from team.types import Reply

Responder = Callable[[str, list[dict[str, str]]], str]


class ScriptedLLM:
    """Replays scripted replies per role and records every call for assertions.

    With a dict, each role pops its list in order and repeats the last reply once exhausted;
    a role with no script gets `default`. With a callable, it is called as fn(role, messages).
    """

    def __init__(self, responses_by_role: dict[str, list[str]] | Responder, default: str = "ok"):
        self._script = responses_by_role
        self._default = default
        self._next: dict[str, int] = {}
        self._lock = threading.Lock()
        self.calls: list[dict] = []

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
        with self._lock:
            if callable(self._script):
                text = self._script(role, messages)
            else:
                queue = self._script.get(role) or []
                i = self._next.get(role, 0)
                self._next[role] = i + 1
                text = queue[min(i, len(queue) - 1)] if queue else self._default
            self.calls.append({"model": model, "role": role, "messages": messages,
                               "temperature": temperature, "json_mode": json_mode, "thinking": thinking,
                               "max_tokens": max_tokens, "reply": text})
        prompt_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        return Reply(text=text, model=model, prompt_tokens=prompt_tokens,
                     completion_tokens=len(text) // 4)

    def calls_for(self, role: str) -> list[dict]:
        return [c for c in self.calls if c["role"] == role]
