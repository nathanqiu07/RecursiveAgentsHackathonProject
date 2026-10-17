"""Optional Codex OAuth transport for Library-owned planning and novelty review.

Install the sibling agent-harness package explicitly to enable this backend.
The harness's isolated Codex planner owns the existing CLI sign-in; this adapter
does not read authentication files, tokens, or provider keys. All scientific and
novelty validation remains in LLMPlanner and the Library queue.
"""

import math

from .planner import LLMPlanner, PlannerError


class CodexPlanner(LLMPlanner):
    def __init__(self, model=None, *, max_paper_chars=100_000, timeout_seconds=90):
        if (
            isinstance(max_paper_chars, bool)
            or not isinstance(max_paper_chars, int)
            or max_paper_chars <= 0
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 90
        ):
            raise PlannerError("invalid_configuration")
        try:
            from agent_harness.codex_model import CodexModel
        except ImportError:
            raise PlannerError("codex_backend_unavailable") from None
        try:
            self._codex = CodexModel(model, timeout=timeout_seconds)
        except Exception:
            raise PlannerError("codex_backend_unavailable") from None
        self.model = self._codex.model
        self.max_paper_chars = max_paper_chars
        self.timeout_seconds = timeout_seconds
        self._closed = False

    def close(self):
        if not self._closed:
            self._closed = True
            self._codex.close()

    def _complete_json(self, system: str, user: dict) -> dict:
        if self._closed:
            raise PlannerError("closed")
        try:
            result = self._codex.complete_json(system, user)
        except Exception as error:
            code = (
                "timeout" if getattr(error, "code", None) == "codex_timeout" else "provider_error"
            )
            raise PlannerError(code) from None
        if type(result) is not dict:
            raise PlannerError("invalid_response")
        return result
