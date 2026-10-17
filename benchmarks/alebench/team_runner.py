"""`run_team` for ALE-Bench AHC039: the solver role writes a complete Python program; only the official scorer judges it.

Each attempt sees the statement, the incumbent program and its per-case feedback (official score, the reference
greedy's score, the ratio, wall time). The run keeps the best program across `attempts`, selected on the task's
feedback seeds. Train and dev tasks are scored on those same seeds; a held-out task's feedback seeds are the public
ones, and its chosen program is then scored on 30 private seeds whose numbers never enter a prompt, a log the
model reads, or the shared brain. Task score = mean over its seeds of case score / reference greedy case score.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.alebench import scorer  # noqa: E402
from team.crusoe import SUPER  # noqa: E402
from team.types import RunResult, Task  # noqa: E402

ROLE = "solver"
MAX_TEXT = 280
FAILURE_ORDER = ("timeout", "runtime_error", "wrong_answer")

DEFAULT_SOLVER_PROMPT = (
    "You write heuristic optimization programs for an AtCoder Heuristic Contest problem. Your program is run on "
    "test cases and scored only by the contest's official scorer. Correctness first (always print a valid answer "
    "within the time limit), then solution quality.")

CONTRACT = """Contract: write ONE complete Python 3.10 program (standard library and numpy only) that reads the input
from standard input and prints the answer to standard output. Each case gets {limit:g} s of wall-clock time
including Python start-up; the program is killed at {kill:g} s and that case scores 0. Check time.time() and stop
searching by about {soft:.1f} s. Only the last polygon printed is scored, so printing a safe answer early is allowed.
Scoring here: each case is divided by the reference greedy's official score on that case (ratio 1.0 = greedy);
the task score is the mean ratio. Strong human contestants score far above the greedy."""

TASK_TEMPLATE = """{statement}

{contract}

Incumbent program (mean ratio {inc_mean:.3f} on the visible cases):
```python
{incumbent}
```
Per-case feedback on the visible cases:
{feedback}

Improve on the incumbent. Reply with one line `TECHNIQUE: <short name of your method>`, then the complete program
in a single ```python block."""

FEEDBACK_TEMPLATE = """Attempt {k} result: {outcome}
Per case:
{feedback}
Best so far: {best}.
Write an improved complete program (same contract). Fix any error first.
Reply with one line `TECHNIQUE: <short name of your method>`, then the program in a single ```python block."""


def extract_code(text: str) -> str | None:
    """Longest fenced block wins, since models sometimes add a short usage snippet."""
    blocks = re.findall(r"```(?:python|py|python3)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    blocks = [b for b in blocks if b.strip()]
    return max(blocks, key=len) if blocks else None


def extract_technique(text: str) -> str:
    m = re.search(r"^\s*\**TECHNIQUE:?\**:?\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return (m.group(1) if m else "unspecified technique")[:100]


def score_cases(code: str, seeds: list[int], time_limit: float, workers: int) -> dict[str, Any]:
    """Run and officially score `code` on `seeds`; ratios are against the reference greedy per seed."""
    ref = scorer.reference_scores(seeds)
    rows = scorer.evaluate(code, seeds, time_limit, workers)
    for r in rows:
        r["greedy"] = ref[r["seed"]]
        r["ratio"] = round(scorer.ratio(r["score"], ref[r["seed"]]), 4) if r["status"] == "ok" else 0.0
    bad = [r["status"] for r in rows if r["status"] != "ok"]
    status = next((c for c in FAILURE_ORDER if c in bad), "ok")
    return {"status": status, "mean_ratio": round(mean(r["ratio"] for r in rows), 4), "cases": rows,
            "total": sum(r["score"] for r in rows)}


def feedback_lines(cases: list[dict[str, Any]]) -> str:
    out = []
    for r in cases:
        line = f"case {r['seed']}: {r['status']}, score {r['score']} (greedy {r['greedy']}, ratio {r['ratio']:.3f}), {r['wall_s']:.2f} s"
        if r.get("error"):
            line += f" - {r['error']}"
        if r.get("stderr_tail"):
            line += f"\n  stderr tail: {r['stderr_tail'][-300:].strip()}"
        out.append(line)
    return "\n".join(out)


def system_prompt(version: Any, brain_text: str) -> str:
    """Playbook rules are numbered so reviewers and evolved prompts can cite them."""
    parts = [getattr(version, "prompts", {}).get(ROLE) or DEFAULT_SOLVER_PROMPT]
    rules = getattr(version, "playbooks", {}).get(ROLE) or []
    if rules:
        parts.append("Playbook rules:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))
    if brain_text.strip():
        parts.append("Shared memory from earlier runs (verified evidence):\n" + brain_text.strip())
    return "\n\n".join(parts)


def alebench_v0() -> Any:
    from team.evolve import TeamVersion
    return TeamVersion(version=0, parent=None, prompts={ROLE: DEFAULT_SOLVER_PROMPT}, playbooks={ROLE: []},
                       config={"models": {ROLE: SUPER}, "thinking": {ROLE: False}, "max_tokens": 8000,
                               "attempts": 2, "temperature": 0.4})


def _settings(version: Any) -> dict[str, Any]:
    cfg = getattr(version, "config", {}) or {}
    contest_tl = scorer.contest_time_limit()
    return {"model": (cfg.get("models") or {}).get(ROLE) or cfg.get("default_model") or SUPER,
            "thinking": bool((cfg.get("thinking") or {}).get(ROLE, False)),
            "max_tokens": int(cfg.get("max_tokens", 8000)), "attempts": max(1, int(cfg.get("attempts", 2))),
            "temperature": float(cfg.get("temperature", 0.4)),
            # The contest limit is a ceiling: a version may lower it, never raise it, so ratings stay comparable.
            "time_limit": min(float(cfg.get("time_limit", contest_tl)), contest_tl),
            "case_workers": int(cfg.get("case_workers", 4)), "log_dir": cfg.get("log_dir")}


def run_team(version: Any, brain: Any, task: Task, llm: Any) -> RunResult:
    start = time.monotonic()
    s = _settings(version)
    p = task.payload
    heldout = task.split == "heldout"
    feedback_seeds = list(p["feedback_seeds"])
    brain_text, brain_ids = brain.read(ROLE, task.text, budget_chars=3000) if brain is not None else ("", [])
    result = RunResult(task_id=task.id, version=getattr(version, "version", 0), score=0.0, passed=False,
                       brain_ids_read=list(brain_ids))
    incumbent = scorer.GREEDY.read_text()
    inc = score_cases(incumbent, feedback_seeds, s["time_limit"], s["case_workers"])
    contract = CONTRACT.format(limit=s["time_limit"], kill=s["time_limit"] + scorer.KILL_SLACK_S,
                               soft=0.85 * s["time_limit"])
    messages = [{"role": "system", "content": system_prompt(version, brain_text)},
                {"role": "user", "content": TASK_TEMPLATE.format(
                    statement=scorer.statement().strip(), contract=contract, inc_mean=inc["mean_ratio"],
                    incumbent=incumbent.strip(), feedback=feedback_lines(inc["cases"]))}]
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for k in range(1, s["attempts"] + 1):
        reply = llm.chat(s["model"], list(messages), temperature=s["temperature"], max_tokens=s["max_tokens"],
                         role=ROLE, thinking=s["thinking"])
        result.calls += 1
        result.prompt_tokens += reply.prompt_tokens
        result.completion_tokens += reply.completion_tokens
        result.cost_usd += reply.cost_usd
        result.iterations = k
        technique, code = extract_technique(reply.text), extract_code(reply.text)
        if code is None:
            attempt = {"k": k, "technique": technique, "status": "no_code", "mean_ratio": 0.0}
            outcome, lines = "no_code: no ```python block in the reply", "(not run)"
        else:
            ev = score_cases(code, feedback_seeds, s["time_limit"], s["case_workers"])
            attempt = {"k": k, "technique": technique, "status": ev["status"], "mean_ratio": ev["mean_ratio"],
                       "code": code}
            outcome = f"{ev['status']}, mean ratio {ev['mean_ratio']:.3f} vs greedy"
            lines = feedback_lines(ev["cases"])
        history.append({key: v for key, v in attempt.items() if key != "code"})
        if attempt["status"] != "ok":
            result.failures.append(f"{ROLE}: {attempt['status']}")
        if code is not None and (best is None or attempt["mean_ratio"] > best["mean_ratio"]):
            best = attempt
        best_text = (f"mean ratio {best['mean_ratio']:.3f} ({best['technique']})" if best else "no program yet")
        messages = messages + [{"role": "assistant", "content": reply.text},
                               {"role": "user", "content": FEEDBACK_TEMPLATE.format(
                                   k=k, outcome=outcome, feedback=lines, best=best_text)}]
    notes: dict[str, Any] = {"attempts": history, "feedback_seeds": feedback_seeds}
    if best is not None:
        final = score_cases(best["code"], list(p["seeds"]), s["time_limit"], s["case_workers"]) if heldout else None
        result.score = final["mean_ratio"] if final else best["mean_ratio"]
        result.passed = result.score >= 1.0
        result.claimed_done = True
        notes.update(technique=best["technique"], feedback_mean_ratio=best["mean_ratio"], program=best["code"],
                     total=final["total"] if final else None)
        if not heldout:
            text = (f"{scorer.TITLE}: {best['technique']} reached mean ratio {best['mean_ratio']:.3f} vs the greedy "
                    f"on {len(feedback_seeds)} {task.split} cases within {s['time_limit']:g}s")[:MAX_TEXT]
            notes["proposals"] = [{"kind": "lesson", "roles": [ROLE], "text": text, "evidence": [{
                "run_id": f"{task.id}-{result.version}", "task_id": task.id,
                "measurement": f"mean_ratio={best['mean_ratio']:.3f}"}]}]
    if result.score <= 1.0:
        result.failures.append(f"{ROLE}: no_improvement")
    result.notes = json.dumps(notes)
    result.wall_s = time.monotonic() - start
    if s["log_dir"]:
        log = Path(s["log_dir"]) / f"{task.id}-v{result.version}.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"task": task.id, "score": result.score, "attempts": history}, indent=1))
        if best is not None:
            log.with_suffix(".py").write_text(best["code"])
        result.log_path = str(log)
    return result
