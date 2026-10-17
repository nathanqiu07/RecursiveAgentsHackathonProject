"""`run_team` for inference serving cost: answer every question of a task, graded exactly in code.

Each question type is a role (prompts/playbooks/models/thinking are keyed by type), so the Reviewer's
cost_per_task_by_role is cost per question type and a config_set can route one type to a cheaper model.
Knobs (version.config): models {qtype: model id}, default_model, thinking {qtype: bool}, max_tokens,
max_tokens_by_type {qtype: int}, batch_size (questions of one type per call, 1..12), temperature.
score = accuracy; passed = accuracy >= PASS_ACCURACY. Cost is Crusoe's billed tokens x team.crusoe.PRICES.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.inference_opt.workload import QTYPES, REFERENCE, coerce, grade  # noqa: E402
from team.crusoe import SUPER, extract_json  # noqa: E402
from team.types import Reply, RunResult, Task  # noqa: E402

PASS_ACCURACY = 0.9
BATCH_RANGE = (1, 12)
TOKENS_RANGE = (256, 32768)
SLOW_CALL_S = 20.0  # a type whose mean call latency exceeds this gets a `slow:<type>` failure line for the Reviewer
INNER_WORKERS = 4

DEFAULT_PROMPT = ("You are a data-center operations analyst. Compute each answer exactly from the facts given, "
                  "using the reference rules. Give plain numbers (no units).")
KNOB_NOTES = ("Roles are question types. models/thinking/max_tokens_by_type are keyed by question type; "
              "batch_size = questions of one type answered in one JSON call (1..12): fewer calls and less repeated "
              "prompt, but one bad reply loses the whole batch. A type's failures are wrong:<type>, "
              "bad_json:<type>, truncated:<type> (hit max_tokens), slow:<type> (mean call latency > "
              f"{SLOW_CALL_S:g} s).")


def v0_config(max_tokens: int = 4000, thinking: bool = True, model: str = SUPER) -> dict[str, Any]:
    """The expensive default a cautious team starts with: one model for everything, reasoning on, one question per call."""
    return {"default_model": model, "models": {}, "thinking": {t: thinking for t in QTYPES},
            "max_tokens": max_tokens, "batch_size": 1, "temperature": 0.0, "knob_notes": KNOB_NOTES}


def v0_prompts() -> dict[str, str]:
    return {t: DEFAULT_PROMPT for t in QTYPES}


def _int_knob(value: Any, default: int, lo: int, hi: int) -> int:
    """A bad value falls back to the default, so one bad config_set fails its Gate, not the run."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        value = default
    return int(min(max(int(value), lo), hi))


def system_prompt(version: Any, qtype: str, brain_text: str) -> str:
    parts = [(getattr(version, "prompts", {}) or {}).get(qtype) or DEFAULT_PROMPT]
    rules = (getattr(version, "playbooks", {}) or {}).get(qtype) or []
    if rules:
        parts.append("Playbook rules:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))
    if brain_text.strip():
        parts.append("Shared memory from earlier runs (verified evidence):\n" + brain_text.strip())
    parts.append('Reply with JSON only: {"answers": {"<question id>": <number>, ...}}, one number per question id.')
    return "\n\n".join(parts)


def user_prompt(task: Task, group: list[dict[str, Any]]) -> str:
    qs = "\n".join(f"[{q['id']}] {q['text']} (answer in {q['unit']})" for q in group)
    return f"Hall {task.payload['hall']}.\n{REFERENCE}\n\nQuestions:\n{qs}"


def groups_for(questions: list[dict[str, Any]], batch: int) -> list[tuple[str, list[dict[str, Any]]]]:
    """Batches never mix types, so each call runs under its own type's model and thinking setting."""
    out = []
    for qtype in QTYPES:
        qs = [q for q in questions if q["qtype"] == qtype]
        out += [(qtype, qs[i:i + batch]) for i in range(0, len(qs), batch)]
    return out


def parse_answers(text: str) -> dict[str, Any] | None:
    try:
        obj = extract_json(text or "")
    except Exception:  # noqa: BLE001 - any parse failure is a bad_json outcome, graded below
        return None
    if isinstance(obj, dict) and isinstance(obj.get("answers"), dict):
        return obj["answers"]
    if isinstance(obj, dict) and any(str(k).startswith("q") for k in obj):
        return obj  # tolerate a flat {"q1": 3}
    if isinstance(obj, dict) and "answer" in obj:
        return {"__single__": obj["answer"]}
    return None


def run_team(version: Any, brain: Any, task: Task, llm: Any) -> RunResult:
    start = time.monotonic()
    cfg = getattr(version, "config", {}) or {}
    models = cfg.get("models") or {}
    thinking = cfg.get("thinking") or {}
    tokens_by_type = cfg.get("max_tokens_by_type") or {}
    default_model = cfg.get("default_model") or SUPER
    batch = _int_knob(cfg.get("batch_size", 1), 1, *BATCH_RANGE)
    max_tokens = _int_knob(cfg.get("max_tokens", 4000), 4000, *TOKENS_RANGE)
    temperature = float(cfg.get("temperature", 0.0))
    questions = task.payload["questions"]
    result = RunResult(task_id=task.id, version=getattr(version, "version", 0), score=0.0, passed=False, iterations=1)

    brain_texts: dict[str, str] = {}
    for qtype in QTYPES:
        text, ids = brain.read(qtype, task.text, budget_chars=800) if brain is not None else ("", [])
        brain_texts[qtype] = text
        result.brain_ids_read += [i for i in ids if i not in result.brain_ids_read]

    def call(item: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
        qtype, group = item
        cap = _int_knob(tokens_by_type.get(qtype, max_tokens), max_tokens, *TOKENS_RANGE)
        think = thinking.get(qtype)
        try:
            reply = llm.chat(models.get(qtype) or default_model,
                             [{"role": "system", "content": system_prompt(version, qtype, brain_texts[qtype])},
                              {"role": "user", "content": user_prompt(task, group)}],
                             temperature=temperature, max_tokens=cap, json_mode=False, role=qtype,
                             thinking=think if isinstance(think, bool) else None)
        except Exception as exc:  # noqa: BLE001 - e.g. Crusoe's sustained 429 after retries: an infra failure, not the team's
            return {"qtype": qtype, "ids": [q["id"] for q in group], "parsed": None, "truncated": False,
                    "reply": Reply(text="", model=models.get(qtype) or default_model),
                    "error": f"{type(exc).__name__}: {str(exc)[:80]}"}
        parsed = parse_answers(reply.text)
        if parsed is not None and "__single__" in parsed and len(group) == 1:
            parsed = {group[0]["id"]: parsed["__single__"]}
        truncated = reply.completion_tokens >= cap - 8
        return {"qtype": qtype, "ids": [q["id"] for q in group], "parsed": parsed, "truncated": truncated,
                "reply": reply}

    groups = groups_for(questions, batch)
    with ThreadPoolExecutor(max(1, min(INNER_WORKERS, len(groups)))) as pool:
        outcomes = list(pool.map(call, groups))

    answers: dict[str, Any] = {}
    per_type: dict[str, dict[str, Any]] = {t: {"n": 0, "correct": 0, "calls": 0, "cost_usd": 0.0, "tokens": 0,
                                                "latency_s": []} for t in QTYPES}
    for o in outcomes:
        r, t = o["reply"], per_type[o["qtype"]]
        result.calls += 1
        result.prompt_tokens += r.prompt_tokens
        result.completion_tokens += r.completion_tokens
        result.cost_usd += r.cost_usd
        t["calls"] += 1
        t["cost_usd"] += r.cost_usd
        t["tokens"] += r.prompt_tokens + r.completion_tokens
        if o.get("error"):
            # team.evolve drops an `infra:` task from both sides of the Gate's paired comparison.
            result.failures.append(f"infra:{o['qtype']} call failed ({o['error']})")
            continue
        t["latency_s"].append(round(r.latency_s, 3))
        if o["parsed"] is None:
            result.failures.append(f"truncated:{o['qtype']}" if o["truncated"] else f"bad_json:{o['qtype']}")
            continue
        for qid in o["ids"]:
            value = coerce(o["parsed"].get(qid))
            if value is not None:
                answers[qid] = value

    g = grade(questions, answers)
    for q in questions:
        t = per_type[q["qtype"]]
        t["n"] += 1
        t["correct"] += int(g["per_q"][q["id"]])
        if not g["per_q"][q["id"]] and q["id"] in answers:
            result.failures.append(f"wrong:{q['qtype']}")
    for qtype, t in per_type.items():
        if t["latency_s"] and statistics.mean(t["latency_s"]) > SLOW_CALL_S:
            result.failures.append(f"slow:{qtype}")
        t["cost_usd"] = round(t["cost_usd"], 8)

    latencies = [o["reply"].latency_s for o in outcomes]
    result.score = g["accuracy"]
    result.passed = g["accuracy"] >= PASS_ACCURACY - 1e-9
    result.claimed_done = bool(answers)
    result.wall_s = time.monotonic() - start
    result.solution = {"case_study": "inference_opt", "accuracy": g["accuracy"], "correct": g["correct"],
                       "n": g["n"], "cost_usd": round(result.cost_usd, 8),
                       "solution": {"seed": task.payload["seed"], "tier": task.payload["tier"], "answers": answers}}
    result.notes = json.dumps({"per_type": per_type, "latencies_s": [round(x, 3) for x in latencies],
                               "batch_size": batch, "wall_s": round(result.wall_s, 3)})
    if cfg.get("log_dir"):
        log = Path(cfg["log_dir"]) / f"{task.id}-v{result.version}.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"task": task.id, "answers": answers, "per_q": g["per_q"],
                                   "replies": [o["reply"].text[-400:] for o in outcomes]}, indent=1))
        result.log_path = str(log)
    return result
