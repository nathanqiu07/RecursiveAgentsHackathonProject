"""`run_team` for mini KernelBench: the kernel role writes a CUDA ModelNew, a non-LLM evaluator scores it.

Each attempt sees the reference source, the incumbent kernel (best so far this run) with its evaluator
feedback, and shared-brain text. score = speedup if correct (capped at 10) else 0; passed = correct and
speedup > 1.0. `failures` holds stable categories only (compile_error, incorrect, static_reject:<rule>,
runtime_reject:<rule>, slower, suspicious, no_gpu, no_code, eval_error); richer lessons go to the brain as
notes proposals. The evaluator is injectable: `make_run_team(fake)` for tests, the default scores on a GPU
via remote.evaluate_candidate (KB_REMOTE or local CUDA) and returns no_gpu otherwise.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.kernelbench_mini import remote  # noqa: E402
from benchmarks.kernelbench_mini.static_checks import rules_summary  # noqa: E402
from team.crusoe import SUPER  # noqa: E402
from team.types import RunResult, Task  # noqa: E402

ROLE = "kernel"
SPEEDUP_CAP = 10.0
MAX_ATTEMPTS = 6  # each attempt is an nvcc build plus ~100 timed GPU runs
MAX_TEXT = 280
Evaluator = Callable[[str, str], dict]  # (problem stem, candidate source) -> evaluate.py JSON

DEFAULT_KERNEL_PROMPT = (
    "You are a CUDA performance engineer. You replace a PyTorch reference Model with ModelNew, whose forward "
    "runs your own CUDA kernels compiled with torch.utils.cpp_extension.load_inline. An independent evaluator "
    "checks exact fp32 correctness on signed random inputs and times both on a GPU; only its measurement counts. "
    "Correct first, then fast."
)

TASK_TEMPLATE = """PROBLEM {pid}: {text}

Reference (the evaluator imports Model, get_inputs and get_init_inputs from this file):
```python
{ref_src}
```

Write ONE complete Python file defining `class ModelNew(nn.Module)` with the same __init__ arguments and forward
signature as Model, returning the same outputs (same dtype and shape, fp32 within atol=rtol=1e-4).
Contract:
- The math happens in your own `__global__` CUDA kernels built with load_inline (cuda_sources + cpp_sources +
  functions=[...]) at module level. Allocate outputs with torch::empty / torch::empty_like in C++ or torch.empty in Python.
- Inputs are drawn from SIGNED distributions (torch.randn at scales 0.01 to 1), not torch.rand. Do not modify inputs.
- Parameters (e.g. conv weights) must come from the same nn layer constructed in __init__ with the same arguments,
  so seeded initialization matches; read its .weight in forward, never call the layer.
- Keep fp32 end to end. Every call must compute from its inputs; no caching between calls.
The evaluator rejects, by rule name:
{rules}

Reply with one line `TECHNIQUE: <short name of your approach>`, then the full file in a single ```python block."""

# KB_DEVICE=mps: Apple GPU via PyTorch MPS (no CUDA host; Crusoe GPU quota was zero). Same task, same evaluator rules
# for hacks; the allowed tools are PyTorch ops and Metal kernels. No hint about what to fuse or rewrite.
MPS_KERNEL_PROMPT = (
    "You are a GPU performance engineer. Target: Apple GPU via PyTorch MPS; CUDA is unavailable. You replace a "
    "PyTorch reference Model with ModelNew, written with PyTorch ops or Metal kernels via torch.mps.compile_shader. "
    "An independent evaluator checks exact fp32 correctness on signed random inputs and times both on the Apple GPU "
    "against PyTorch eager on MPS; only its measurement counts. Correct first, then fast."
)

MPS_TASK_TEMPLATE = """PROBLEM {pid}: {text}

Target: Apple GPU via PyTorch MPS (device "mps"); CUDA is unavailable, so load_inline / CUDA kernels cannot build.
Reference (the evaluator imports Model, get_inputs and get_init_inputs from this file; its speed on MPS is the baseline):
```python
{ref_src}
```

Write ONE complete Python file defining `class ModelNew(nn.Module)` with the same __init__ arguments and forward
signature as Model, returning the same outputs (same dtype and shape, fp32 within atol=rtol=1e-4).
Contract:
- Write ModelNew using PyTorch ops or Metal kernels via torch.mps.compile_shader. API: `lib =
  torch.mps.compile_shader(msl_source)` takes one string of Metal Shading Language; each `kernel void name(...)` in it
  is then called as `lib.name(*tensors_and_scalars, threads=<grid size>, group_size=<threads per group>)` (both
  optional; threads defaults to the first tensor's numel). Inputs arrive on "mps"; return outputs on "mps".
- Inputs are drawn from SIGNED distributions (torch.randn at scales 0.01 to 1), not torch.rand. Do not modify inputs.
- Parameters (e.g. conv weights) must come from the same nn layer constructed in __init__ with the same arguments,
  so seeded initialization matches.
- Keep fp32 end to end. Every call must compute from its inputs; no caching between calls.
- A forward that issues exactly the reference's own torch ops is rejected (reference_copy): it is the baseline.
The evaluator rejects, by rule name:
{rules}

Reply with one line `TECHNIQUE: <short name of your approach>`, then the full file in a single ```python block."""

FEEDBACK_TEMPLATE = """Attempt {k} evaluator result: {outcome}
Incumbent (best so far): {incumbent}
{incumbent_code}Write an improved complete file (same contract). Fix any rejection or error first, then make it faster.
Reply with one line `TECHNIQUE: <short name of your approach>`, then the full file in a single ```python block."""


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len)
    return text if "class ModelNew" in text else None


def extract_technique(text: str) -> str:
    m = re.search(r"^\s*TECHNIQUE:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return (m.group(1) if m else "unspecified technique")[:100]


def category_of(ev: dict[str, Any]) -> str:
    """One stable failure category per evaluation, or "" for a correct kernel that is faster."""
    if ev.get("error") == "no_gpu":
        return "no_gpu"
    if ev.get("error"):
        return "eval_error"
    cat = ev.get("category")
    if cat:
        return str(cat)
    if not ev.get("compiled"):
        return "compile_error"
    if not ev.get("correct"):
        return "incorrect"
    return "" if float(ev.get("speedup", 0.0)) > 1.0 else "slower"


def score_of(ev: dict[str, Any]) -> float:
    return min(float(ev.get("speedup", 0.0)), SPEEDUP_CAP) if ev.get("correct") else 0.0


def describe(ev: dict[str, Any]) -> str:
    if ev.get("error"):
        return f"{ev['error']}: {ev.get('detail', '')}"[:600]
    head = (f"compiled={ev.get('compiled')} correct={ev.get('correct')} speedup={float(ev.get('speedup', 0)):.3f}x"
            f" category={category_of(ev) or 'faster'}")
    t = ev.get("timings") or {}
    if "ref_ms_median" in t:
        head += f" (reference {t['ref_ms_median']:.4f} ms, yours {t['cand_ms_median']:.4f} ms, median of {t.get('trials')})"
    reasons = "\n".join(f"- {r}" for r in (ev.get("reasons") or [])[:6])
    return head + ("\n" + reasons[:1500] if reasons else "")


def default_prompt() -> str:
    return MPS_KERNEL_PROMPT if remote.device_kind() == "mps" else DEFAULT_KERNEL_PROMPT


def task_text(task: Task) -> str:
    device = remote.device_kind()
    template = MPS_TASK_TEMPLATE if device == "mps" else TASK_TEMPLATE
    return template.format(pid=task.id, text=task.text, ref_src=task.payload["ref_src"], rules=rules_summary(device))


def system_prompt(version: Any, brain_text: str) -> str:
    parts = [getattr(version, "prompts", {}).get(ROLE) or default_prompt()]
    rules = getattr(version, "playbooks", {}).get(ROLE) or []
    if rules:
        parts.append("Playbook rules:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))
    if brain_text.strip():
        parts.append("Shared memory from earlier runs (verified evidence):\n" + brain_text.strip())
    return "\n\n".join(parts)


def lesson(task: Task, technique: str, ev: dict[str, Any], cat: str) -> str:
    p = task.payload
    if not cat:
        return f"{p['category']} {task.id}: {technique} was correct at {float(ev['speedup']):.2f}x vs torch eager"[:MAX_TEXT]
    first = (ev.get("reasons") or [ev.get("detail", "")])[0]
    return f"{p['category']} {task.id}: {technique} -> {cat}; {first}"[:MAX_TEXT]


def make_run_team(evaluator: Evaluator | None = None) -> Callable[[Any, Any, Task, Any], RunResult]:
    def run_team(version: Any, brain: Any, task: Task, llm: Any) -> RunResult:
        return _run_team(version, brain, task, llm, evaluator or remote.evaluate_candidate)
    return run_team


def _run_team(version: Any, brain: Any, task: Task, llm: Any, evaluator: Evaluator) -> RunResult:
    start = time.monotonic()
    cfg = getattr(version, "config", {}) or {}
    model = (cfg.get("models") or {}).get(ROLE, cfg.get("default_model", SUPER))
    thinking = bool((cfg.get("thinking") or {}).get(ROLE, False))
    attempts = max(1, min(int(cfg.get("attempts", 3)), MAX_ATTEMPTS))  # the Reviewer can set any value
    max_tokens = int(cfg.get("max_tokens", 8000))
    temperature = float(cfg.get("temperature", 0.3))
    p = task.payload
    brain_text, brain_ids = brain.read(ROLE, task.text, budget_chars=3000) if brain is not None else ("", [])
    result = RunResult(task_id=task.id, version=getattr(version, "version", 0), score=0.0, passed=False,
                       brain_ids_read=list(brain_ids))
    messages = [{"role": "system", "content": system_prompt(version, brain_text)},
                {"role": "user", "content": task_text(task)}]
    best: dict[str, Any] | None = None
    history, categories, proposals = [], [], []
    for k in range(1, attempts + 1):
        reply = llm.chat(model, list(messages), temperature=temperature, max_tokens=max_tokens, role=ROLE,
                         thinking=thinking)
        result.calls += 1
        result.prompt_tokens += reply.prompt_tokens
        result.completion_tokens += reply.completion_tokens
        result.cost_usd += reply.cost_usd
        result.iterations = k
        technique, code = extract_technique(reply.text), extract_code(reply.text)
        ev = evaluator(p["problem"], code) if code else {"error": "no_code", "detail": "no ```python block in the reply"}
        cat = "no_code" if ev.get("error") == "no_code" else category_of(ev)
        history.append({"attempt": k, "technique": technique, "category": cat or "faster",
                        "speedup": ev.get("speedup"), "correct": ev.get("correct"),
                        "reasons": (ev.get("reasons") or [ev.get("detail", "")])[:3]})
        if cat and cat not in categories:
            categories.append(cat)
        if cat != "no_gpu":
            proposals.append({"kind": "lesson" if not cat else "pitfall", "roles": [ROLE],
                              "text": lesson(task, technique, ev, cat),
                              "evidence": [{"run_id": f"{task.id}-{result.version}-a{k}", "task_id": task.id,
                                            "measurement": f"category={cat or 'faster'} speedup={ev.get('speedup')}"}]})
        if ev.get("correct") and (best is None or score_of(ev) > best["score"]):
            best = {"score": score_of(ev), "ev": ev, "code": code, "technique": technique}
        if cat == "no_gpu":
            break
        incumbent = (f"{best['technique']}, correct, {float(best['ev']['speedup']):.3f}x" if best
                     else "none yet (no correct kernel)")
        incumbent_code = (f"Incumbent source:\n```python\n{best['code']}\n```\n"
                          if best and best["code"] != code else "")
        messages = messages[:2] + [{"role": "assistant", "content": reply.text},
                                   {"role": "user", "content": FEEDBACK_TEMPLATE.format(
                                       k=k, outcome=describe(ev), incumbent=incumbent, incumbent_code=incumbent_code)}]
    result.failures = categories
    if best is not None:
        ev = best["ev"]
        result.score = round(best["score"], 4)
        result.passed = bool(ev.get("correct")) and float(ev.get("speedup", 0)) > 1.0
        result.claimed_done = True
        result.solution = {"case_study": "kernelbench", "problem": p["problem"], "technique": best["technique"],
                           "speedup": ev.get("speedup"), "solution": {"model_new": best["code"]}}
    result.notes = json.dumps({"proposals": proposals[-3:], "attempts": history,
                               "best": ({"technique": best["technique"], "speedup": best["ev"].get("speedup"),
                                         "suspicious": best["ev"].get("suspicious", False)} if best else None),
                               "correct": best is not None})
    result.wall_s = time.monotonic() - start
    if cfg.get("log_dir"):
        log = Path(cfg["log_dir"]) / f"{task.id}-v{result.version}.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"task": task.id, "attempts": history,
                                   "best_code": best["code"] if best else None}, indent=1))
        result.log_path = str(log)
    return result


run_team = make_run_team()


def kb_v0() -> Any:
    """The starting team: one kernel role, Nemotron Super, reasoning off, 3 attempts."""
    from team.evolve import TeamVersion
    if remote.device_kind() == "mps":  # the Reviewer reads config and changelog, so both name the device
        return TeamVersion(version=0, parent=None, prompts={ROLE: MPS_KERNEL_PROMPT}, playbooks={ROLE: []},
                           config={"roles_enabled": [ROLE], "models": {ROLE: SUPER}, "thinking": {ROLE: False},
                                   "default_model": SUPER, "max_tokens": 8000, "attempts": 3, "temperature": 0.3,
                                   "tool_catalog": [], "device": remote.MPS_LABEL},
                           changelog=["v0: one kernel role on the Apple GPU (PyTorch MPS, not CUDA): PyTorch ops or "
                                      "torch.mps.compile_shader Metal kernels, evaluator feedback between attempts, "
                                      "keep the best; speedup is vs PyTorch eager on MPS"])
    return TeamVersion(version=0, parent=None, prompts={ROLE: DEFAULT_KERNEL_PROMPT}, playbooks={ROLE: []},
                       config={"roles_enabled": [ROLE], "models": {ROLE: SUPER}, "thinking": {ROLE: False},
                               "default_model": SUPER, "max_tokens": 8000, "attempts": 3, "temperature": 0.3,
                               "tool_catalog": []},
                       changelog=["v0: one CUDA kernel role, evaluator feedback between attempts, keep the best"])
