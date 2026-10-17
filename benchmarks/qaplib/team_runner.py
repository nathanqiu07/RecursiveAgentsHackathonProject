"""`run_team` for QAPLIB: the solver role writes a heuristic program, we run it and verify the answer exactly.

Each attempt's verified gap (or error) is fed back so the model can improve its own program;
the best verified attempt is the score. Failure lines and one technique note flow to the shared brain.
The generated program runs in a subprocess with a scrubbed environment (no API keys) and a hard
timeout; it is not a security sandbox, so only run trusted models.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.qaplib import dc_instances  # noqa: E402
from benchmarks.qaplib.fetch import family  # noqa: E402
from benchmarks.qaplib.verify import parse_dat, verify  # noqa: E402
from team.types import RunResult, Task  # noqa: E402

SUPER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
BASELINE_JSON = ROOT / "reports" / "qaplib_baseline.json"
PASS_GAP_PCT = 1.0
MAX_TEXT = 280
KILL_GRACE_S = 5.0
TIME_LIMIT_ENV = "QAP_TIME_LIMIT_S"
# The Evolver range-checks max_tokens and temperature but not these solver knobs, so a Reviewer
# config_set cannot turn one run into hundreds of calls or minutes of program time.
ATTEMPTS_RANGE = (1, 6)
# Capped at the 2-swap baseline's budget (reports/qaplib_baseline.json and dc_instances.TWO_SWAP_TIME_S, both
# 10 s), so a Reviewer config_set can trade time away but never buy the team more wall clock than the baseline.
TIME_LIMIT_RANGE_S = (0.1, dc_instances.TWO_SWAP_TIME_S)

DEFAULT_SOLVER_PROMPT = (
    "You write heuristic optimization programs for the Quadratic Assignment Problem. "
    "Your program is executed and its answer is scored by an exact verifier; only the verified cost counts. "
    "Correctness first (always print a valid permutation before the time limit), then solution quality."
)

TASK_TEMPLATE = """Instance {name}: {text}. n = {n}.
Best known objective (BKS) = {bks}. Lower is better; matching the BKS scores 1.0, beating it scores more than 1.0.
Objective: cost(p) = sum over i, j of A[i][j] * B[p[i]][p[j]], where p[i] is the 0-based location of facility i.

Write ONE self-contained Python 3.10 program (standard library and numpy only) that:
- reads the instance JSON path from sys.argv[1]; the JSON has keys "n", "A", "B" (lists of lists of ints),
- reads its wall-clock budget in seconds from sys.argv[2] (also in the environment variable {env}); the budget is {limit:g} s,
- prints a line `PERMUTATION: <n space-separated 1-based ints>` (value i = p[i] + 1); only the LAST such line counts.

Run contract: the runner times the program from launch (interpreter start-up included) and reports the time used.
A program still running at budget + {grace:g} s is killed and scores 0, whatever it printed. Budget left unused earns
nothing. {reference}

Reply with one line `TECHNIQUE: <short name of your method>`, then the program in a single ```python block."""

FEEDBACK_TEMPLATE = """Attempt {k} result: {outcome}
Runtime: {runtime}.
Reference: {reference}
Best so far: {best}.
Write an improved complete program (same input/output contract, same time budget). Fix any error first.
Reply with one line `TECHNIQUE: <short name of your method>`, then the program in a single ```python block."""


def two_swap_reference(name: str, bks: int) -> dict[str, float] | None:
    """The stored 2-swap baseline (no LLM, restarts for its whole budget) on this instance: gap % and budget s.

    The same numbers the final vs-2-swap table uses, so per-task feedback and the held-out comparison agree.
    """
    try:
        if dc_instances.is_dc(name):
            cost = dc_instances.load_index()[name]["two_swap"]["cost"]
            return {"gap_pct": (cost - bks) / bks * 100.0, "budget_s": dc_instances.TWO_SWAP_TIME_S}
        data = json.loads(BASELINE_JSON.read_text())
        return {"gap_pct": float(data["instances"][name]["two_swap"]["gap_pct"]),
                "budget_s": float(data["time_limit_s"])}
    except (OSError, KeyError, ValueError, ZeroDivisionError):
        return None  # instances outside the frozen library (e.g. test fixtures) have no stored baseline


def reference_text(ref: dict[str, float] | None, limit: float) -> str:
    if ref is None:
        return "no stored 2-swap baseline for this instance."
    same = "the same budget" if abs(ref["budget_s"] - limit) < 1e-9 else f"{ref['budget_s']:g} s vs your {limit:g} s"
    return (f"the 2-swap baseline (no LLM, local search with random restarts for its whole budget; {same}) "
            f"reaches gap {ref['gap_pct']:.2f}% on this instance; your program is compared against it.")


def extract_code(text: str) -> str | None:
    """Longest fenced block wins, since models sometimes add a short usage snippet."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len)
    return text if "PERMUTATION" in text and "import" in text else None


def extract_technique(text: str) -> str:
    m = re.search(r"^\s*TECHNIQUE:\s*(.+?)\s*$", text, flags=re.MULTILINE)
    return (m.group(1) if m else "unspecified technique")[:100]


def error_name(stderr: str, returncode: int) -> str:
    for line in reversed(stderr.strip().splitlines()):
        m = re.match(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))\b", line.strip())
        if m:
            return m.group(1).rsplit(".", 1)[-1]
    return f"exit_code_{returncode}"


def run_program(code: str, instance_json: Path, time_limit: float, workdir: Path, k: int) -> dict[str, Any]:
    """Run in its own process group so a timeout also kills any worker processes it spawned."""
    program = workdir / f"solver_{k}.py"
    program.write_text(code)
    python = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    # Built from scratch rather than copied from os.environ, so CRUSOE_API_KEY and every other secret stays out.
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir), "PYTHONHASHSEED": "0", TIME_LIMIT_ENV: f"{time_limit:g}",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    t0 = time.monotonic()
    proc = subprocess.Popen([python, "-I", str(program), str(instance_json), f"{time_limit:g}"], cwd=workdir, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=time_limit + KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        return {"status": "timeout", "stdout": stdout, "stderr": stderr, "runtime_s": time.monotonic() - t0}
    runtime = time.monotonic() - t0
    if proc.returncode != 0:
        return {"status": "runtime_error", "error": error_name(stderr, proc.returncode), "stdout": stdout,
                "stderr": stderr, "runtime_s": runtime}
    return {"status": "ran", "stdout": stdout, "stderr": stderr, "runtime_s": runtime}


def evaluate_attempt(reply_text: str, task: Task, instance_json: Path, time_limit: float, workdir: Path, k: int) -> dict[str, Any]:
    p = task.payload
    technique = extract_technique(reply_text)
    code = extract_code(reply_text)
    if code is None:
        return {"status": "no_code", "technique": technique, "error": "no ```python block in the reply"}
    run = run_program(code, instance_json, time_limit, workdir, k)
    result = {"technique": technique, "status": run["status"], "runtime_s": round(run["runtime_s"], 3),
              "time_limit_s": time_limit, "stderr_tail": run.get("stderr", "")[-600:],
              "stdout_tail": run.get("stdout", "")[-300:], "program": code}
    if run["status"] == "timeout":
        result["error"] = (f"still running at the {time_limit + KILL_GRACE_S:g} s kill (budget {time_limit:g} s); "
                           "nothing it printed is scored")
        return result
    if run["status"] == "runtime_error":
        result["error"] = run["error"]
        return result
    lines = [ln for ln in run["stdout"].splitlines() if "PERMUTATION:" in ln]
    if not lines:
        return {**result, "status": "invalid", "error": "no `PERMUTATION:` line on stdout"}
    v = verify(p["name"], lines[-1], bks=p["bks"], dat_path=p["path"])
    if v["status"] != "ok":
        return {**result, "status": "invalid", "error": v["error"]}
    return {**result, "status": "ok", "cost": v["cost"], "score": v["score"], "gap_pct": v["gap_pct"],
            "perm": [int(x) for x in v["perm"]]}


def describe(a: dict[str, Any], bks: int) -> str:
    if a["status"] == "ok":
        return f"valid, cost {a['cost']} vs BKS {bks}, score {a['score']:.4f}, gap {a['gap_pct']:.2f}%"
    if a["status"] == "no_code":
        return f"no_code (score 0): {a.get('error', '')}"
    tail = f"\nstderr tail:\n{a['stderr_tail']}" if a.get("stderr_tail") else ""
    if a["status"] == "invalid" and a.get("stdout_tail"):
        tail += f"\nstdout tail:\n{a['stdout_tail']}"
    return f"{a['status']} (score 0): {a.get('error', '')}{tail}"


def runtime_text(a: dict[str, Any]) -> str:
    if "runtime_s" not in a:
        return "not run"
    limit, used = a["time_limit_s"], a["runtime_s"]
    over = f", {used - limit:.2f} s over budget" if used > limit else ""
    return f"{used:.2f} s used of the {limit:g} s budget ({used / limit:.0%}){over}"


def failure_line(a: dict[str, Any], name: str, ref: dict[str, float] | None = None) -> str:
    """Budget and baseline come first: the brain keeps about 80 characters of each line."""
    used = f" in {a['runtime_s']:.1f}/{a['time_limit_s']:g}s budget" if "runtime_s" in a else ""
    if a["status"] == "ok":
        vs = f" (2-swap {ref['gap_pct']:.2f}%)" if ref else ""
        return f"solver: {name} gap {a['gap_pct']:.2f}%{vs}{used} with {a['technique']}"
    if a["status"] == "runtime_error":
        return f"solver: {name} runtime_error {a['error']}{used}"
    if a["status"] == "timeout":
        return f"solver: {name} timeout past {a['time_limit_s']:g}s budget (killed at +{KILL_GRACE_S:g}s)"
    return f"solver: {name} {a['status']}{used} ({a.get('error', '')[:80]})"


def system_prompt(version: Any, brain_text: str) -> str:
    """Playbook rules are numbered so reviewers and evolved prompts can cite them."""
    parts = [getattr(version, "prompts", {}).get("solver") or DEFAULT_SOLVER_PROMPT]
    rules = getattr(version, "playbooks", {}).get("solver") or []
    if rules:
        parts.append("Playbook rules:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)))
    if brain_text.strip():
        parts.append("Shared memory from earlier runs (verified evidence):\n" + brain_text.strip())
    return "\n\n".join(parts)


def bounded(cfg: dict[str, Any], key: str, default: float, lo: float, hi: float) -> float:
    """A missing or non-numeric value falls back to the default, so one bad config_set fails its Gate, not the run."""
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        value = default
    return min(max(float(value), lo), hi)


def run_team(version: Any, brain: Any, task: Task, llm: Any) -> RunResult:
    start = time.monotonic()
    cfg = getattr(version, "config", {}) or {}
    model = (cfg.get("models") or {}).get("solver") or cfg.get("default_model") or SUPER
    thinking = (cfg.get("thinking") or {}).get("solver")  # None leaves the model's default
    attempts = int(bounded(cfg, "attempts", 3, *ATTEMPTS_RANGE))
    time_limit = bounded(cfg, "time_limit_s", 10.0, *TIME_LIMIT_RANGE_S)
    temperature = float(cfg.get("temperature", 0.4))
    max_tokens = int(cfg.get("max_tokens", 6000))
    p = task.payload
    ref = two_swap_reference(p["name"], p["bks"])
    ref_text = reference_text(ref, time_limit)
    brain_text, brain_ids = brain.read("solver", task.text, budget_chars=3000) if brain is not None else ("", [])
    result = RunResult(task_id=task.id, version=getattr(version, "version", 0), score=0.0, passed=False,
                       brain_ids_read=list(brain_ids))
    messages = [{"role": "system", "content": system_prompt(version, brain_text)},
                {"role": "user", "content": TASK_TEMPLATE.format(
                    name=p["name"], text=task.text, n=p["n"], bks=p["bks"], limit=time_limit, env=TIME_LIMIT_ENV,
                    grace=KILL_GRACE_S, reference="Reference: " + ref_text)}]
    best: dict[str, Any] | None = None
    history = []
    with tempfile.TemporaryDirectory(prefix="qaplib_") as tmp:
        workdir = Path(tmp)
        n, a, b = parse_dat(p["path"])
        instance_json = workdir / "instance.json"
        instance_json.write_text(json.dumps({"name": p["name"], "n": n, "A": a, "B": b}))
        for k in range(1, attempts + 1):
            reply = llm.chat(model, list(messages), temperature=temperature, max_tokens=max_tokens, role="solver",
                             thinking=thinking)
            result.calls += 1
            result.prompt_tokens += reply.prompt_tokens
            result.completion_tokens += reply.completion_tokens
            result.cost_usd += reply.cost_usd
            result.iterations = k
            attempt = evaluate_attempt(reply.text, task, instance_json, time_limit, workdir, k)
            history.append({k2: v for k2, v in attempt.items() if k2 not in ("stderr_tail", "stdout_tail", "perm")}
                           | {"two_swap_gap_pct": ref["gap_pct"] if ref else None})
            if attempt["status"] == "ok" and (best is None or attempt["cost"] < best["cost"]):
                best = attempt
            if attempt["status"] != "ok" or attempt["cost"] > p["bks"]:
                result.failures.append(failure_line(attempt, p["name"], ref))
            if best is not None and best["cost"] <= p["bks"]:
                break
            best_text = f"gap {best['gap_pct']:.2f}% ({best['technique']})" if best else "no valid answer yet"
            messages = messages + [{"role": "assistant", "content": reply.text},
                         {"role": "user", "content": FEEDBACK_TEMPLATE.format(
                             k=k, outcome=describe(attempt, p["bks"]), runtime=runtime_text(attempt),
                             reference=ref_text, best=best_text)}]
    if best is not None:
        result.score = best["score"]
        result.passed = best["gap_pct"] <= PASS_GAP_PCT
        result.claimed_done = True
        # The attempts (program, runtime, status) ride along so saved solutions/ can be re-run and audited.
        result.solution = {"case_study": "qaplib", "verified_cost": best["cost"],
                           "solution": {"instance": p["name"], "permutation": best["perm"]}, "attempts": history}
        vs = f", 2-swap {ref['gap_pct']:.2f}%" if ref else ""
        text = (f"QAP {family(p['name'])} n={p['n']}: {best['technique']} reached gap {best['gap_pct']:.2f}% "
                f"on {p['name']} in {best['runtime_s']:.1f} of {time_limit:g}s budget{vs}")[:MAX_TEXT]
        result.notes = json.dumps({"proposals": [{
            "kind": "lesson", "roles": ["solver"], "text": text,
            "evidence": [{"run_id": f"{task.id}-{result.version}", "task_id": task.id,
                          "measurement": f"gap={best['gap_pct']:.2f}%"}]}]})
    result.wall_s = time.monotonic() - start
    if cfg.get("log_dir"):
        log = Path(cfg["log_dir"]) / f"{task.id}-v{result.version}.json"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({"task": task.id, "attempts": history}, indent=1))
        result.log_path = str(log)
    return result
