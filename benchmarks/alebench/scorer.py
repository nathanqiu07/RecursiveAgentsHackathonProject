"""Official ALE-Bench scoring for one AtCoder Heuristic Contest problem, Docker-free.

Inputs come from the contest's own generator and outputs are scored by the contest's own visualizer, both
compiled to WASM and driven from Node by `vis_shim.mjs`; no score ever comes from an LLM. Candidate programs
run on the host in their own process group with a minimal environment (no API keys or tokens) and a hard
timeout; that is not a security sandbox, so only run programs from models you trust.
The problem data is CC BY-ND 4.0 (SakanaAI/ALE-Bench); it is downloaded by `fetch.py` and used unmodified.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBLEM = "ahc039"
TITLE = "AHC039 Purse Seine Fishing"
PROBLEMS = HERE / "problems"
PROBLEM_DIR = PROBLEMS / PROBLEM / PROBLEM
INPUTS_DIR = PROBLEMS / f"{PROBLEM}_inputs"
REFERENCE_JSON = PROBLEMS / f"{PROBLEM}_reference.json"
SHIM = HERE / "vis_shim.mjs"
GREEDY = HERE / "programs" / f"greedy_{PROBLEM}.py"
NAIVE = HERE / "programs" / f"naive_{PROBLEM}.py"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
KILL_SLACK_S = 0.5
_SECRET = re.compile(r"(_KEY|_TOKEN|SECRET|PASSWORD)$|^CRUSOE", re.IGNORECASE)


class MissingDataError(RuntimeError):
    pass


def node() -> str:
    exe = shutil.which("node")
    if not exe:
        raise RuntimeError("node is not on PATH; the official scorer runs the contest WASM visualizer under Node")
    return exe


def require_data() -> None:
    if not (PROBLEM_DIR / "data.json").is_file():
        raise MissingDataError(f"{PROBLEM} data missing; run: .venv/bin/python -m benchmarks.alebench.fetch")


@lru_cache(maxsize=1)
def metadata() -> dict[str, Any]:
    require_data()
    return json.loads((PROBLEM_DIR / "data.json").read_text())


def contest_time_limit() -> float:
    return float(metadata()["constraints"]["time_limit"])


def statement() -> str:
    require_data()
    return (PROBLEM_DIR / "statement_en.md").read_text()


def input_path(seed: int) -> Path:
    """The generator's input for `seed`, cached on disk (inputs are a pure function of the seed)."""
    path = INPUTS_DIR / f"{seed}.txt"
    if not path.is_file():
        require_data()
        INPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out = subprocess.run([node(), str(SHIM), str(PROBLEM_DIR), "gen", str(seed)], capture_output=True,
                             text=True, check=True, timeout=60)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(out.stdout)
        tmp.replace(path)
    return path


def score_files(pairs: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
    """Official scores for (input, output) file pairs in one Node process: [{"score", "err", "turns"}]."""
    if not pairs:
        return []
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump([[str(i), str(o)] for i, o in pairs], fh)
    try:
        out = subprocess.run([node(), str(SHIM), str(PROBLEM_DIR), "score-many", fh.name], capture_output=True,
                             text=True, check=True, timeout=60 + 2 * len(pairs))
    finally:
        os.unlink(fh.name)
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
    if len(rows) != len(pairs):
        raise RuntimeError(f"scorer returned {len(rows)} results for {len(pairs)} cases: {out.stderr[-300:]}")
    return rows


def child_env(workdir: Path) -> dict[str, str]:
    """A minimal environment: nothing from the parent survives except a safe PATH, so no key reaches the program."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir), "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LANG": "C.UTF-8"}
    return {k: v for k, v in env.items() if not _SECRET.search(k)}


def error_name(stderr: str, returncode: int) -> str:
    for line in reversed(stderr.strip().splitlines()):
        m = re.match(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))\b", line.strip())
        if m:
            return m.group(1).rsplit(".", 1)[-1]
    return f"exit_code_{returncode}"


def run_case(program: Path, seed: int, time_limit: float, workdir: Path) -> dict[str, Any]:
    """Run `program` on one seed's input (stdin) and return its status, output file and wall time."""
    inp = input_path(seed)
    out_path = workdir / f"out_{seed}.txt"
    python = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    t0 = time.monotonic()
    with open(inp) as fin, open(out_path, "w") as fout:
        proc = subprocess.Popen([python, "-I", str(program)], cwd=workdir, env=child_env(workdir), stdin=fin,
                                stdout=fout, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            _, stderr = proc.communicate(timeout=time_limit + KILL_SLACK_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            _, stderr = proc.communicate()
            return {"seed": seed, "status": "timeout", "wall_s": round(time.monotonic() - t0, 2),
                    "error": f"killed after {time_limit + KILL_SLACK_S:g} s", "input": inp, "output": out_path}
    wall = round(time.monotonic() - t0, 2)
    if proc.returncode != 0:
        return {"seed": seed, "status": "runtime_error", "wall_s": wall, "error": error_name(stderr, proc.returncode),
                "stderr_tail": stderr[-400:], "input": inp, "output": out_path}
    return {"seed": seed, "status": "ran", "wall_s": wall, "input": inp, "output": out_path}


def evaluate(code: str, seeds: list[int], time_limit: float, workers: int = 4) -> list[dict[str, Any]]:
    """Run a complete program on every seed and score each case officially.

    Each row: seed, status (ok | wrong_answer | timeout | runtime_error), score (0 unless ok), wall_s, error.
    """
    with tempfile.TemporaryDirectory(prefix="alebench_") as tmp:
        work = Path(tmp)
        program = work / "solver.py"
        program.write_text(code)
        dirs = []
        for s in seeds:
            d = work / f"case_{s}"
            d.mkdir()
            dirs.append(d)
        with ThreadPoolExecutor(max(1, workers)) as pool:
            runs = list(pool.map(lambda sd: run_case(program, sd[0], time_limit, sd[1]), zip(seeds, dirs)))
        ran = [r for r in runs if r["status"] == "ran"]
        for r, sc in zip(ran, score_files([(r["input"], r["output"]) for r in ran])):
            r["score"] = int(sc["score"]) if not sc["err"] else 0
            r["status"] = "wrong_answer" if sc["err"] else "ok"
            if sc["err"]:
                r["error"] = sc["err"][:200]
    rows = []
    for r in runs:
        rows.append({k: v for k, v in r.items() if k not in ("input", "output")} | {"score": r.get("score", 0)})
    return rows


def load_reference() -> dict[str, int]:
    return json.loads(REFERENCE_JSON.read_text()) if REFERENCE_JSON.is_file() else {}


def reference_scores(seeds: list[int], workers: int = 6) -> dict[int, int]:
    """The reference greedy's official score per seed, computed once and cached in the git-ignored data dir."""
    ref = load_reference()
    missing = [s for s in seeds if str(s) not in ref]
    if missing:
        rows = evaluate(GREEDY.read_text(), missing, contest_time_limit(), workers)
        bad = [r for r in rows if r["status"] != "ok"]
        if bad:
            raise RuntimeError(f"reference greedy failed on {len(bad)} seeds: {bad[0]}")
        ref = load_reference() | {str(r["seed"]): r["score"] for r in rows}
        REFERENCE_JSON.write_text(json.dumps(ref, indent=0, sort_keys=True))
    return {s: ref[str(s)] for s in seeds}


def ratio(case_score: float, reference: float) -> float:
    """Case score relative to the reference greedy; AHC039 is maximize (minimize problems would invert this)."""
    if metadata()["metadata"].get("score_type", "maximize") == "minimize":
        return reference / case_score if case_score > 0 else 0.0
    return case_score / max(reference, 1)


@lru_cache(maxsize=1)
def _standings() -> tuple[list[float], dict[int, int]]:
    scores = [float(r["score"]) for r in csv.DictReader(open(PROBLEM_DIR / "standings_scores_lite.csv"))]
    perf = {int(r["rank"]): int(r["performance"]) for r in csv.DictReader(open(PROBLEM_DIR / "performance.csv"))}
    return scores, perf


def rank_performance(total: float) -> dict[str, Any]:
    """Rank among the contest's human entrants (Lite standings, 150 private cases) and the AtCoder performance.

    A total ties with the human it equals; performance is read at that rank from the contest's own table.
    """
    scores, perf = _standings()
    rank = 1 + sum(1 for s in scores if s > total)
    ranks = sorted(perf)
    at = max([r for r in ranks if r <= rank] or [ranks[0]])
    return {"total": int(total), "rank": rank, "entrants": len(scores), "performance": perf[at],
            "top_human_total": int(max(scores))}
