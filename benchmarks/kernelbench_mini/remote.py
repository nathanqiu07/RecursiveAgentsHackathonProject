"""Where a candidate gets scored: a remote CUDA host over ssh, a local GPU, or nowhere (`{"error": "no_gpu"}`).

  KB_REMOTE=user@host       run evaluate.py on that host (see setup_gpu_vm.sh); takes precedence
  KB_REMOTE_DIR=~/kb_mini   where the evaluator lives on the host (default ~/kb_mini)
  KB_REMOTE_PYTHON=...      its interpreter (default ~/kb_venv/bin/python)
  KB_SSH_OPTS="-i key"      extra ssh/scp options (optional)

With no KB_REMOTE and torch.cuda.is_available() here, evaluate.py runs on this machine: in a child process
with a scrubbed environment by default (a kernel that faults poisons its CUDA context, and the child cannot
read API keys), or in-process with KB_INPROCESS=1. Otherwise `{"error": "no_gpu"}`; nothing raises.
The ssh/scp children get a minimal environment, so no API key can travel to the host (ssh SendEnv only
forwards variables that exist). One evaluation at a time per process: parallel timing on one GPU is noise.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROBLEMS = HERE / "problems"
SHIP_FILES = ("evaluate.py", "static_checks.py", "kb_static_checker.py")
_EVAL_LOCK = threading.Lock()
_SYNCED: set[str] = set()


def ssh_env() -> dict[str, str]:
    """Only what ssh needs to find the user's agent and config; never API keys."""
    keep = ("PATH", "HOME", "USER", "LOGNAME", "SSH_AUTH_SOCK", "TMPDIR")
    return {k: os.environ[k] for k in keep if k in os.environ}


def local_cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch
    return bool(torch.cuda.is_available())


def gpu_mode() -> str | None:
    """"remote", "local" or None."""
    if os.environ.get("KB_REMOTE"):
        return "remote"
    return "local" if local_cuda() else None


def problem_path(problem: str) -> Path:
    path = PROBLEMS / (problem if problem.endswith(".py") else f"{problem}.py")
    if not path.exists():
        raise FileNotFoundError(f"unknown problem {problem!r}; see {PROBLEMS}")
    return path


def _ship_digest() -> str:
    h = hashlib.sha256()
    for name in SHIP_FILES:
        h.update((HERE / name).read_bytes())
    for p in sorted(PROBLEMS.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def scp_dir(remote_dir: str) -> str:
    """scp destinations are relative to the remote home, which sidesteps `~` handling differences."""
    return remote_dir[2:] if remote_dir.startswith("~/") else remote_dir


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=ssh_env(), capture_output=True, text=True, timeout=timeout)


def _last_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return None


def sync_evaluator(remote: str, remote_dir: str, ssh_opts: list[str]) -> None:
    """Copy evaluate.py, the checkers and problems/ once per process (again if they changed)."""
    key = f"{remote}:{remote_dir}:{_ship_digest()}"
    if key in _SYNCED:
        return
    mk = _run(["ssh", *ssh_opts, remote, f"mkdir -p {remote_dir}/problems {remote_dir}/cands"], 60)
    if mk.returncode != 0:
        raise RuntimeError(f"ssh {remote} failed: {mk.stderr.strip()[-400:]}")
    cp = _run(["scp", "-q", *ssh_opts, *[str(HERE / n) for n in SHIP_FILES], f"{remote}:{scp_dir(remote_dir)}/"], 120)
    cp2 = _run(["scp", "-q", *ssh_opts, *[str(p) for p in sorted(PROBLEMS.glob("*.py"))], str(PROBLEMS / "LICENSE"), f"{remote}:{scp_dir(remote_dir)}/problems/"], 120)
    for c in (cp, cp2):
        if c.returncode != 0:
            raise RuntimeError(f"scp to {remote} failed: {c.stderr.strip()[-400:]}")
    _SYNCED.add(key)


def evaluate_remote(remote: str, problem: str, candidate_src: str, trials: int = 50, timeout: float = 900) -> dict[str, Any]:
    remote_dir = os.environ.get("KB_REMOTE_DIR", "~/kb_mini")
    python = os.environ.get("KB_REMOTE_PYTHON", "~/kb_venv/bin/python")
    ssh_opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new", *shlex.split(os.environ.get("KB_SSH_OPTS", ""))]
    sync_evaluator(remote, remote_dir, ssh_opts)
    digest = hashlib.sha256(candidate_src.encode()).hexdigest()[:16]
    with tempfile.TemporaryDirectory(prefix="kbmini_") as tmp:
        local = Path(tmp) / f"{digest}.py"
        local.write_text(candidate_src)
        cp = _run(["scp", "-q", *ssh_opts, str(local), f"{remote}:{scp_dir(remote_dir)}/cands/{digest}.py"], 120)
    if cp.returncode != 0:
        return {"error": "remote_failed", "detail": f"scp: {cp.stderr.strip()[-400:]}"}
    name = problem_path(problem).name
    cmd = (f"cd {remote_dir} && env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin HOME=$HOME "
           f"{python} evaluate.py --problem problems/{shlex.quote(name)} --candidate cands/{digest}.py --trials {int(trials)}")
    run = _run(["ssh", *ssh_opts, remote, cmd], timeout)
    out = _last_json(run.stdout)
    if out is None:
        return {"error": "remote_failed", "detail": (run.stderr or run.stdout).strip()[-800:]}
    out["host"] = remote
    return out


def evaluate_local(problem: str, candidate_src: str, trials: int = 50, timeout: float = 900) -> dict[str, Any]:
    if os.environ.get("KB_INPROCESS") == "1":
        sys.path.insert(0, str(HERE))
        import evaluate as ev
        return ev.evaluate(problem_path(problem), candidate_src, trials=trials)
    with tempfile.TemporaryDirectory(prefix="kbmini_") as tmp:
        cand = Path(tmp) / "cand.py"
        cand.write_text(candidate_src)
        env = {k: os.environ[k] for k in ("PATH", "HOME", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
               if k in os.environ}
        run = subprocess.run([sys.executable, str(HERE / "evaluate.py"), "--problem", str(problem_path(problem)),
                              "--candidate", str(cand), "--trials", str(int(trials))],
                             env=env, capture_output=True, text=True, timeout=timeout)
    out = _last_json(run.stdout)
    return out if out is not None else {"error": "eval_failed", "detail": (run.stderr or run.stdout).strip()[-800:]}


def evaluate_candidate(problem: str, candidate_src: str, trials: int = 50) -> dict[str, Any]:
    """Score one candidate; returns evaluate.py's JSON, or {"error": ...}. Serialized per process."""
    mode = gpu_mode()
    if mode is None:
        return {"error": "no_gpu", "detail": "set KB_REMOTE=user@host or run on a CUDA machine"}
    with _EVAL_LOCK:
        try:
            if mode == "remote":
                return evaluate_remote(os.environ["KB_REMOTE"], problem, candidate_src, trials)
            return evaluate_local(problem, candidate_src, trials)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            return {"error": "remote_failed" if mode == "remote" else "eval_failed",
                    "detail": f"{type(exc).__name__}: {exc}"[-800:]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", help="problem file name, e.g. 19_ReLU")
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--trials", type=int, default=50)
    a = ap.parse_args()
    print(json.dumps(evaluate_candidate(a.problem, a.candidate.read_text(), a.trials), indent=1))
