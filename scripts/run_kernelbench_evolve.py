"""Evolve the CUDA kernel team on mini KernelBench: Nemotron writes ModelNew, a GPU evaluator scores it, the team edits itself.

Usage (repo venv):
  .venv/bin/python scripts/run_kernelbench_evolve.py --offline            # scripted LLM + FAKE evaluator, runs on a Mac
  KB_REMOTE=ubuntu@<gpu-ip> .venv/bin/python scripts/run_kernelbench_evolve.py --live --one   # one train problem, v0
  KB_REMOTE=ubuntu@<gpu-ip> .venv/bin/python scripts/run_kernelbench_evolve.py --live [--generations 2] [--out DIR]
Score per problem = speedup over torch eager if correct (capped at 10), else 0; pass (fast_1) = correct and faster.
Offline numbers come from a fake evaluator and prove plumbing only; never report them.
Exit codes: 0 done, 2 no Crusoe API key configured, 3 no GPU (set KB_REMOTE=user@host or run on a CUDA machine).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.kernelbench_mini import remote  # noqa: E402
from benchmarks.kernelbench_mini.fake_eval import FakeEvaluator  # noqa: E402
from benchmarks.kernelbench_mini.splits import SPLIT_SIZES, load_tasks  # noqa: E402
from benchmarks.kernelbench_mini.team_runner import ROLE, kb_v0, make_run_team  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.crusoe import SUPER, CrusoeLLM, MissingKeyError  # noqa: E402
from team.evolve import Evolver, Gate, TeamVersion  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.types import LLM, RunResult, Task  # noqa: E402

CANDIDATES_PER_GEN = 3  # Reviewer max_mutations
_PROBLEM = re.compile(r"PROBLEM (\S+):")
# Offline only: the speedup the fake evaluator reports for a plain / a vectorized kernel per problem.
OFFLINE_BASE = {"kb_matmul_square": 0.6, "kb_matmul_tb": 0.55, "kb_sum_dim": 1.1, "kb_softmax": 1.3, "kb_rmsnorm": 1.2,
                "kb_relu": 1.05, "kb_gelu": 1.4, "kb_conv2d": 0.8, "kb_cumsum": 1.6, "kb_argmax": 1.2, "kb_mse": 1.5,
                "kb_scale_copy": 1.0}

KERNEL_SKELETON = '''import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
// fake: {tag}
__global__ void k(const float* __restrict__ x, float* __restrict__ y, long n) {{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = {expr};
}}
torch::Tensor run(torch::Tensor x) {{
    auto y = torch::empty_like(x);
    long n = x.numel();
    k<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}}
"""
ext = load_inline(name="kbmini_{pid}", cpp_sources="torch::Tensor run(torch::Tensor x);", cuda_sources=cuda_src,
                  functions=["run"], verbose=False)


class ModelNew(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x, *rest):
{body}
'''


def offline_kernel(pid: str, kind: str, speedup: float) -> str:
    """Scripted candidates. `kind`: ok, identity (the ReLU hack), fallback (try/except to torch)."""
    if kind == "fallback":
        body = "        try:\n            return ext.run(x)\n        except RuntimeError:\n            return torch.relu(x)"
        return KERNEL_SKELETON.format(tag="behaviour=exact speedup=1.0", expr="x[i]", pid=pid, body=body)
    behaviour, expr = ("identity", "x[i]") if kind == "identity" else ("exact", "x[i] > 0.f ? x[i] : 0.f")
    return KERNEL_SKELETON.format(tag=f"behaviour={behaviour} speedup={speedup}", expr=expr, pid=pid,
                                  body="        return ext.run(x.contiguous())")


def offline_llm() -> ScriptedLLM:
    """v0's first attempt is a hack (try/except fallback, or the ReLU identity); after feedback it writes a plain
    kernel. A playbook rule naming float4 makes the first attempt a vectorized kernel at 1.5x the plain speed."""
    reviews = iter([
        {"mutations": [
            {"type": "playbook_add", "role": ROLE, "step": "Never fall back to torch; use float4 vectorized loads "
             "and a grid-stride loop; inputs are signed", "rationale": "first attempts are rejected fallbacks",
             "claim": {"dev_gain": 0.3, "cost_change_pct": -30.0, "why": "first attempt lands, and faster"}},
            {"type": "config_set", "key": "attempts", "value": 2, "rationale": "cheaper",
             "claim": {"dev_gain": 0.0, "cost_change_pct": -33.0, "why": "one fewer kernel call"}},
            {"type": "config_set", "key": "thinking", "value": {"nobody": True}, "rationale": "invalid on purpose",
             "claim": {"dev_gain": 0.1, "cost_change_pct": 50.0, "why": "unknown role, must be dropped"}}]},
        {"mutations": [
            {"type": "config_set", "key": "attempts", "value": 1, "rationale": "the first attempt is now clean",
             "claim": {"dev_gain": 0.0, "cost_change_pct": -50.0, "why": "half the calls"}}]},
    ])

    def respond(role: str, messages: list[dict[str, str]]) -> str:
        if role == "reviewer":
            return json.dumps(next(reviews, {"mutations": []}))
        found = _PROBLEM.search(messages[1]["content"])
        pid = found.group(1) if found else "kb_relu"
        base = OFFLINE_BASE.get(pid, 1.0)
        if "float4" in messages[0]["content"]:
            code = offline_kernel(pid, "ok", round(base * 1.5, 3))
            return f"TECHNIQUE: float4 grid-stride\n```python\n{code}```"
        if len(messages) == 2:
            kind = "identity" if pid == "kb_relu" else "fallback"
            return f"TECHNIQUE: quick {kind}\n```python\n{offline_kernel(pid, kind, base)}```"
        return f"TECHNIQUE: one thread per element\n```python\n{offline_kernel(pid, 'ok', base)}```"

    return ScriptedLLM(respond)


def estimate_calls(version: TeamVersion, n_train: int, n_dev: int, n_heldout: int, generations: int) -> dict[str, int]:
    attempts = int(version.config.get("attempts", 3))
    runs = generations * (n_train + n_dev + CANDIDATES_PER_GEN * n_dev) + 3 * n_heldout
    return {"runs": runs, "kernel_max": runs * attempts, "evals_max": runs * attempts, "reviewer": generations}


def _best(r: RunResult) -> dict[str, Any]:
    notes = json.loads(r.notes) if r.notes.strip().startswith("{") else {}
    return notes.get("best") or {}


def kb_row(label: str, version: TeamVersion, rs: list[RunResult]) -> dict[str, Any]:
    n = max(1, len(rs))
    correct = [r for r in rs if _best(r)]
    speedups = [float(_best(r).get("speedup") or 0.0) for r in correct]
    tokens, cost = sum(r.prompt_tokens + r.completion_tokens for r in rs), sum(r.cost_usd for r in rs)
    return {"config": label, "version": version.version, "n": len(rs),
            "fast_1": round(sum(r.passed for r in rs) / n, 4), "correct_rate": round(len(correct) / n, 4),
            "mean_speedup_correct": round(mean(speedups), 4) if speedups else 0.0,
            "mean_score": round(sum(r.score for r in rs) / n, 4),
            "calls_per_task": round(sum(r.calls for r in rs) / n, 3), "tokens_per_task": round(tokens / n, 1),
            "cost_per_task_usd": round(cost / n, 8), "tokens": tokens, "cost_usd": round(cost, 6),
            "failures": sorted({f for r in rs for f in r.failures})}


def kb_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Config | fast_1 | Correct | Mean speedup (correct) | Mean score | Calls/task | Tokens/task | Cost/task (USD) |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['config']} | {r['fast_1']:.0%} | {r['correct_rate']:.0%} | {r['mean_speedup_correct']:.2f}x | "
              f"{r['mean_score']:.3f} | {r['calls_per_task']:.2f} | {r['tokens_per_task']:.0f} | {r['cost_per_task_usd']:.5f} |"
              for r in rows]
    return "\n".join(lines) + "\n"


def final_report(evolver: Evolver, heldout: list[Task], out: Path) -> list[dict[str, Any]]:
    """Held-out ablation like Evolver.final_report, kept per-run so correctness and fast_1 can be shown."""
    evolver._final = True  # the Evolver's held-out guard; this is the one place held-out tasks run
    with tempfile.TemporaryDirectory() as tmp:
        arms = [("v0 + empty brain (cold)", evolver.v0, SharedBrain(tmp)),
                ("v0 + final brain", evolver.v0, evolver.brain),
                (f"v{evolver.current.version} + final brain", evolver.current, evolver.brain)]
        rows = [kb_row(label, v, evolver._batch(v, b, heldout)) for label, v, b in arms]
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "before_after.json").write_text(json.dumps({"rows": rows, "history": evolver.history}, indent=1))
    (reports / "before_after.md").write_text(kb_table(rows))
    return rows


def run_one(llm: LLM, task: Task, v0: TeamVersion, run_team: Any) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        r = run_team(v0, SharedBrain(tmp), task, llm)
    notes = json.loads(r.notes)
    print(f"task {r.task_id} ({task.text})")
    print(f"score {r.score:.3f} passed={r.passed} iterations={r.iterations} calls={r.calls} "
          f"tokens={r.prompt_tokens}+{r.completion_tokens} cost=${r.cost_usd:.5f} wall={r.wall_s:.1f}s")
    print(f"failures: {r.failures}")
    for a in notes["attempts"]:
        print(f"  attempt {a['attempt']}: {a['technique']} -> {a['category']} speedup={a['speedup']} "
              f"{(a['reasons'] or [''])[0][:160]}")


def evolve(llm: LLM, run_team: Any, reviewer_model: str, out: Path, generations: int, seed: int, workers: int,
           v0: TeamVersion, splits: tuple[list[Task], list[Task], list[Task]], min_gain: float) -> None:
    train, dev, heldout = splits
    evolver = Evolver(run_team, llm, reviewer_model, SharedBrain(out / "brain"), out / "versions",
                      max_workers=workers, seed=seed, initial=v0, gate=Gate(min_gain=min_gain),
                      reports_dir=out / "reports")
    t0 = time.perf_counter()
    final = evolver.run(generations, train, dev, heldout)
    for h in evolver.history:
        print(f"gen {h['generation']}: v{h['parent']} -> v{h['version']} accepted={h['accepted']} "
              f"train={h['train_mean']:.3f} dev={h['dev_mean']:.3f} brain_active={h['brain']['active']}", flush=True)
    print("\nchangelog:\n  " + "\n  ".join(final.changelog))
    rows = final_report(evolver, heldout, out)
    print(f"\nHeld-out ({len(heldout)} unseen problems: {', '.join(t.id for t in heldout)}); "
          "fast_1 = correct and faster than torch eager:\n")
    print(kb_table(rows))
    print("failure categories per arm: " + "; ".join(f"{r['config']}: {r['failures']}" for r in rows))
    print(f"\nwall {time.perf_counter() - t0:.1f}s; artifacts: {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="scripted LLM + fake evaluator, no network, no GPU")
    mode.add_argument("--live", action="store_true", help="Crusoe Managed Inference + a CUDA evaluator")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--one", action="store_true", help="run one train problem with v0 and print its bill")
    ap.add_argument("--out", type=Path, default=None, help="default runs/kernelbench-<timestamp>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="Evolver threads; GPU evaluations are still serialized")
    ap.add_argument("--attempts", type=int, default=3, help="v0 kernel attempts per problem")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--min-gain", type=float, default=0.05, help="Gate: dev mean score gain to keep a mutation")
    ap.add_argument("--kernel-model", default=SUPER)
    ap.add_argument("--reviewer-model", default=SUPER)
    args = ap.parse_args(argv)

    v0 = kb_v0()
    v0.config.update(attempts=args.attempts, max_tokens=args.max_tokens, default_model=args.kernel_model,
                     models={ROLE: args.kernel_model})
    if args.offline:
        llm: Any = offline_llm()
        run_team = make_run_team(FakeEvaluator())
        print("OFFLINE: scripted LLM and a FAKE evaluator; numbers below prove plumbing, not kernels.", flush=True)
    else:
        gpu = remote.gpu_mode()
        if gpu is None:
            print("No GPU evaluator: set KB_REMOTE=user@host (a CUDA VM prepared with "
                  "benchmarks/kernelbench_mini/setup_gpu_vm.sh) or run on a machine where torch.cuda.is_available().",
                  file=sys.stderr)
            return 3
        try:
            llm = CrusoeLLM()
        except MissingKeyError as exc:
            print(f"No Crusoe API key configured: {exc}", file=sys.stderr)
            return 2
        run_team = make_run_team(None)
        print(f"GPU evaluator: {gpu}" + (f" ({remote.os.environ['KB_REMOTE']})" if gpu == "remote" else ""), flush=True)
    train, dev, heldout = load_tasks()
    if args.one:
        run_one(llm, train[0], v0, run_team)
        if args.live:
            print(json.dumps(llm.ledger.summary()["by_role"], indent=1))
        return 0

    est = estimate_calls(v0, len(train), len(dev), len(heldout), args.generations)
    print(f"Estimated: ~{est['runs']} team runs -> up to {est['kernel_max']} kernel calls ({args.kernel_model}) and "
          f"{est['evals_max']} GPU evaluations, + {est['reviewer']} reviewer calls ({args.reviewer_model})", flush=True)
    out = args.out or Path("runs") / f"kernelbench-{time.strftime('%Y%m%d-%H%M%S')}"
    evolve(llm, run_team, args.reviewer_model, out, args.generations, args.seed, args.workers, v0,
           (train, dev, heldout), args.min_gain)
    if args.live:
        usage = llm.ledger.summary()
        (out / "reports" / "usage.json").write_text(json.dumps(usage, indent=1))
        print(f"usage: {json.dumps(usage['total'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
