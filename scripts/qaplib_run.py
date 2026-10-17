"""Run the QAPLIB solver team on N tasks of a split against Crusoe and report verified scores.

Usage: .venv/bin/python scripts/qaplib_run.py --split dev --n 2 [--workers 2 --attempts 3 --time-limit 10]
Writes reports/qaplib_<split>.json. Exits 2 when CRUSOE_API_KEY is not configured.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.qaplib.load import load_tasks  # noqa: E402
from benchmarks.qaplib.team_runner import SUPER, run_team  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "heldout"])
    ap.add_argument("--n", type=int, default=2, help="number of tasks (first N of the split)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--model", default=SUPER)
    args = ap.parse_args()

    from team.crusoe import CrusoeLLM, MissingKeyError

    try:
        llm = CrusoeLLM()
    except MissingKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    version = SimpleNamespace(version=0, prompts={}, playbooks={}, config={
        "models": {"solver": args.model}, "attempts": args.attempts, "time_limit_s": args.time_limit,
        "temperature": 0.4, "log_dir": str(ROOT / "reports" / "qaplib_logs")})
    tasks = load_tasks(args.split)[: args.n]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda t: run_team(version, None, t, llm), tasks))
    rows = []
    for task, r in zip(tasks, results):
        gap = (task.payload["bks"] / r.score - 1) * 100 if r.score > 0 else None
        rows.append({"name": task.payload["name"], "n": task.payload["n"], "score": round(r.score, 6),
                     "gap_pct": None if gap is None else round(gap, 3), "passed": r.passed,
                     "attempts": r.iterations, "failures": r.failures, "result": asdict(r)})
        print(f"{task.payload['name']:>8} n={task.payload['n']:<3} score={r.score:.4f} "
              f"gap={'n/a' if gap is None else f'{gap:.2f}%'} attempts={r.iterations} cost=${r.cost_usd:.4f}")
    mean = sum(r.score for r in results) / len(results) if results else 0.0
    print(f"mean score {mean:.4f} over {len(results)} {args.split} tasks")
    out = ROOT / "reports" / f"qaplib_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "model": args.model, "mean_score": mean, "tasks": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
