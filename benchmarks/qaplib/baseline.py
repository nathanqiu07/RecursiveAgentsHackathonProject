"""Classical no-LLM baselines, so LLM-team scores can be read against a floor and a cheap heuristic.

`random`: one seeded random permutation. `two_swap`: first-improvement 2-swap local search
with random restarts until the time limit, the textbook heuristic an LLM should beat.
Run: .venv/bin/python -m benchmarks.qaplib.baseline --time-limit 10 --workers 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.qaplib.verify import INSTANCES, parse_dat, verify  # noqa: E402


def swap_deltas(a: np.ndarray, bp: np.ndarray, r: int) -> np.ndarray:
    """Cost change of swapping facility r with every s (Taillard's O(n) delta, vectorized over s).

    bp[i, j] = B[p[i], p[j]]; handles asymmetric matrices and non-zero diagonals.
    """
    diag_a, diag_b = np.diag(a), np.diag(bp)
    s1 = ((a[:, r][:, None] - a) * (bp - bp[:, r][:, None])).sum(axis=0)
    s1 -= (a[r, r] - a[r, :]) * (bp[r, :] - bp[r, r]) + (a[:, r] - diag_a) * (diag_b - bp[:, r])
    s2 = ((a[r, :][None, :] - a) * (bp - bp[r, :][None, :])).sum(axis=1)
    s2 -= (a[r, r] - a[:, r]) * (bp[:, r] - bp[r, r]) + (a[r, :] - diag_a) * (diag_b - bp[r, :])
    const = (a[r, r] - diag_a) * (diag_b - bp[r, r]) + (a[r, :] - a[:, r]) * (bp[:, r] - bp[r, :])
    delta = const + s1 + s2
    delta[r] = 0
    return delta


def two_swap(a: np.ndarray, b: np.ndarray, time_limit: float, seed: int = 0) -> tuple[list[int], int]:
    """Restarted first-improvement 2-swap descent; returns the best 0-based permutation and its cost."""
    rng = np.random.default_rng(seed)
    n = len(a)
    deadline = time.monotonic() + time_limit
    best_perm, best_cost = None, None
    while best_perm is None or time.monotonic() < deadline:
        perm = rng.permutation(n)
        bp = b[perm][:, perm]
        current = int((a * bp).sum())
        improved = True
        while improved and time.monotonic() < deadline:
            improved = False
            for r in rng.permutation(n):
                delta = swap_deltas(a, bp, int(r))
                s = int(delta.argmin())
                if delta[s] < 0:
                    perm[[r, s]] = perm[[s, r]]
                    bp = b[perm][:, perm]
                    current += int(delta[s])
                    improved = True
        if best_cost is None or current < best_cost:
            best_perm, best_cost = perm.tolist(), current
    return best_perm, best_cost


def evaluate(name: str, time_limit: float, seed: int = 0) -> dict:
    """Both baselines on one instance, scored by the exact verifier rather than our own arithmetic."""
    n, a_list, b_list = parse_dat(INSTANCES / f"{name}.dat")
    rand_perm = list(range(n))
    random.Random(seed).shuffle(rand_perm)
    ls_perm, _ = two_swap(np.array(a_list, dtype=np.int64), np.array(b_list, dtype=np.int64), time_limit, seed)
    out = {"name": name, "n": n}
    for label, perm in (("random", rand_perm), ("two_swap", ls_perm)):
        v = verify(name, perm)
        out[label] = {"score": round(v["score"], 6), "gap_pct": round(v["gap_pct"], 3), "cost": v["cost"]}
    return out


def main() -> None:
    from benchmarks.qaplib.splits import load_splits

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "reports" / "qaplib_baseline.json"))
    args = ap.parse_args()
    splits = load_splits()
    names = [n for s in ("train", "dev", "heldout") for n in splits[s]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = {r["name"]: r for r in pool.map(evaluate, names, [args.time_limit] * len(names))}
    summary = {}
    for split in ("train", "dev", "heldout"):
        part = [rows[n] for n in splits[split]]
        summary[split] = {label: {"mean_score": round(sum(r[label]["score"] for r in part) / len(part), 4),
                                  "mean_gap_pct": round(sum(r[label]["gap_pct"] for r in part) / len(part), 3),
                                  "at_bks": sum(r[label]["gap_pct"] <= 0 for r in part)}
                          for label in ("random", "two_swap")}
        summary[split]["n_instances"] = len(part)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"time_limit_s": args.time_limit, "summary": summary,
                                          "instances": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
