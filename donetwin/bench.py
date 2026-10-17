"""Benchmark runner: score a solver over generated halls (or one fixed hall).

  python bench.py --solver greedy|hill_climb --tier small|medium|large|all --split heldout|train --n 10
                  [--seed-start 0] [--hall scenarios/data_hall_2n.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from donetwin import score as scoring  # noqa: E402

TIERS = ["small", "medium", "large"]


def _crusoe_key_ok():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("CRUSOE_API_KEY", "").strip()
    placeholder = None
    ex = ROOT / ".env.example"
    if ex.exists():
        for line in ex.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("CRUSOE_API_KEY="):
                placeholder = line.split("=", 1)[1].strip()
    return bool(key) and key != placeholder


def get_solver(name, split="train"):
    if name == "greedy":
        from donetwin.greedy import solve
        return solve
    if not _crusoe_key_ok():
        print("error: hill_climb needs CRUSOE_API_KEY (set it in .env or the environment; "
              "the .env.example placeholder does not count)", file=sys.stderr)
        sys.exit(2)
    from donetwin.hill_climb import solve as hc

    def solve(hall):
        # held-out halls must not be written into the example library (would leak into later prompts)
        best, history = hc(hall, library_dir=None) if split == "heldout" else hc(hall)
        solve.history = history
        return best
    return solve


def halls_for(args):
    if args.hall:
        p = Path(args.hall)
        p = p if p.is_absolute() else ROOT / p
        hall = json.loads(p.read_text(encoding="utf-8"))
        yield {"tier": None, "seed": None, "hall": hall}
        return
    from donetwin import halls
    if args.split == "heldout":
        seeds = list(halls.heldout_seeds())[args.seed_start:args.seed_start + args.n]
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.n))
    for tier in (TIERS if args.tier == "all" else [args.tier]):
        for s in seeds:
            yield {"tier": tier, "seed": s, "hall": halls.generate(s, tier)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", choices=["greedy", "hill_climb"], default="greedy")
    ap.add_argument("--tier", choices=TIERS + ["all"], default="all")
    ap.add_argument("--split", choices=["heldout", "train"], default="heldout")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--hall", default=None, help="score one fixed hall JSON instead of generated halls")
    args = ap.parse_args()

    solve = get_solver(args.solver, "fixed" if args.hall else args.split)
    hdr = f"{'hall':<28} {'racks':>5} {'it_kw':>8} {'bound':>8} {'dens':>6} {'resil':>6} {'hose':>6} {'score':>7} feas"
    print(hdr)
    print("-" * len(hdr))
    results = []
    for item in halls_for(args):
        hall = item["hall"]
        t0 = time.time()
        try:
            payload = solve(hall)
            res = scoring.score(hall, payload)
            err = None
        except Exception as e:  # a crashing solver scores 0 on that hall
            payload, res, err = {"racks": []}, {"score": 0.0, "feasible": False, "it_kw": 0,
                                                "upper_bound_kw": None, "components": {}}, repr(e)
        c = res.get("components") or {}
        name = str(hall.get("name", "?"))
        print(f"{name[:28]:<28} {len(payload.get('racks', [])):>5} {res.get('it_kw', 0):>8.0f} "
              f"{(res.get('upper_bound_kw') or 0):>8.0f} {c.get('density', 0):>6.3f} {c.get('resilience', 0):>6.3f} "
              f"{c.get('hose', 0):>6.3f} {res.get('score', 0):>7.2f} {'yes' if res.get('feasible') else 'NO'}"
              + (f"  ERROR {err}" if err else ""), flush=True)
        results.append({"hall": name, "tier": item["tier"], "seed": item["seed"], "secs": round(time.time() - t0, 2),
                        "error": err, "result": res, "payload": payload,
                        "history": getattr(solve, "history", None)})
    mean = sum(r["result"].get("score", 0) for r in results) / max(len(results), 1)
    print("-" * len(hdr))
    print(f"mean score: {mean:.3f}  over {len(results)} halls  "
          f"(feasible {sum(bool(r['result'].get('feasible')) for r in results)}/{len(results)})")
    out = ROOT / "runs" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.solver}.json"
    path.write_text(json.dumps({"args": vars(args), "mean_score": mean, "results": results}, indent=1, default=str),
                    encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
