"""Modern QAP instances from an AI data center: GPU rack placement in generated DoneTwin halls.

QAPLIB (1991) is the external anchor: exact scoring and published optima, but its small instances are solved
and its answers have been online for decades. These instances are fresh, seeded, and on-theme, and they are
written in QAPLIB's own .dat format, so verify.py, baseline.py and team_runner.py score them unchanged.

  facilities = the n racks of one training job; locations = the n rack slots of a hall's greedy layout
               (donetwin.halls.generate(seed, tier) -> donetwin.greedy.solve, as team/datahall.py does)
  A[i][j]    = traffic between racks i and j in GB/s (model below)
  B[k][l]    = Manhattan cable-tray distance between slot centres k and l, in decimetres (QAPLIB files are integers)
  cost(p)    = sum_ij A[i][j] * B[p[i]][p[j]]  (GB/s x dm of cable the job's traffic crosses)

There is no published optimum, so each instance's "bks" is a reference cost: the best of a time-boxed robust
tabu search and the 2-swap baseline. Score = reference / cost; above 1.0 beats our own classical reference.
Train/dev halls come from donetwin's train seeds (0..9999), held-out halls from its held-out seeds
(10000..10049), so the splits are disjoint by seed.

Data lives in benchmarks/qaplib/dc_instances/: <name>.dat, index.json, splits.json.
  .venv/bin/python -m benchmarks.qaplib.dc_instances [--time-limit 20] [--workers 4]   # regenerate + table
  .venv/bin/python -m benchmarks.qaplib.dc_instances --table                           # table from stored data
Top-level imports are stdlib only, so the stdlib verifier service can read the index; numpy and donetwin load lazily.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DC_DIR = HERE / "dc_instances"
SEED = 20261017
SPLIT_SIZES = (4, 3, 6)
TIER_CYCLE = ("medium", "large")
# 24..40 racks: on a first cut at 12..40, 10 s of 2-swap restarts already matched the 20 s tabu reference on every
# hall with n <= 22 (9 of 13), which leaves no headroom to measure. Small halls (13..19 racks) never qualify.
N_RANGE = (24, 40)
REFERENCE_TIME_S = 20.0
TWO_SWAP_TIME_S = 10.0  # the same budget as reports/qaplib_baseline.json's 2-swap
PREFIX = "dc-"

# Traffic model (GB/s between two racks, both directions summed). Assumptions for a realistic shape, not measurements:
#   Tensor parallel (TP): each TP group spans a rack pair and all-reduces every layer, so its two racks exchange
#     TP_GBPS = 400 (heavy; a scale-up domain stretched across two racks).
#   Pipeline parallel (PP): a model replica is S TP groups in sequence (stages); activations and their gradients flow
#     stage k <-> k+1 rack-by-rack at PP_GBPS = 100 (medium).
#   Data parallel (DP): the R replicas of each stage all-reduce gradients in a ring (replica r -> r+1 mod R),
#     rack-by-rack at DP_GBPS = 25 (light: once per step, overlapped with compute).
#   Racks left over after S x R groups are storage/ingest racks streaming training data to every stage-0 rack at
#     INGEST_GBPS = 10. S in {2, 3, 4} is the stage count that leaves the fewest racks over with R >= 2 (seeded tie-break).
#   Roles go to rack indices by a seeded random permutation, so index order carries no structure. Every link is scaled
#   by a seeded factor in [1 - JITTER, 1 + JITTER] (uneven stage sizes, MoE imbalance), which also breaks the ties that
#   made a perfectly regular job easy for 2-swap.
TP_GBPS, PP_GBPS, DP_GBPS, INGEST_GBPS = 400, 100, 25, 10
JITTER = 0.25


def is_dc(name: str) -> bool:
    return name.startswith(PREFIX)


def instance_name(seed: int, tier: str) -> str:
    return f"{PREFIX}{tier}-{seed}"


# ---------------------------------------------------------------- generation (deterministic by seed, tier)

def _donetwin() -> tuple[Any, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    donetwin_dir = str(ROOT / "donetwin")
    if donetwin_dir not in sys.path:
        sys.path.append(donetwin_dir)  # appended, as team/datahall.py does: donetwin/ also holds a tests package
    from donetwin import greedy, halls
    return greedy, halls


def hall_slots(seed: int, tier: str) -> tuple[str, list[tuple[float, float]]]:
    """Rack slot centres (m) of the hall's greedy layout, in the layout's own order."""
    greedy, halls = _donetwin()
    hall = halls.generate(seed, tier)
    racks = greedy.solve(hall)["racks"]
    return hall["name"], [(float(r["x"]), float(r["y"])) for r in racks]


def distance_matrix(slots: list[tuple[float, float]]) -> list[list[int]]:
    """Manhattan cable-tray distance between slot centres, in decimetres."""
    return [[round(10 * (abs(x1 - x2) + abs(y1 - y2))) for (x2, y2) in slots] for (x1, y1) in slots]


def job_shape(n: int, rng: random.Random) -> tuple[int, int, int]:
    """(stages S, replicas R, storage racks) for n racks in 2-rack TP groups."""
    groups = n // 2
    options = [(n - 2 * s * (groups // s), rng.random(), s) for s in (2, 3, 4) if groups // s >= 2]
    left, _, s = min(options)
    return s, groups // s, left


def traffic_matrix(n: int, seed: int) -> tuple[list[list[int]], dict[str, Any]]:
    rng = random.Random(f"dc-traffic:{seed}:{n}")
    stages, replicas, storage = job_shape(n, rng)
    rack = list(range(n))
    rng.shuffle(rack)  # rack[k] = the facility index that plays role slot k
    a = [[0] * n for _ in range(n)]

    def add(i: int, j: int, v: int) -> None:
        v = round(v * rng.uniform(1 - JITTER, 1 + JITTER))
        a[i][j] += v
        a[j][i] += v

    def group(r: int, s: int) -> tuple[int, int]:  # the two racks of replica r, stage s
        k = 2 * (r * stages + s)
        return rack[k], rack[k + 1]

    for r in range(replicas):
        for s in range(stages):
            g = group(r, s)
            add(g[0], g[1], TP_GBPS)
            if s + 1 < stages:
                nxt = group(r, s + 1)
                add(g[0], nxt[0], PP_GBPS)
                add(g[1], nxt[1], PP_GBPS)
    for s in range(stages):
        ring = {tuple(sorted((r, (r + 1) % replicas))) for r in range(replicas)}
        for r1, r2 in sorted(ring):
            g1, g2 = group(r1, s), group(r2, s)
            add(g1[0], g2[0], DP_GBPS)
            add(g1[1], g2[1], DP_GBPS)
    for k in range(2 * stages * replicas, n):
        for r in range(replicas):
            for x in group(r, 0):
                add(rack[k], x, INGEST_GBPS)
    return a, {"stages": stages, "replicas": replicas, "tp_racks": 2, "storage_racks": storage}


def build(seed: int, tier: str) -> dict[str, Any]:
    hall_id, slots = hall_slots(seed, tier)
    n = len(slots)
    a, job = traffic_matrix(n, seed)
    return {"name": instance_name(seed, tier), "seed": seed, "tier": tier, "hall": hall_id, "n": n, "job": job,
            "A": a, "B": distance_matrix(slots)}


def dat_text(n: int, a: list[list[int]], b: list[list[int]]) -> str:
    rows = lambda m: "\n".join(" ".join(str(v) for v in row) for row in m)  # noqa: E731
    return f"{n}\n\n{rows(a)}\n\n{rows(b)}\n"


def task_text(e: dict[str, Any]) -> str:
    job = e["job"]
    return (f"GPU rack placement in hall {e['hall']}: place {e['n']} racks to minimize traffic x cable distance. "
            f"A = rack-to-rack GB/s of a training job ({job['stages']} pipeline stages x {job['replicas']} "
            f"data-parallel replicas of tensor-parallel rack pairs, {job['storage_racks']} storage racks); "
            f"B = Manhattan cable-tray distance between slots in dm; BKS is a reference cost, not a proven optimum")


def choose_seeds(seed: int = SEED, sizes: tuple[int, int, int] = SPLIT_SIZES) -> dict[str, list[tuple[int, str]]]:
    """Seeded (seed, tier) picks: train/dev from donetwin's train seeds, held-out from its held-out seeds.

    Tiers cycle small/medium/large within each split; a hall whose greedy layout has n outside N_RANGE is skipped.
    """
    _, halls = _donetwin()
    rng = random.Random(seed)
    train_dev = rng.sample(range(10000), 200)  # one pool, consumed in order, so train and dev never share a seed
    heldout = list(halls.heldout_seeds())
    rng.shuffle(heldout)
    out: dict[str, list[tuple[int, str]]] = {}
    cursor = {"td": 0, "h": 0}
    for split, k in zip(("train", "dev", "heldout"), sizes):
        pool, key = (heldout, "h") if split == "heldout" else (train_dev, "td")
        picks: list[tuple[int, str]] = []
        while len(picks) < k:
            s = pool[cursor[key]]
            cursor[key] += 1
            tier = TIER_CYCLE[len(picks) % len(TIER_CYCLE)]
            if N_RANGE[0] <= len(hall_slots(s, tier)[1]) <= N_RANGE[1]:
                picks.append((s, tier))
        out[split] = picks
    return out


# ---------------------------------------------------------------- solvers for the stored reference and baseline

def delta_matrix(a: Any, bp: Any) -> Any:
    """Cost change of swapping the locations of facilities r and s, for every (r, s) at once.

    Symmetric A and B with zero diagonals only (true here). bp[i, j] = B[p[i], p[j]].
    """
    m = a @ bp
    d = m.diagonal()
    return 2 * (m + m.T - d[:, None] - d[None, :] + 2 * a * bp)


def robust_tabu(a: Any, b: Any, time_limit: float, seed: int = 0) -> tuple[list[int], int]:
    """Taillard-style robust tabu search over 2-swaps, random tenure ~n, aspiration on the best, restarts from a
    perturbed best after a stall. Returns the best 0-based permutation and its exact cost."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(a)
    iu_r, iu_s = np.triu_indices(n, 1)
    big = np.iinfo(np.int64).max
    perm = rng.permutation(n)
    bp = b[perm][:, perm]
    cur = int((a * bp).sum())
    best, best_perm = cur, perm.copy()
    tabu = np.zeros((n, n), dtype=np.int64)
    it = last = 0
    stall = max(2000, 40 * n)
    deadline = time.monotonic() + time_limit
    while time.monotonic() < deadline:
        it += 1
        delta = delta_matrix(a, bp)[iu_r, iu_s]
        forbidden = (tabu[iu_r, perm[iu_s]] >= it) & (tabu[iu_s, perm[iu_r]] >= it) & (cur + delta >= best)
        cand = np.where(forbidden, big, delta)
        k = int(cand.argmin())
        if cand[k] == big:
            k = int(rng.integers(len(delta)))
        r, s = int(iu_r[k]), int(iu_s[k])
        lo = max(1, int(0.9 * n))
        tabu[r, perm[r]] = it + int(rng.integers(lo, int(1.1 * n) + 2))
        tabu[s, perm[s]] = it + int(rng.integers(lo, int(1.1 * n) + 2))
        perm[[r, s]] = perm[[s, r]]
        bp = b[perm][:, perm]
        cur += int(delta[k])
        if cur < best:
            best, best_perm, last = cur, perm.copy(), it
        elif it - last > stall:  # diversify: restart from the best with n/4 random swaps and a clean tabu list
            perm = best_perm.copy()
            for _ in range(max(2, n // 4)):
                i, j = rng.choice(n, 2, replace=False)
                perm[[i, j]] = perm[[j, i]]
            bp = b[perm][:, perm]
            cur = int((a * bp).sum())
            tabu[:] = 0
            last = it
    assert int((a * b[best_perm][:, best_perm]).sum()) == best
    return best_perm.tolist(), best


def _solve(job: tuple[str, str, float]) -> tuple[str, str, list[int], int, float]:
    import numpy as np

    from benchmarks.qaplib.baseline import two_swap
    from benchmarks.qaplib.verify import parse_dat

    name, method, limit = job
    _, a_list, b_list = parse_dat(DC_DIR / f"{name}.dat")
    a, b = np.array(a_list, dtype=np.int64), np.array(b_list, dtype=np.int64)
    t0 = time.monotonic()
    perm, value = (robust_tabu if method == "tabu" else two_swap)(a, b, limit, 0)
    return name, method, [int(x) for x in perm], int(value), round(time.monotonic() - t0, 1)


# ---------------------------------------------------------------- stored data (stdlib only)

def load_index() -> dict[str, dict[str, Any]]:
    return {e["name"]: e for e in json.loads((DC_DIR / "index.json").read_text())["instances"]}


def load_splits() -> dict[str, Any]:
    return json.loads((DC_DIR / "splits.json").read_text())


def load_tasks(split: str) -> list[Any]:
    """The same Task shape as benchmarks/qaplib/load.py, so team_runner.run_team runs these unchanged."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from team.types import Task

    if split not in ("train", "dev", "heldout"):
        raise ValueError(f"unknown split {split!r}; expected train, dev or heldout")
    index = load_index()
    tasks = []
    for name in load_splits()[split]:
        e = index[name]
        tasks.append(Task(id=f"qaplib-{name}", split=split, text=e["comment"],
                          payload={"name": name, "path": str(DC_DIR / f"{name}.dat"), "n": e["n"], "bks": e["bks"]}))
    return tasks


def table(index: dict[str, dict[str, Any]], splits: dict[str, Any]) -> str:
    lines = ["| split | instance | n | 2-swap cost | reference cost | 2-swap score |", "|---|---|---|---|---|---|"]
    for split in ("train", "dev", "heldout"):
        for name in splits[split]:
            e = index[name]
            lines.append(f"| {split} | {name} | {e['n']} | {e['two_swap']['cost']} | {e['bks']} | "
                         f"{e['bks'] / e['two_swap']['cost']:.4f} |")
    return "\n".join(lines)


def generate(time_limit: float = REFERENCE_TIME_S, workers: int = 4) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from benchmarks.qaplib.splits import heldout_hash

    picks = choose_seeds()
    DC_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DC_DIR.glob(f"{PREFIX}*.dat"):  # this module owns the directory; a regenerated set replaces it whole
        stale.unlink()
    built = {}
    for split in ("train", "dev", "heldout"):
        for seed, tier in picks[split]:
            inst = build(seed, tier)
            text = dat_text(inst["n"], inst["A"], inst["B"])
            (DC_DIR / f"{inst['name']}.dat").write_text(text)
            inst["sha256"] = hashlib.sha256(text.encode()).hexdigest()
            built[inst["name"]] = inst
    jobs = [(name, m, time_limit if m == "tabu" else TWO_SWAP_TIME_S) for name in built for m in ("tabu", "two_swap")]
    print(f"solving {len(built)} instances: robust tabu {time_limit:g} s + 2-swap {TWO_SWAP_TIME_S:g} s each, "
          f"{workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        solved = {(name, m): (perm, value, wall) for name, m, perm, value, wall in pool.map(_solve, jobs)}
    entries = []
    for name, inst in built.items():
        tabu_perm, tabu_cost, tabu_wall = solved[(name, "tabu")]
        two_perm, two_cost, _ = solved[(name, "two_swap")]
        ref_perm, ref_cost, ref_by = ((tabu_perm, tabu_cost, "robust tabu") if tabu_cost <= two_cost
                                      else (two_perm, two_cost, "2-swap"))
        e = {"name": name, "n": inst["n"], "family": "dc", "bks": ref_cost, "bks_proven_optimal": False,
             "hall": inst["hall"], "seed": inst["seed"], "tier": inst["tier"], "job": inst["job"],
             "sha256": inst["sha256"],
             "reference": {"cost": ref_cost, "found_by": ref_by, "perm": ref_perm,
                           "tabu_cost": tabu_cost, "tabu_time_s": tabu_wall},
             "two_swap": {"cost": two_cost, "perm": two_perm, "time_limit_s": TWO_SWAP_TIME_S, "seed": 0}}
        e["comment"] = task_text(e)
        entries.append(e)
    names = {split: [instance_name(s, t) for s, t in picks[split]] for split in ("train", "dev", "heldout")}
    by_name = {e["name"]: e for e in entries}
    (DC_DIR / "index.json").write_text(json.dumps({
        "source": "benchmarks/qaplib/dc_instances.py: GPU rack placement in donetwin halls (greedy layout slots)",
        "traffic_gbps": {"tp": TP_GBPS, "pp": PP_GBPS, "dp": DP_GBPS, "ingest": INGEST_GBPS},
        "distance_unit": "dm (Manhattan between rack slot centres)", "reference_time_s": time_limit,
        "instances": entries}, indent=1) + "\n")
    (DC_DIR / "splits.json").write_text(json.dumps({
        "seed": SEED, **names, "seeds": {k: [[s, t] for s, t in v] for k, v in picks.items()},
        "heldout_sha256": heldout_hash(by_name, names["heldout"])}, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", action="store_true", help="print the table from stored data; do not regenerate")
    ap.add_argument("--time-limit", type=float, default=REFERENCE_TIME_S, help="robust tabu seconds per instance")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if not args.table:
        generate(args.time_limit, args.workers)
    print(table(load_index(), load_splits()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
