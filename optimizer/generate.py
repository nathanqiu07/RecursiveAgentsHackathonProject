"""Seeded variant generator: perturb the approximate hall, solve both modes, summarise.

Each variant is an unseen hall with a proven optimum (or proven infeasible target),
which is what lets an LLM agent be scored as a fraction of the optimum rather than
by vibes. Infeasible targets are kept on purpose: the right agent move is to escalate.

Usage: python generate.py [--n 50] [--seed 7] [--no-binding] [--fine-check K] [--out DIR]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from solve import binding, capacity_bound_racks, solve

HERE = Path(__file__).resolve().parent


def make_variant(base: dict[str, Any], rng: random.Random, idx: int) -> dict[str, Any]:
    """Perturb columns, keep-outs, derate, CDUs, hose length and target."""
    s = copy.deepcopy(base)
    s["name"] = f"variant_{idx:03d}"
    for col in s["columns"]:
        col["x"] = round(min(21.5, max(2.5, col["x"] + rng.uniform(-3, 3))), 1)
    zones = {z["id"]: z for z in s["zones"]}
    vault = zones["vault_slab"]
    vault["x0"] = round(rng.uniform(14.0, 20.0), 1)
    vault["y0"] = round(rng.uniform(8.0, 9.0), 1)
    vault["y1"] = round(rng.uniform(13.4, 14.6), 1)
    nec = zones["nec_PNL_1"]
    nec["x0"] = round(rng.uniform(17.0, 21.5), 1)
    nec["y0"] = round(3.4 - rng.uniform(0.8, 1.2), 1)
    nec["y1"] = round(3.4 + rng.uniform(0.8, 1.2), 1)
    for b in s["busways"]:
        b["derate"] = round(rng.uniform(0.7, 0.9), 2)
    n_cdu = rng.randint(3, 5)
    hose = round(rng.uniform(10.0, 16.0) * 2) / 2
    spacing = 20.0 / n_cdu
    s["cdus"] = [{
        "id": f"CDU_{k + 1}",
        "port_x": round(min(23.0, max(1.0, 2.0 + spacing * (k + 0.5) + rng.uniform(-1.5, 1.5))), 1),
        "port_y": 14.8,
        "capacity_kw": rng.choice([600, 700, 800, 900, 1000]),
        "hose_m": hose,
    } for k in range(n_cdu)]
    s["goal"]["target_kw"] = round(rng.uniform(1.0, 3.0), 1) * 1000
    s["goal"]["text"] = f"Deploy {s['goal']['target_kw'] / 1000:.1f} MW of GB200 NVL72 racks with N+1 cooling"
    return s


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile; enough for a 50-sample summary."""
    v = sorted(values)
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-binding", action="store_true")
    ap.add_argument("--fine-check", type=int, default=0, help="re-solve the first K variants on a 0.3 m x grid")
    ap.add_argument("--out", default=str(HERE / "out"))
    args = ap.parse_args()

    base = json.loads((HERE / "scene_approx_data_hall.json").read_text())
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.n):
        scene = make_variant(base, rng, i)
        kw = scene["rack_catalog"][scene["goal"]["rack"]]["kw"]
        a = solve(scene, "max_kw")
        b = solve(scene, "min_hose")
        need = math.ceil(scene["goal"]["target_kw"] / kw)
        row = {
            "name": scene["name"], "target_kw": scene["goal"]["target_kw"], "need_racks": need,
            "cap_bound_racks": capacity_bound_racks(scene), "hose_m": scene["cdus"][0]["hose_m"],
            "n_cdu": len(scene["cdus"]),
            "max_status": a["status"], "max_kw": a["total_kw"], "max_wall": a["wall_s"],
            "max_verify": a.get("verify_pass", not a["layout"]["racks"]),
            "hose_status": b["status"], "hose_total_m": b["objective"], "hose_wall": b["wall_s"],
            "hose_verify": b.get("verify_pass") if b["layout"]["racks"] else None,
            "verify_hose_m": b.get("verify_hose_m"),
        }
        if b["status"] == "INFEASIBLE":
            row["infeasible_kind"] = "capacity" if need > row["cap_bound_racks"] else "geometry"
        if not args.no_binding:
            gains = binding(scene, a["total_kw"])
            row["binding"] = [f for f, g in gains.items() if g > 0]
        if i < args.fine_check:
            fine = solve(scene, "max_kw", x_step=0.3)
            row["fine_max_kw"] = fine["total_kw"]
            row["fine_status"] = fine["status"]
        rows.append(row)
        print(f"{scene['name']} target={row['target_kw']:.0f} max={row['max_kw']} ({a['status']}, {a['wall_s']}s) "
              f"hose={b['status']} {row['hose_total_m']} ({b['wall_s']}s) bind={row.get('binding')}", flush=True)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "variants.jsonl", "a" if i else "w") as f:
            f.write(json.dumps({"scene": scene, "result": row, "max_kw_layout": a["layout"],
                                "min_hose_layout": b["layout"]}) + "\n")

    summarise(rows)


def summarise(rows: list[dict[str, Any]]) -> None:
    """Print the numbers the spike verdict rests on."""
    n = len(rows)
    feas = [r for r in rows if r["hose_status"] == "OPTIMAL"]
    infeas = [r for r in rows if r["hose_status"] == "INFEASIBLE"]
    unproven = [r for r in rows if r["hose_status"] not in ("OPTIMAL", "INFEASIBLE")]
    max_unproven = [r for r in rows if r["max_status"] != "OPTIMAL"]
    opt = [r["max_kw"] for r in rows]
    racks: dict[int, int] = {}
    for r in rows:
        racks[r["max_kw"] // 132] = racks.get(r["max_kw"] // 132, 0) + 1
    checked = [r for r in rows if r["max_verify"] is not None] + [r for r in rows if r["hose_verify"] is not None]
    agree = sum(1 for r in rows if r["max_verify"]) + sum(1 for r in rows if r["hose_verify"])
    times_a = [r["max_wall"] for r in rows]
    times_b = [r["hose_wall"] for r in rows]
    print("\n=== summary ===")
    print(f"variants: {n}")
    print(f"target feasible (min_hose OPTIMAL): {len(feas)}/{n} = {len(feas) / n:.0%}")
    print(f"target proven INFEASIBLE: {len(infeas)} "
          f"(capacity-bound: {sum(r['infeasible_kind'] == 'capacity' for r in infeas)}, "
          f"geometry-only: {sum(r['infeasible_kind'] == 'geometry' for r in infeas)}); unproven: {len(unproven)}")
    print(f"max_kw not proven optimal: {len(max_unproven)}")
    print(f"optimum kW: min {min(opt)} p25 {pct(opt, .25)} median {statistics.median(opt)} "
          f"p75 {pct(opt, .75)} max {max(opt)}")
    print(f"optimum racks histogram: {dict(sorted(racks.items()))}")
    print(f"max_kw wall s: mean {statistics.mean(times_a):.2f} max {max(times_a):.2f}")
    print(f"min_hose wall s: mean {statistics.mean(times_b):.2f} max {max(times_b):.2f}")
    print(f"verifier agreed on solver layouts: {agree}/{len(checked)}")
    hose_match = [r for r in feas if abs(r["verify_hose_m"] - r["hose_total_m"]) < 0.02]
    print(f"verifier hose total == solver objective: {len(hose_match)}/{len(feas)}")
    if any("binding" in r for r in rows):
        counts: dict[str, int] = {}
        for r in rows:
            for f in r.get("binding", []):
                counts[f] = counts.get(f, 0) + 1
        print(f"binding families (single relaxation raises max kW): {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
        print(f"variants with no single binding family: {sum(1 for r in rows if not r.get('binding'))}")
    fine = [r for r in rows if "fine_max_kw" in r]
    if fine:
        same = sum(1 for r in fine if r["fine_max_kw"] == r["max_kw"])
        print(f"0.3 m grid max kW == 0.6 m grid: {same}/{len(fine)}")


if __name__ == "__main__":
    main()
