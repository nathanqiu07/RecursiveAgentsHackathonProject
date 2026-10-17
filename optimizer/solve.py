"""Exact CP-SAT model for the data-hall layout benchmark.

The hall is discretised into candidate rack slots (busway x grid times a few row
y-offsets inside tap reach), so "optimal" means optimal over that slot set. Every
returned layout is re-checked by verify.py, which shares no code with this file.

Modes:
  max_kw   maximise deployed kW subject to every constraint
  min_hose deploy at least target kW, minimise total hose run (or prove infeasible)

Usage: python solve.py [scene.json] [--target-kw N] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

from verify import failed, verify

EPS = 1e-6
RELAXABLE = ("busway", "cdu", "n_plus_1", "hose", "columns", "zones", "aisle")
STATUS = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE", cp_model.INFEASIBLE: "INFEASIBLE",
          cp_model.MODEL_INVALID: "MODEL_INVALID", cp_model.UNKNOWN: "UNKNOWN"}


@dataclass(frozen=True)
class Slot:
    busway: str
    x: float
    y: float
    offset_idx: int


def _rect(x: float, y: float, w: float, d: float) -> tuple[float, float, float, float]:
    return (x - w / 2, y - d / 2, x + w / 2, y + d / 2)


def _hits(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > EPS and min(a[3], b[3]) - max(a[1], b[1]) > EPS


def build_slots(scene: dict[str, Any], x_step: float, y_step: float, relax: frozenset[str]) -> list[Slot]:
    """Candidate centres per busway, pruned against bounds, columns and keep-out zones."""
    rack = scene["rack_catalog"][scene["goal"]["rack"]]
    W, D = scene["floor"]["width"], scene["floor"]["depth"]
    slots = []
    for b in scene["busways"]:
        n_off = int(math.floor(b["tap_reach"] / y_step + EPS))
        offsets = [k * y_step for k in range(-n_off, n_off + 1)]
        n_x = int(math.floor((b["x1"] - b["x0"]) / x_step + EPS))
        for k in range(n_x + 1):
            x = round(b["x0"] + k * x_step, 6)
            for oi, off in enumerate(offsets):
                y = round(b["y"] + off, 6)
                r = _rect(x, y, rack["w"], rack["d"])
                if r[0] < -EPS or r[1] < -EPS or r[2] > W + EPS or r[3] > D + EPS:
                    continue
                if "columns" not in relax and any(
                        _hits(r, _rect(c["x"], c["y"], c["w"], c["d"])) for c in scene["columns"]):
                    continue
                if "zones" not in relax and any(
                        _hits(r, (z["x0"], z["y0"], z["x1"], z["y1"])) for z in scene["zones"]):
                    continue
                slots.append(Slot(b["id"], x, y, oi))
    return slots


def conflict_pairs(slots: list[Slot], w: float, d: float, aisle: float, relax: frozenset[str]) -> list[tuple[int, int]]:
    """Slot pairs that cannot coexist: overlap, or x-overlapping rows closer than the aisle."""
    order = sorted(range(len(slots)), key=lambda i: slots[i].x)
    pairs = []
    for pos, i in enumerate(order):
        si = slots[i]
        for j in order[pos + 1:]:
            sj = slots[j]
            if sj.x - si.x >= w - EPS:
                break
            if si.busway == sj.busway and si.offset_idx != sj.offset_idx:
                continue
            gap = abs(si.y - sj.y) - d
            if gap < -EPS or ("aisle" not in relax and gap < aisle - EPS):
                pairs.append((i, j))
    return pairs


def solve(scene: dict[str, Any], mode: str, target_kw: float | None = None, x_step: float = 0.6,
          y_step: float = 0.2, relax: frozenset[str] = frozenset(), time_limit_s: float = 10.0,
          check: bool = True) -> dict[str, Any]:
    """Build and solve the CP-SAT model; verify the result unless constraints were relaxed."""
    t0 = time.perf_counter()
    rack = scene["rack_catalog"][scene["goal"]["rack"]]
    kw = int(rack["kw"])
    target = scene["goal"]["target_kw"] if target_kw is None else target_kw
    slots = build_slots(scene, x_step, y_step, relax)
    cdus = scene["cdus"]

    m = cp_model.CpModel()
    place = [m.NewBoolVar(f"p{i}") for i in range(len(slots))]
    assign: dict[tuple[int, int], cp_model.IntVar] = {}
    hose_cm: dict[tuple[int, int], int] = {}
    for i, s in enumerate(slots):
        options = []
        for c_idx, c in enumerate(cdus):
            run = abs(s.x - c["port_x"]) + abs(s.y - c["port_y"])
            if "hose" not in relax and run > c["hose_m"] + EPS:
                continue
            v = m.NewBoolVar(f"a{i}_{c_idx}")
            assign[i, c_idx] = v
            hose_cm[i, c_idx] = int(round(run * 100))
            options.append(v)
        m.Add(sum(options) == place[i])

    rows: dict[str, dict[int, cp_model.IntVar]] = {}
    for i, s in enumerate(slots):
        row = rows.setdefault(s.busway, {})
        if s.offset_idx not in row:
            row[s.offset_idx] = m.NewBoolVar(f"row_{s.busway}_{s.offset_idx}")
        m.AddImplication(place[i], row[s.offset_idx])
    for row in rows.values():
        m.AddAtMostOne(row.values())

    for i, j in conflict_pairs(slots, rack["w"], rack["d"], scene["rules"]["aisle_min_m"], relax):
        m.AddBoolOr([place[i].Not(), place[j].Not()])

    if "busway" not in relax:
        for b in scene["busways"]:
            members = [place[i] for i, s in enumerate(slots) if s.busway == b["id"]]
            m.Add(kw * sum(members) <= int(math.floor(b["rating_kw"] * b["derate"] + EPS)))
    if "cdu" not in relax:
        for c_idx, c in enumerate(cdus):
            members = [v for (i, ci), v in assign.items() if ci == c_idx]
            m.Add(kw * sum(members) <= int(c["capacity_kw"]))
    total_kw = kw * sum(place)
    if "n_plus_1" not in relax and cdus:
        caps = [int(c["capacity_kw"]) for c in cdus]
        m.Add(total_kw <= sum(caps) - max(caps))

    if mode == "max_kw":
        m.Maximize(total_kw)
    elif mode == "min_hose":
        m.Add(total_kw >= int(math.ceil(target - EPS)))
        m.Minimize(sum(hose_cm[k] * v for k, v in assign.items()))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 8
    code = solver.Solve(m)
    wall = time.perf_counter() - t0
    status = STATUS.get(code, str(code))

    layout: dict[str, Any] = {"racks": []}
    if code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, s in enumerate(slots):
            if solver.Value(place[i]):
                c_idx = next(ci for (si, ci), v in assign.items() if si == i and solver.Value(v))
                layout["racks"].append({"id": f"R{len(layout['racks']) + 1:02d}", "type": scene["goal"]["rack"],
                                        "x": s.x, "y": s.y, "busway": s.busway, "cdu": cdus[c_idx]["id"]})
    result = {
        "mode": mode, "status": status, "wall_s": round(wall, 3), "solver_s": round(solver.WallTime(), 3),
        "n_slots": len(slots), "n_assign": len(assign), "target_kw": target,
        "objective": solver.ObjectiveValue() if code in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "total_kw": kw * len(layout["racks"]), "layout": layout, "relax": sorted(relax),
    }
    if mode == "min_hose" and result["objective"] is not None:
        result["objective"] = round(result["objective"] / 100, 2)
    if check and not relax and layout["racks"]:
        report = verify(scene, layout, target)
        ignore = ("total_power",) if mode == "max_kw" else ()
        bad = failed(report, ignore)
        assert not bad, f"verifier rejected solver layout ({mode}): {bad}"
        result["verify_pass"] = True
        result["verify_total_power"] = report["checks"]["total_power"]["pass"]
        result["verify_hose_m"] = report["total_hose_m"]
    return result


def binding(scene: dict[str, Any], base_kw: float, **kw: Any) -> dict[str, float]:
    """kW gained by relaxing each constraint family alone: positive means it binds."""
    gains = {}
    for fam in RELAXABLE:
        r = solve(scene, "max_kw", relax=frozenset({fam}), check=False, **kw)
        gains[fam] = r["total_kw"] - base_kw
    return gains


def capacity_bound_racks(scene: dict[str, Any]) -> int:
    """Geometry-free upper bound on racks; a target above it is infeasible on any grid."""
    kw = scene["rack_catalog"][scene["goal"]["rack"]]["kw"]
    caps = [c["capacity_kw"] for c in scene["cdus"]]
    by_busway = sum(int(math.floor(b["rating_kw"] * b["derate"] / kw + EPS)) for b in scene["busways"])
    by_cdu = sum(int(c // kw) for c in caps)
    by_n1 = int(math.floor((sum(caps) - max(caps)) / kw + EPS)) if caps else 0
    return min(by_busway, by_cdu, by_n1)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default=str(here / "scene_approx_data_hall.json"))
    ap.add_argument("--target-kw", type=float, default=None)
    ap.add_argument("--x-step", type=float, default=0.6)
    ap.add_argument("--y-step", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="directory to write layout JSON files")
    args = ap.parse_args()
    with open(args.scene) as f:
        scene = json.load(f)

    a = solve(scene, "max_kw", args.target_kw, args.x_step, args.y_step)
    b = solve(scene, "min_hose", args.target_kw, args.x_step, args.y_step)
    gains = binding(scene, a["total_kw"], x_step=args.x_step, y_step=args.y_step)
    summary = {
        "scene": scene["name"], "target_kw": b["target_kw"],
        "max_kw": {k: a[k] for k in ("status", "total_kw", "wall_s", "n_slots", "verify_pass")},
        "max_kw_racks_per_busway": _per(a["layout"], "busway"),
        "binding_gain_kw": gains,
        "min_hose": {k: b.get(k) for k in ("status", "total_kw", "objective", "wall_s", "verify_pass", "verify_hose_m")},
        "min_hose_racks_per_busway": _per(b["layout"], "busway"),
        "min_hose_racks_per_cdu": _per(b["layout"], "cdu"),
        "capacity_bound_racks": capacity_bound_racks(scene),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "layout_max_kw.json").write_text(json.dumps(a["layout"], indent=2))
        (out / "layout_min_hose.json").write_text(json.dumps(b["layout"], indent=2))


def _per(layout: dict[str, Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in layout["racks"]:
        counts[r[key]] = counts.get(r[key], 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
