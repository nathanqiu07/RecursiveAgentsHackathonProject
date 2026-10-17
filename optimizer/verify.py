"""Independent layout verifier for the data-hall layout benchmark.

Pure Python with no solver import: every check is re-derived from scene geometry,
so a bug in the solver's slot pruning cannot also hide here. An LLM agent's layout
and the solver's layout are judged by exactly the same code.

Usage: python verify.py scene.json layout.json [--target-kw N]
"""
from __future__ import annotations

import argparse
import json
from typing import Any

EPS = 1e-6
Rect = tuple[float, float, float, float]


def rect_at(x: float, y: float, w: float, d: float) -> Rect:
    """Axis-aligned footprint (x0, y0, x1, y1) centred on (x, y)."""
    return (x - w / 2, y - d / 2, x + w / 2, y + d / 2)


def span_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Signed 1-D overlap; negative values are the gap between the spans."""
    return min(a1, b1) - max(a0, b0)


def overlap_area(a: Rect, b: Rect) -> float:
    """Positive-area intersection only; touching edges are allowed (racks abut)."""
    ox = span_overlap(a[0], a[2], b[0], b[2])
    oy = span_overlap(a[1], a[3], b[1], b[3])
    return ox * oy if ox > EPS and oy > EPS else 0.0


def hose_run(x: float, y: float, cdu: dict[str, Any]) -> float:
    """Manhattan run from rack centre to CDU port (hoses follow orthogonal trays)."""
    return abs(x - cdu["port_x"]) + abs(y - cdu["port_y"])


def _check(ok: bool, value: Any, limit: Any, violations: list[Any]) -> dict[str, Any]:
    return {"pass": ok, "value": value, "limit": limit, "violations": violations[:20]}


def verify(scene: dict[str, Any], layout: dict[str, Any], target_kw: float | None = None) -> dict[str, Any]:
    """Return per-constraint pass/fail with magnitudes for one layout."""
    racks = layout.get("racks", [])
    catalog = scene["rack_catalog"]
    busways = {b["id"]: b for b in scene["busways"]}
    cdus = {c["id"]: c for c in scene["cdus"]}
    rtype = scene["goal"]["rack"]
    target = scene["goal"]["target_kw"] if target_kw is None else target_kw
    checks: dict[str, dict[str, Any]] = {}

    schema_errs: list[str] = []
    ids = [r.get("id") for r in racks]
    if len(set(ids)) != len(ids):
        schema_errs.append("duplicate rack ids")
    for r in racks:
        if r.get("type", rtype) not in catalog:
            schema_errs.append(f"{r['id']}: unknown rack type")
        if r.get("busway") not in busways:
            schema_errs.append(f"{r['id']}: unknown busway {r.get('busway')}")
        if r.get("cdu") not in cdus:
            schema_errs.append(f"{r['id']}: unknown cdu {r.get('cdu')}")
    checks["schema"] = _check(not schema_errs, len(racks), None, schema_errs)
    if schema_errs:
        return {"all_pass": False, "checks": checks, "total_kw": 0.0, "total_hose_m": 0.0}

    spec = {r["id"]: catalog[r.get("type", rtype)] for r in racks}
    rects = {r["id"]: rect_at(r["x"], r["y"], spec[r["id"]]["w"], spec[r["id"]]["d"]) for r in racks}
    W, D = scene["floor"]["width"], scene["floor"]["depth"]

    excursions = []
    for rid, (x0, y0, x1, y1) in rects.items():
        worst = max(-x0, -y0, x1 - W, y1 - D)
        if worst > EPS:
            excursions.append({"rack": rid, "out_m": round(worst, 3)})
    checks["in_bounds"] = _check(not excursions, max((e["out_m"] for e in excursions), default=0.0), 0.0, excursions)

    hits = []
    rid_list = list(rects)
    for i, a in enumerate(rid_list):
        for b in rid_list[i + 1:]:
            area = overlap_area(rects[a], rects[b])
            if area > 0:
                hits.append({"a": a, "b": b, "area_m2": round(area, 4)})
        for col in scene["columns"]:
            area = overlap_area(rects[a], rect_at(col["x"], col["y"], col["w"], col["d"]))
            if area > 0:
                hits.append({"a": a, "b": col["id"], "area_m2": round(area, 4)})
    checks["no_collision"] = _check(not hits, max((h["area_m2"] for h in hits), default=0.0), 0.0, hits)

    zone_hits = []
    for rid, rr in rects.items():
        for z in scene["zones"]:
            area = overlap_area(rr, (z["x0"], z["y0"], z["x1"], z["y1"]))
            if area > 0:
                zone_hits.append({"rack": rid, "zone": z["id"], "kind": z["kind"], "area_m2": round(area, 4)})
    checks["zone_clear"] = _check(not zone_hits, max((h["area_m2"] for h in zone_hits), default=0.0), 0.0, zone_hits)

    reach_bad, worst_reach = [], 0.0
    for r in racks:
        b = busways[r["busway"]]
        dy = abs(r["y"] - b["y"]) - b["tap_reach"]
        dx = max(b["x0"] - r["x"], r["x"] - b["x1"], 0.0)
        excess = max(dy, dx)
        worst_reach = max(worst_reach, excess)
        if excess > EPS:
            reach_bad.append({"rack": r["id"], "busway": b["id"], "excess_m": round(excess, 3)})
    checks["feed_reach"] = _check(not reach_bad, round(worst_reach, 3), 0.0, reach_bad)

    rows: dict[str, set[float]] = {}
    for r in racks:
        rows.setdefault(r["busway"], set()).add(round(r["y"], 4))
    crooked = [{"busway": b, "ys": sorted(ys)} for b, ys in rows.items() if len(ys) > 1]
    checks["row_alignment"] = _check(not crooked, len(crooked), 0, crooked)

    bw_load = {bid: 0.0 for bid in busways}
    for r in racks:
        bw_load[r["busway"]] += spec[r["id"]]["kw"]
    bw_over = []
    for bid, load in bw_load.items():
        usable = busways[bid]["rating_kw"] * busways[bid]["derate"]
        if load > usable + EPS:
            bw_over.append({"busway": bid, "load_kw": load, "usable_kw": usable})
    checks["busway_capacity"] = _check(
        not bw_over, {k: v for k, v in bw_load.items() if v},
        {k: round(b["rating_kw"] * b["derate"], 1) for k, b in busways.items()}, bw_over)

    cdu_load = {cid: 0.0 for cid in cdus}
    for r in racks:
        cdu_load[r["cdu"]] += spec[r["id"]]["kw"]
    cdu_over = [{"cdu": c, "load_kw": l, "cap_kw": cdus[c]["capacity_kw"]}
                for c, l in cdu_load.items() if l > cdus[c]["capacity_kw"] + EPS]
    checks["cdu_capacity"] = _check(not cdu_over, {k: v for k, v in cdu_load.items() if v},
                                    {k: c["capacity_kw"] for k, c in cdus.items()}, cdu_over)

    total_kw = sum(spec[r["id"]]["kw"] for r in racks)
    caps = [c["capacity_kw"] for c in cdus.values()]
    n1_cap = sum(caps) - max(caps) if caps else 0.0
    checks["n_plus_1"] = _check(total_kw <= n1_cap + EPS, total_kw, n1_cap,
                                [] if total_kw <= n1_cap + EPS else [{"short_kw": total_kw - n1_cap}])

    hose_bad, total_hose, worst_hose = [], 0.0, 0.0
    for r in racks:
        c = cdus[r["cdu"]]
        run = hose_run(r["x"], r["y"], c)
        total_hose += run
        worst_hose = max(worst_hose, run - c["hose_m"])
        if run > c["hose_m"] + EPS:
            hose_bad.append({"rack": r["id"], "cdu": c["id"], "run_m": round(run, 3), "limit_m": c["hose_m"]})
    checks["hose_length"] = _check(not hose_bad, round(worst_hose, 3), 0.0, hose_bad)

    aisle = scene["rules"]["aisle_min_m"]
    aisle_bad, min_gap = [], None
    for i, a in enumerate(rid_list):
        ra = rects[a]
        for b in rid_list[i + 1:]:
            rb = rects[b]
            if span_overlap(ra[0], ra[2], rb[0], rb[2]) <= EPS:
                continue
            gap = -span_overlap(ra[1], ra[3], rb[1], rb[3])
            min_gap = gap if min_gap is None else min(min_gap, gap)
            if gap < aisle - EPS:
                aisle_bad.append({"a": a, "b": b, "gap_m": round(gap, 3)})
    checks["aisle_width"] = _check(not aisle_bad, None if min_gap is None else round(min_gap, 3), aisle, aisle_bad)

    slab = scene["floor"]["slab_rating_kg_m2"]
    load_bad = []
    for r in racks:
        s = spec[r["id"]]
        q = s["kg"] / (s["w"] * s["d"])
        if q > slab + EPS:
            load_bad.append({"rack": r["id"], "kg_m2": round(q), "rating": slab})
    checks["floor_load"] = _check(not load_bad, round(max((spec[r["id"]]["kg"] / (spec[r["id"]]["w"] * spec[r["id"]]["d"]) for r in racks), default=0.0)), slab, load_bad)

    checks["total_power"] = _check(total_kw >= target - EPS, total_kw, target,
                                   [] if total_kw >= target - EPS else [{"short_kw": target - total_kw}])

    return {"all_pass": all(c["pass"] for c in checks.values()), "checks": checks,
            "total_kw": total_kw, "total_hose_m": round(total_hose, 3)}


def failed(report: dict[str, Any], ignore: tuple[str, ...] = ()) -> list[str]:
    """Names of failing checks, minus any the caller deliberately ignores."""
    return [k for k, c in report["checks"].items() if not c["pass"] and k not in ignore]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("layout")
    ap.add_argument("--target-kw", type=float, default=None)
    args = ap.parse_args()
    with open(args.scene) as f:
        scene = json.load(f)
    with open(args.layout) as f:
        layout = json.load(f)
    print(json.dumps(verify(scene, layout, args.target_kw), indent=2))


if __name__ == "__main__":
    main()
