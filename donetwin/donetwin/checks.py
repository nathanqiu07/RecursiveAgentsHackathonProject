"""Deterministic geometry verifier: the ground truth for "done".

Every check returns violations carrying a measured magnitude (meters) and a
concrete correction (a translation vector) the layout agent can apply.
"""
import math

from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import nearest_points

from .scene import PHYSICAL_KINDS, Scene

EPS = 1e-6

# Shown to the Spec agent so it can compose its own definition of done.
CHECK_LIBRARY = {
    "exists": {
        "params": {"id": "object id", "kind": "expected kind"},
        "doc": "The object must exist in the final layout (catches agents that silently drop the task).",
    },
    "no_collision": {
        "params": {},
        "doc": "No two physical footprints may intersect (pre-existing fixed/fixed pairs are ignored).",
    },
    "in_bounds": {
        "params": {},
        "doc": "Every footprint must lie inside the floor.",
    },
    "fixed_unchanged": {
        "params": {},
        "doc": "Fixed structures (pillars, conveyors, walkways, sensors) must not be moved or removed.",
    },
    "min_clearance": {
        "params": {"subject": "object id", "against_kinds": "list of kinds", "distance": "meters"},
        "doc": "Footprint gap between subject and every object of the given kinds must be >= distance "
               "(maintenance access, forklift aisles).",
    },
    "reach": {
        "params": {"robot": "robot id", "point": "'obj.point' ref", "margin": "fraction of max reach usable, e.g. 0.9"},
        "doc": "min_reach <= distance(robot base, point) <= reach * margin.",
    },
    "swing_clear": {
        "params": {"robot": "robot id", "from": "'obj.point'", "to": "'obj.point'",
                   "against_kinds": "kinds the arm cannot pass (full-height obstacles)", "arm_width": "meters"},
        "doc": "The arm's swept sector between the two points (in at least one rotation direction) "
               "must not intersect any full-height obstacle.",
    },
    "zone_clear": {
        "params": {"zone": "zone id", "include_envelopes": "bool"},
        "doc": "No new/movable footprint may intrude into the zone; with include_envelopes, no robot's "
               "max-reach envelope may overlap it either (people walk there).",
    },
    "aligned": {
        "params": {"a": "'obj.point' to adjust", "b": "'obj.point' reference", "tolerance": "meters"},
        "doc": "Point a must be within tolerance of point b (e.g. bin drop point under the fill sensor).",
    },
    "total_power": {
        "params": {"min_kw": "kW of IT load that must be deployed"},
        "doc": "Sum of rack kW must reach the requested deployment size.",
    },
    "capacity": {
        "params": {"kind": "'cdu' or 'busway'"},
        "doc": "Every rack has a valid feeder of that kind, and no feeder's assigned rack load exceeds its usable "
               "capacity (busways: capacity_kw * derate for continuous load).",
    },
    "redundancy": {
        "params": {"kind": "'cdu'"},
        "doc": "N+1: total rack load must fit in the combined capacity minus the largest unit.",
    },
    "hose_length": {
        "params": {"max_m": "optional override; default each CDU's hose_m"},
        "doc": "Manhattan hose run from each rack to its assigned CDU port must not exceed the hose length.",
    },
    "feed_reach": {
        "params": {"max_m": "optional override; default each busway's tap_m"},
        "doc": "Each rack must sit within tap-whip reach of its assigned overhead busway.",
    },
    "aisle_width": {
        "params": {"min_m": "minimum aisle between rack rows, e.g. 1.2"},
        "doc": "Racks either abut in a row or are separated by at least a full aisle.",
    },
}

MANDATORY = [
    {"type": "no_collision", "params": {}, "why": "mandatory: physical collisions are never acceptable"},
    {"type": "in_bounds", "params": {}, "why": "mandatory: the layout must fit the floor"},
    {"type": "fixed_unchanged", "params": {}, "why": "mandatory: structural elements are immovable"},
]


def footprint(o):
    p = box(o["x"] - o["w"] / 2, o["y"] - o["d"] / 2, o["x"] + o["w"] / 2, o["y"] + o["d"] / 2)
    if o.get("rot"):
        p = affinity.rotate(p, o["rot"], origin=(o["x"], o["y"]))
    return p


def envelope(robot):
    return Point(robot["x"], robot["y"]).buffer(robot["reach"], 96)


def sector(cx, cy, r, a0, a1, n=72):
    """Pie slice swept counter-clockwise from angle a0 to a1 (degrees)."""
    span = (a1 - a0) % 360
    pts = [(cx, cy)]
    for i in range(n + 1):
        a = math.radians(a0 + span * i / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return Polygon(pts)


def _vec(dx, dy):
    return f"(dx={dx:+.2f}, dy={dy:+.2f})"


def _mtv_options(a, b):
    """Axis translations that move polygon a fully out of polygon b, smallest first."""
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = b.bounds
    opts = [(bx1 - ax0, 1, 0), (ax1 - bx0, -1, 0), (by1 - ay0, 0, 1), (ay1 - by0, 0, -1)]
    return sorted(opts)


def _physical(scene):
    return [o for o in scene.data["objects"] if o["kind"] in PHYSICAL_KINDS]


def _fixed_in_base(base, obj_id):
    return base.has(obj_id) and base.get(obj_id).get("fixed", False)


# --------------------------------------------------------------------------- checks

def check_exists(scene, base, p):
    if scene.has(p["id"]) and scene.get(p["id"])["kind"] == p.get("kind", scene.get(p["id"])["kind"]):
        return []
    return [{"objects": [p["id"]], "magnitude": 1.0,
             "detail": f"{p['id']} ({p.get('kind', '?')}) is missing from the layout",
             "fix": f"add {p['id']} as specified"}]


def check_no_collision(scene, base, p):
    out = []
    objs = _physical(scene)
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            if _fixed_in_base(base, a["id"]) and _fixed_in_base(base, b["id"]):
                continue
            inter = footprint(a).intersection(footprint(b)).area
            if inter <= EPS:
                continue
            # move the movable one; if both movable, move the later-added one
            mover, other = (b, a) if not b.get("fixed") else (a, b)
            opts = _mtv_options(footprint(mover), footprint(other))[:2]
            fixes = " or ".join(f"translate {mover['id']} by {_vec(m * sx + 0.05 * sx, m * sy + 0.05 * sy)}"
                                for m, sx, sy in opts)
            out.append({"objects": [a["id"], b["id"]], "magnitude": round(opts[0][0], 3),
                        "detail": f"{a['id']} footprint collides with {b['id']} "
                                  f"({inter:.2f} m2 overlap, {opts[0][0]:.2f} m penetration)",
                        "fix": fixes})
    return out


def check_in_bounds(scene, base, p):
    out = []
    W, D = scene.floor_w, scene.floor_d
    for o in scene.data["objects"]:
        x0, y0, x1, y1 = footprint(o).bounds
        dx = (-x0 if x0 < -EPS else 0) + (W - x1 if x1 > W + EPS else 0)
        dy = (-y0 if y0 < -EPS else 0) + (D - y1 if y1 > D + EPS else 0)
        if dx or dy:
            out.append({"objects": [o["id"]], "magnitude": round(math.hypot(dx, dy), 3),
                        "detail": f"{o['id']} extends outside the {W}x{D} m floor",
                        "fix": f"translate {o['id']} by {_vec(dx, dy)}"})
    return out


def check_fixed_unchanged(scene, base, p):
    out = []
    for o in base.data["objects"]:
        if not o.get("fixed"):
            continue
        if not scene.has(o["id"]):
            out.append({"objects": [o["id"]], "magnitude": 1.0,
                        "detail": f"fixed {o['kind']} {o['id']} was REMOVED",
                        "fix": f"restore {o['id']} at x={o['x']}, y={o['y']} and design around it"})
            continue
        n = scene.get(o["id"])
        d = math.hypot(n["x"] - o["x"], n["y"] - o["y"])
        if d > 1e-3 or abs(n.get("rot", 0) - o.get("rot", 0)) > 1e-3:
            out.append({"objects": [o["id"]], "magnitude": round(d, 3),
                        "detail": f"fixed {o['kind']} {o['id']} was moved by {d:.2f} m",
                        "fix": f"move {o['id']} back to x={o['x']}, y={o['y']}, rot={o.get('rot', 0)}"})
    return out


def check_min_clearance(scene, base, p):
    out = []
    if not scene.has(p["subject"]):
        return check_exists(scene, base, {"id": p["subject"]})
    s = scene.get(p["subject"])
    sf = footprint(s)
    need = float(p["distance"])
    for o in scene.data["objects"]:
        if o["id"] == s["id"] or o["kind"] not in p["against_kinds"]:
            continue
        of = footprint(o)
        gap = sf.distance(of)
        if gap >= need - EPS:
            continue
        if gap > EPS:
            q_obs, q_sub = nearest_points(of, sf)
            ux, uy = q_sub.x - q_obs.x, q_sub.y - q_obs.y
            L = math.hypot(ux, uy)
            k = (need - gap + 0.02) / L
            fix = f"translate {s['id']} by {_vec(ux * k, uy * k)}"
        else:
            m, sx, sy = _mtv_options(sf, of)[0]
            fix = f"translate {s['id']} by {_vec((m + need + 0.02) * sx, (m + need + 0.02) * sy)}"
        out.append({"objects": [s["id"], o["id"]], "magnitude": round(need - gap, 3),
                    "detail": f"{s['id']} is {gap:.2f} m from {o['id']}; needs >= {need:.2f} m",
                    "fix": fix})
    return out


def check_reach(scene, base, p):
    if not scene.has(p["robot"]):
        return check_exists(scene, base, {"id": p["robot"], "kind": "robot"})
    r = scene.get(p["robot"])
    px, py = scene.point(p["point"])
    dx, dy = px - r["x"], py - r["y"]
    dist = math.hypot(dx, dy)
    limit = r["reach"] * float(p.get("margin", 0.9))
    if dist > limit + EPS:
        k = (dist - limit + 0.02) / dist
        return [{"objects": [r["id"], p["point"].split(".")[0]], "magnitude": round(dist - limit, 3),
                 "detail": f"{p['point']} is {dist:.2f} m from {r['id']} base; usable reach is "
                           f"{limit:.2f} m ({r['reach']} m x {p.get('margin', 0.9)})",
                 "fix": f"translate {r['id']} toward it by {_vec(dx * k, dy * k)}"}]
    if dist < r["min_reach"] - EPS:
        k = -(r["min_reach"] - dist + 0.02) / max(dist, 1e-3)
        return [{"objects": [r["id"], p["point"].split(".")[0]], "magnitude": round(r["min_reach"] - dist, 3),
                 "detail": f"{p['point']} is only {dist:.2f} m from {r['id']}; inside its "
                           f"{r['min_reach']} m dead zone",
                 "fix": f"translate {r['id']} away by {_vec(dx * k, dy * k)}"}]
    return []


def swing_hits(scene, robot, from_ref, to_ref, against_kinds, arm_width):
    """Obstacles hit by the counter-clockwise and clockwise sweeps from one point to the other."""
    fx, fy = scene.point(from_ref)
    tx, ty = scene.point(to_ref)
    cx, cy = robot["x"], robot["y"]
    a_from = math.degrees(math.atan2(fy - cy, fx - cx))
    a_to = math.degrees(math.atan2(ty - cy, tx - cx))
    rad = max(math.hypot(fx - cx, fy - cy), math.hypot(tx - cx, ty - cy)) + arm_width / 2
    sweeps = {"ccw": sector(cx, cy, rad, a_from, a_to).buffer(arm_width / 2),
              "cw": sector(cx, cy, rad, a_to, a_from).buffer(arm_width / 2)}
    hits = {}
    for name, poly in sweeps.items():
        hits[name] = [o["id"] for o in scene.data["objects"]
                      if o["kind"] in against_kinds and o["id"] != robot["id"]
                      and poly.intersects(footprint(o))]
    return hits, (a_from, a_to, rad)


def check_swing_clear(scene, base, p):
    if not scene.has(p["robot"]):
        return check_exists(scene, base, {"id": p["robot"], "kind": "robot"})
    r = scene.get(p["robot"])
    kinds = p.get("against_kinds", ["pillar", "fence", "rack"])
    hits, (a0, a1, rad) = swing_hits(scene, r, p["from"], p["to"], kinds, float(p.get("arm_width", 0.4)))
    if not hits["ccw"] or not hits["cw"]:
        return []
    blockers = sorted(set(hits["ccw"]) | set(hits["cw"]))
    locs = "; ".join(f"{b} at ({scene.get(b)['x']:.1f}, {scene.get(b)['y']:.1f})" for b in blockers)
    return [{"objects": [r["id"]] + blockers, "magnitude": 0.5 * len(blockers),
             "detail": f"arm sweep {p['from']} -> {p['to']} (radius {rad:.2f} m, {a0:.0f} deg -> {a1:.0f} deg) "
                       f"hits {', '.join(hits['ccw'])} going CCW and {', '.join(hits['cw'])} going CW",
             "fix": f"relocate {r['id']} so that no full-height obstacle lies inside the swept sector "
                    f"in at least one direction; obstacles: {locs}"}]


def check_zone_clear(scene, base, p):
    out = []
    z = footprint(scene.get(p["zone"]))
    for o in scene.data["objects"]:
        if o["kind"] not in PHYSICAL_KINDS or _fixed_in_base(base, o["id"]):
            continue
        f = footprint(o)
        if f.intersects(z) and f.intersection(z).area > EPS:
            m, sx, sy = _mtv_options(f, z)[0]
            out.append({"objects": [o["id"], p["zone"]], "magnitude": round(m, 3),
                        "detail": f"{o['id']} footprint intrudes {m:.2f} m into {p['zone']}",
                        "fix": f"translate {o['id']} by {_vec((m + 0.05) * sx, (m + 0.05) * sy)}"})
        if p.get("include_envelopes", True) and o["kind"] == "robot":
            c = Point(o["x"], o["y"])
            gap = c.distance(z)
            if gap < o["reach"] - EPS:
                q_z, q_c = nearest_points(z, c)
                ux, uy = q_c.x - q_z.x, q_c.y - q_z.y
                L = math.hypot(ux, uy) or 1.0
                k = (o["reach"] - gap + 0.05) / L
                out.append({"objects": [o["id"], p["zone"]], "magnitude": round(o["reach"] - gap, 3),
                            "detail": f"{o['id']} max-reach envelope (r={o['reach']} m) extends "
                                      f"{o['reach'] - gap:.2f} m into {p['zone']}: people there are inside "
                                      f"the robot's restricted space (ISO 10218-2)",
                            "fix": f"translate {o['id']} by {_vec(ux * k, uy * k)}"})
    return out


def check_aligned(scene, base, p):
    ax, ay = scene.point(p["a"])
    bx, by = scene.point(p["b"])
    d = math.hypot(bx - ax, by - ay)
    tol = float(p.get("tolerance", 0.05))
    if d <= tol + EPS:
        return []
    obj = p["a"].split(".")[0]
    return [{"objects": [obj, p["b"].split(".")[0]], "magnitude": round(d, 3),
             "detail": f"{p['a']} is {d:.3f} m from {p['b']} (tolerance {tol:.3f} m): items will miss",
             "fix": f"translate {obj} by {_vec(bx - ax, by - ay)}"}]


# --------------------------------------------------------------------------- data hall checks

def _racks(scene):
    return [o for o in scene.data["objects"] if o["kind"] == "rack"]


def _feeders(scene, kind):
    return {o["id"]: o for o in scene.data["objects"] if o["kind"] == kind}


def _usable(f):
    return f["capacity_kw"] * (f.get("derate", 1.0) if f["kind"] == "busway" else 1.0)


def feeder_loads(scene_dict):
    """{feeder_id: {"load_kw", "cap_kw"}} for every CDU and busway (also used by the UI)."""
    s = Scene(scene_dict)
    out = {}
    for kind in ("cdu", "busway"):
        for fid, f in _feeders(s, kind).items():
            load = sum(r["kw"] for r in _racks(s) if r.get(kind) == fid)
            out[fid] = {"load_kw": load, "cap_kw": round(_usable(f), 1)}
    return out


def check_total_power(scene, base, p):
    racks = _racks(scene)
    kw = sum(r["kw"] for r in racks)
    need = float(p["min_kw"])
    if kw >= need - EPS:
        return []
    per = racks[0]["kw"] if racks else 132
    return [{"objects": [], "magnitude": round((need - kw) / 1000, 3),
             "detail": f"{len(racks)} racks = {kw:.0f} kW of IT load; the goal needs >= {need:.0f} kW",
             "fix": f"add {math.ceil((need - kw) / per)} more rack(s)"}]


def check_capacity(scene, base, p):
    kind = p["kind"]
    feeders = _feeders(scene, kind)
    loads = feeder_loads(scene.data)
    spare = ", ".join(f"{fid} {loads[fid]['cap_kw'] - loads[fid]['load_kw']:.0f} kW"
                      for fid in feeders if loads[fid]["cap_kw"] - loads[fid]["load_kw"] > 0)
    out = []
    groups = {}
    for r in _racks(scene):
        groups.setdefault(r.get(kind), []).append(r)
    for fid, racks in groups.items():
        ids = [r["id"] for r in racks]
        if fid not in feeders:
            out.append({"objects": ids, "magnitude": 1.0,
                        "detail": f"{len(ids)} rack(s) have no valid {kind} ({fid!r}): {', '.join(ids[:8])}",
                        "fix": f"assign each rack one of {sorted(feeders)}"})
            continue
        f = feeders[fid]
        load, cap = sum(r["kw"] for r in racks), _usable(f)
        if load > cap + EPS:
            derate = f" ({f['capacity_kw']} kW x {f['derate']} continuous-load derate)" if kind == "busway" else ""
            out.append({"objects": [fid] + ids, "magnitude": round((load - cap) / 100, 3),
                        "detail": f"{fid} carries {load:.0f} kW from {len(ids)} racks; limit {cap:.0f} kW{derate}",
                        "fix": f"move at least {math.ceil((load - cap) / racks[0]['kw'])} rack(s) off {fid}; "
                               f"spare capacity: {spare or 'none'}"})
    return out


def check_redundancy(scene, base, p):
    feeders = _feeders(scene, p.get("kind", "cdu"))
    if not feeders:
        return []
    caps = [_usable(f) for f in feeders.values()]
    n1 = sum(caps) - max(caps)
    load = sum(r["kw"] for r in _racks(scene))
    if load <= n1 + EPS:
        return []
    return [{"objects": sorted(feeders), "magnitude": round((load - n1) / 100, 3),
             "detail": f"total load {load:.0f} kW exceeds N+1 capacity {n1:.0f} kW (all {p.get('kind', 'cdu')}s "
                       f"minus the largest): one failure would overheat racks",
             "fix": f"reduce IT load by {load - n1:.0f} kW or add cooling capacity"}]


def _hose(scene, rack, cdu):
    px, py = scene.point(f"{cdu['id']}.port") if "port" in cdu.get("points", {}) else (cdu["x"], cdu["y"])
    return abs(rack["x"] - px) + abs(rack["y"] - py)


def check_hose_length(scene, base, p):
    cdus = _feeders(scene, "cdu")
    out = []
    for r in _racks(scene):
        c = cdus.get(r.get("cdu"))
        if c is None:
            continue  # reported by capacity(cdu)
        L, lim = _hose(scene, r, c), float(p.get("max_m", c["hose_m"]))
        if L <= lim + EPS:
            continue
        alts = sorted((round(_hose(scene, r, o), 1), oid) for oid, o in cdus.items() if oid != c["id"])
        ok = [f"{oid} ({d} m)" for d, oid in alts if d <= lim]
        out.append({"objects": [r["id"], c["id"]], "magnitude": round(L - lim, 3),
                    "detail": f"{r['id']} -> {c['id']} hose run {L:.1f} m exceeds {lim:.1f} m",
                    "fix": f"reassign {r['id']} to " + (" or ".join(ok[:2]) if ok else
                                                         f"a closer CDU; nearest is {alts[0][1]} ({alts[0][0]} m)")})
    return out


def check_feed_reach(scene, base, p):
    bws = _feeders(scene, "busway")
    out = []
    for r in _racks(scene):
        b = bws.get(r.get("busway"))
        if b is None:
            continue  # reported by capacity(busway)
        lim = float(p.get("max_m", b.get("tap_m", 0.8)))
        gap = Point(r["x"], r["y"]).distance(footprint(b))
        if gap <= lim + EPS:
            continue
        x0, y0, x1, y1 = footprint(b).bounds
        tx = min(max(r["x"], x0 + 0.3), x1 - 0.3)
        out.append({"objects": [r["id"], b["id"]], "magnitude": round(gap - lim, 3),
                    "detail": f"{r['id']} is {gap:.2f} m from busway {b['id']}; tap whips reach {lim:.2f} m",
                    "fix": f"move {r['id']} under {b['id']} (y={b['y']:.2f}, x within {x0:.1f}..{x1:.1f}), "
                           f"e.g. translate by {_vec(tx - r['x'], b['y'] - r['y'])}"})
    return out


def check_aisle_width(scene, base, p):
    need = float(p.get("min_m", 1.2))
    racks = _racks(scene)
    out = []
    for i, a in enumerate(racks):
        for b in racks[i + 1:]:
            ax0, ay0, ax1, ay1 = footprint(a).bounds
            bx0, by0, bx1, by1 = footprint(b).bounds
            facing_y = min(ax1, bx1) - max(ax0, bx0) > 0.05  # fronts face each other across an aisle
            facing_x = min(ay1, by1) - max(ay0, by0) > 0.05 and abs(a.get("rot", 0) - 90) % 180 < 1
            if not (facing_y or facing_x):
                continue
            gap = footprint(a).distance(footprint(b))
            if 0.02 < gap < need - EPS:
                out.append({"objects": [a["id"], b["id"]], "magnitude": round(need - gap, 3),
                            "detail": f"aisle between {a['id']} and {b['id']} is {gap:.2f} m; needs >= {need:.2f} m "
                                      "(racks in one row must abut; rows need a full aisle)",
                            "fix": f"abut them in one row or separate the rows by {need - gap:.2f} m more"})
    return out[:10]


CHECKS = {
    "exists": check_exists,
    "no_collision": check_no_collision,
    "in_bounds": check_in_bounds,
    "fixed_unchanged": check_fixed_unchanged,
    "min_clearance": check_min_clearance,
    "reach": check_reach,
    "swing_clear": check_swing_clear,
    "zone_clear": check_zone_clear,
    "aligned": check_aligned,
    "total_power": check_total_power,
    "capacity": check_capacity,
    "redundancy": check_redundancy,
    "hose_length": check_hose_length,
    "feed_reach": check_feed_reach,
    "aisle_width": check_aisle_width,
}


def verify(scene_dict, base_dict, checks):
    scene, base = Scene(scene_dict), Scene(base_dict)
    results = []
    for c in checks:
        fn = CHECKS.get(c["type"])
        if fn is None:
            results.append({**c, "passed": False, "violations": [
                {"objects": [], "magnitude": 1.0, "detail": f"unknown check type {c['type']}", "fix": None}]})
            continue
        try:
            v = fn(scene, base, c.get("params", {}))
        except Exception as e:  # a malformed check or missing object is a failure, not a crash
            v = [{"objects": [], "magnitude": 1.0, "detail": f"check error: {e}", "fix": None}]
        results.append({**c, "passed": not v, "violations": v})
    failing = [r for r in results if not r["passed"]]
    penalty = sum(1 + sum(v["magnitude"] for v in r["violations"]) for r in failing)
    return {"checks": results, "passed": len(results) - len(failing), "total": len(results),
            "all_pass": not failing, "penalty": round(penalty, 3)}


def gap_between(scene_dict, a_id, b_id):
    """Measured footprint gap in meters, used to adjudicate vision-inspector claims."""
    s = Scene(scene_dict)
    if not (s.has(a_id) and s.has(b_id)):
        return None
    return footprint(s.get(a_id)).distance(footprint(s.get(b_id)))
