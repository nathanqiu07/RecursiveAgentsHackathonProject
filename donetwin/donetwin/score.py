"""Benchmark scorer for data-hall layouts.

A solver submits only decisions ({"racks": [...]}); the scene is rebuilt from the
fixed hall + catalog truth, checked against hard rules (any violation => 0), then
scored on density, CDU-failure resilience and hose economy.

CLI: python -m donetwin.score <hall.json> <payload.json>
"""
import copy
import json
import math
import re
import sys

from shapely.geometry import Point

from .checks import EPS, footprint
from .scene import PHYSICAL_KINDS, RACK_CATALOG, ZONE_KINDS, Scene

COOLING = {"GB200-NVL72": "liquid", "GB300-NVL72": "liquid", "HGX-B200": "air"}
RACK_KEYS = ("id", "model", "x", "y", "rot", "cdu", "feed_a", "feed_b")
REQUIRED = ("id", "model", "x", "y", "rot")
MAX_RACKS = 300
ID_RE = re.compile(r"[A-Za-z0-9_-]{1,32}")
ROTS = (0, 90, 180, 270)
ABUT, FACE_OV, FACE_MAX = 0.02, 0.05, 4.0
EXACT_MAX = 14
RULES = ["payload", "no_collision", "in_bounds", "zones_clear", "cooling_type", "dual_feed", "cdu_capacity",
         "power_2n", "air_capacity", "hose", "aisle_width", "hot_cold_aisle"]  # severity order for feedback


def _v(detail, magnitude, fix):
    return {"detail": detail, "magnitude": round(float(magnitude), 3), "fix": fix}


def _vec(dx, dy):
    return f"(dx={dx:+.2f}, dy={dy:+.2f})"


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _kind(hall, kind):
    return [o for o in hall.get("objects", []) if o.get("kind") == kind]


def _usable(b):
    return b["capacity_kw"] * b.get("derate", 1.0)


def upper_bound_kw(hall):
    """Max deployable IT kW: min(2N busway budget = sum(capacity*derate)/2, total CDU capacity + room air).

    N+1 cooling is not a hard rule (only per-CDU capacity is); it is rewarded by the resilience term,
    so the bound uses the full CDU capacity and no feasible layout can exceed it.
    """
    power = sum(_usable(b) for b in _kind(hall, "busway")) / 2
    cool = sum(c["capacity_kw"] for c in _kind(hall, "cdu")) + hall.get("air_cooling_kw", 0.0)
    return round(float(min(power, cool)), 3)


# --------------------------------------------------------------------------- payload -> scene

def _validate(hall, payload):
    errs, racks = [], []
    allowed = ", ".join(RACK_KEYS)
    if not isinstance(payload, dict):
        return racks, [("payload must be a JSON object", 'send {"racks": [...]}')]
    for k in payload:
        if k != "racks":
            errs.append((f"unknown top-level key {k!r}", f'remove {k!r}; the payload holds only "racks" (the hall is fixed)'))
    rl = payload.get("racks")
    if not isinstance(rl, list):
        return racks, errs + [('"racks" must be a list of rack objects', 'send "racks": [{...}, ...]')]
    if len(rl) > MAX_RACKS:
        errs.append((f"{len(rl)} racks exceeds the {MAX_RACKS}-rack limit", f"submit at most {MAX_RACKS} racks"))
        rl = rl[:MAX_RACKS]
    hall_ids = {o["id"] for o in hall.get("objects", [])}
    seen = set()
    for i, r in enumerate(rl):
        tag = f"racks[{i}]"
        if not isinstance(r, dict):
            errs.append((f"{tag} is not an object", f"make {tag} an object with keys {allowed}"))
            continue
        n0 = len(errs)
        for k in r:
            if k not in RACK_KEYS:
                errs.append((f"{tag}: forbidden key {k!r}",
                             f"remove {k!r}; allowed keys: {allowed} (kW, size and capacities come from the catalog/hall)"))
        missing = [k for k in REQUIRED if k not in r]
        if missing:
            errs.append((f"{tag}: missing {', '.join(missing)}", f"add {', '.join(missing)} to {tag}"))
            continue
        rid = r["id"]
        if not isinstance(rid, str) or not ID_RE.fullmatch(rid):
            errs.append((f"{tag}: id {rid!r} must match ^[A-Za-z0-9_-]{{1,32}}$", f"rename {tag}, e.g. R{i + 1:02d}"))
        elif rid in hall_ids:
            errs.append((f"{tag}: id {rid!r} clashes with a hall object", f"rename {rid}, e.g. R{i + 1:02d}"))
        elif rid in seen:
            errs.append((f"{tag}: duplicate id {rid!r}", f"give {tag} a unique id"))
        if not isinstance(r["model"], str) or r["model"] not in RACK_CATALOG:
            errs.append((f"{tag}: unknown model {r['model']!r}", f"use one of {', '.join(RACK_CATALOG)}"))
        for k in ("x", "y"):
            if not _num(r[k]):
                errs.append((f"{tag}: {k} must be a finite number, got {r[k]!r}", f"set {k} to a number in meters"))
        if not _num(r["rot"]) or r["rot"] not in ROTS:
            errs.append((f"{tag}: rot {r['rot']!r} not in {list(ROTS)}", "set rot to 0, 90, 180 or 270"))
        for k in ("cdu", "feed_a", "feed_b"):
            if r.get(k) is not None and not isinstance(r[k], str):
                errs.append((f"{tag}: {k} must be an id string or null", f"set {k} to an object id"))
        if len(errs) == n0:
            seen.add(rid)
            racks.append({k: r.get(k) for k in RACK_KEYS})
    return racks, errs


def _rebuild(hall, payload):
    racks, errs = _validate(hall, payload)
    data = copy.deepcopy(hall)
    objs = data.setdefault("objects", [])
    for r in racks:
        s = RACK_CATALOG[r["model"]]
        objs.append({"id": r["id"], "kind": "rack", "model": r["model"], "x": float(r["x"]), "y": float(r["y"]),
                     "rot": float(r["rot"]), "w": s["w"], "d": s["d"], "h": s["h"], "kw": s["kw"], "kg": s["kg"],
                     "cooling": COOLING[r["model"]], "cdu": r["cdu"], "feed_a": r["feed_a"], "feed_b": r["feed_b"],
                     "fixed": False})
    return data, errs


def rebuild(hall, payload):
    data, errs = _rebuild(hall, payload)
    return data, [d for d, _ in errs]


# --------------------------------------------------------------------------- geometry helpers

def _front(rot):
    a = math.radians(rot)
    return (round(-math.sin(a)), round(math.cos(a)))


def _axis(rot):  # row axis: 0 = x (rot 0/180), 1 = y (rot 90/270)
    return 0 if round(rot) % 180 == 0 else 1


def _ov(a, b, k):
    return min(a[k + 2], b[k + 2]) - max(a[k], b[k])


def _gap(a, b):
    return math.hypot(max(0, b[0] - a[2], a[0] - b[2]), max(0, b[1] - a[3], a[1] - b[3]))


def _mtv(a, b):  # axis moves of bounds a out of bounds b, smallest first
    return sorted([(b[2] - a[0], 1, 0), (a[2] - b[0], -1, 0), (b[3] - a[1], 0, 1), (a[3] - b[1], 0, -1)])


def _move_fix(oid, a, b, n=2):
    return " or ".join(f"translate {oid} by {_vec((m + 0.01) * sx, (m + 0.01) * sy)}" for m, sx, sy in _mtv(a, b)[:n])


def _hose(c, r, cid):
    px, py = c["ports"][cid]
    return abs(r["x"] - px) + abs(r["y"] - py)


# --------------------------------------------------------------------------- hard rules

def r_no_collision(c):
    out, B = [], c["B"]
    fixed = [o for o in c["hall_objs"] if o["kind"] in PHYSICAL_KINDS]
    for r in c["racks"]:
        fr = footprint(r)
        for o in fixed:
            fo = footprint(o)
            inter = fr.intersection(fo).area
            if inter > EPS:
                out.append(_v(f"{r['id']} collides with {o['kind']} {o['id']} ({inter:.2f} m2 overlap)",
                              _mtv(fr.bounds, fo.bounds)[0][0], _move_fix(r["id"], fr.bounds, fo.bounds)))
    rs = c["racks"]
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            ba, bb = B[a["id"]], B[b["id"]]
            ox, oy = _ov(ba, bb, 0), _ov(ba, bb, 1)
            if ox > 1e-6 and oy > 1e-6 and ox * oy > EPS:
                out.append(_v(f"{b['id']} collides with rack {a['id']} ({ox * oy:.2f} m2 overlap)",
                              _mtv(bb, ba)[0][0], _move_fix(b["id"], bb, ba)))
    return out


def r_in_bounds(c):
    W, D = c["hall"]["floor"]["w"], c["hall"]["floor"]["d"]
    out = []
    for r in c["racks"]:
        x0, y0, x1, y1 = c["B"][r["id"]]
        dx = (-x0 if x0 < -EPS else 0) + (W - x1 if x1 > W + EPS else 0)
        dy = (-y0 if y0 < -EPS else 0) + (D - y1 if y1 > D + EPS else 0)
        if dx or dy:
            out.append(_v(f"{r['id']} extends outside the {W}x{D} m floor", math.hypot(dx, dy),
                          f"move {r['id']} to x={r['x'] + dx:.2f}, y={r['y'] + dy:.2f}"))
    return out


def r_zones_clear(c):
    out = []
    zones = [(o, footprint(o)) for o in c["hall_objs"] if o["kind"] in ZONE_KINDS]
    for r in c["racks"]:
        fr = footprint(r)
        for z, fz in zones:
            if fr.intersection(fz).area > EPS:
                m = _mtv(fr.bounds, fz.bounds)[0][0]
                out.append(_v(f"{r['id']} intrudes {m:.2f} m into {z['kind']} {z['id']}", m,
                              _move_fix(r["id"], fr.bounds, fz.bounds)))
    return out


def r_aisle_width(c):
    need, B, out = c["hall"].get("aisle_min_m", 1.2), c["B"], []
    rs = c["racks"]
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            ba, bb = B[a["id"]], B[b["id"]]
            ka, kb = _axis(a["rot"]), _axis(b["rot"])
            for k in ([ka] if ka == kb else [0, 1]):
                if _ov(ba, bb, k) <= FACE_OV:
                    continue
                x = 1 - k  # aisle runs across the other axis
                s = 1 if bb[x] >= ba[x + 2] - EPS else -1
                g = bb[x] - ba[x + 2] if s > 0 else ba[x] - bb[x + 2]
                if ABUT < g < need - EPS:
                    ax, cb = "xy"[x], (b["x"], b["y"])[x]
                    out.append(_v(f"aisle between {a['id']} and {b['id']} is {g:.2f} m; needs >= {need:.2f} m "
                                  "(facing rows need a full aisle; back-to-back rows abut)", need - g,
                                  f"move {b['id']} to {ax}={cb + s * (need - g + 0.01):.2f} (full aisle) or "
                                  f"{ax}={cb - s * g:.2f} (abut {a['id']})"))
                break
    return out


def r_cooling_type(c):
    out = []
    for r in c["racks"]:
        if r["cooling"] == "liquid" and r["cdu"] not in c["cdus"]:
            alts = sorted((_hose(c, r, k), k) for k in c["cdus"])
            fix = f"assign {r['id']} to {alts[0][1]} ({alts[0][0]:.1f} m hose)" if alts else "use an air-cooled model"
            out.append(_v(f"{r['id']} ({r['model']}) is liquid-cooled but cdu={r['cdu']!r} is not a CDU in this hall",
                          1.0, fix))
        elif r["cooling"] == "air" and r["cdu"] is not None:
            out.append(_v(f"{r['id']} ({r['model']}) is air-cooled but has cdu={r['cdu']!r}", 1.0,
                          f"set {r['id']} cdu to null (air racks draw on the room's air_cooling_kw)"))
    return out


def _cdu_loads(c):
    loads = {k: 0.0 for k in c["cdus"]}
    for r in c["racks"]:
        if r["cooling"] == "liquid" and r["cdu"] in loads:
            loads[r["cdu"]] += r["kw"]
    return loads


def _min_moves(kws, excess):
    n = 0
    for kw in sorted(kws, reverse=True):
        if excess <= EPS:
            break
        excess -= kw
        n += 1
    return n


def r_cdu_capacity(c):
    loads, out = _cdu_loads(c), []
    spare = ", ".join(f"{k} {c['cdus'][k]['capacity_kw'] - l:.0f} kW" for k, l in sorted(loads.items())
                      if c["cdus"][k]["capacity_kw"] - l > 0)
    for k, l in sorted(loads.items()):
        cap = c["cdus"][k]["capacity_kw"]
        if l > cap + EPS:
            kws = [r["kw"] for r in c["racks"] if r["cooling"] == "liquid" and r["cdu"] == k]
            out.append(_v(f"{k} cools {l:.0f} kW from {len(kws)} racks; capacity {cap:.0f} kW", l - cap,
                          f"move at least {_min_moves(kws, l - cap)} rack(s) off {k}; spare: {spare or 'none'}"))
    return out


def r_hose(c):
    out = []
    for r in c["racks"]:
        if r["cooling"] != "liquid" or r["cdu"] not in c["cdus"]:
            continue
        cdu = c["cdus"][r["cdu"]]
        L, lim = _hose(c, r, r["cdu"]), cdu["hose_m"]
        if L <= lim + EPS:
            continue
        ok = sorted((_hose(c, r, k), k) for k, o in c["cdus"].items() if k != r["cdu"] and _hose(c, r, k) <= o["hose_m"])
        px, py = c["ports"][r["cdu"]]
        fix = (f"reassign {r['id']} to {ok[0][1]} ({ok[0][0]:.1f} m)" if ok else
               f"move {r['id']} within {lim:.1f} m (Manhattan) of {r['cdu']} port at ({px:.2f}, {py:.2f})")
        out.append(_v(f"{r['id']} -> {r['cdu']} hose run {L:.1f} m exceeds {lim:.1f} m", L - lim, fix))
    return out


def _pair_fix(c, r):
    """Suggest the nearest A/B busway pair (grouped by 'row') for rack r."""
    groups = {}
    for bid, b in sorted(c["bws"].items()):
        groups.setdefault(b.get("row", bid), []).append(bid)
    best = None
    for ids in groups.values():
        if len(ids) < 2:
            continue
        ids = sorted(ids, key=lambda i: (c["bws"][i].get("feed", "Z"), i))[:2]
        d = max(Point(r["x"], r["y"]).distance(c["bwf"][i]) for i in ids)
        if best is None or d < best[0]:
            best = (d, ids)
    if best is None:
        return "this hall has no busway pair; 2N power is impossible here"
    d, (fa, fb) = best
    tap = min(c["bws"][fa].get("tap_m", 0.8), c["bws"][fb].get("tap_m", 0.8))
    if d <= tap + EPS:
        return f"set {r['id']} feed_a={fa}, feed_b={fb}"
    b0, b1 = c["bwf"][fa].bounds, c["bwf"][fb].bounds
    if b0[2] - b0[0] >= b0[3] - b0[1]:  # runs along x
        lo, hi = max(b0[0], b1[0]), min(b0[2], b1[2])
        ty = (c["bws"][fa]["y"] + c["bws"][fb]["y"]) / 2
        where = f"y={ty:.2f} (x within {lo:.1f}..{hi:.1f})"
    else:
        lo, hi = max(b0[1], b1[1]), min(b0[3], b1[3])
        tx = (c["bws"][fa]["x"] + c["bws"][fb]["x"]) / 2
        where = f"x={tx:.2f} (y within {lo:.1f}..{hi:.1f})"
    return f"move {r['id']} to {where} under {fa}/{fb} and set feed_a={fa}, feed_b={fb}"


def r_dual_feed(c):
    out = []
    for r in c["racks"]:
        fa, fb = r["feed_a"], r["feed_b"]
        if fa == fb:
            out.append(_v(f"{r['id']} feed_a and feed_b are both {fa!r}; 2N needs two different busways", 1.0,
                          _pair_fix(c, r)))
        for side, bid in (("feed_a", fa), ("feed_b", fb)) if fa != fb else (("feed_a", fa),):
            if bid is None and fa == fb:
                continue
            if bid not in c["bws"]:
                out.append(_v(f"{r['id']} {side}={bid!r} is not a busway in this hall", 1.0, _pair_fix(c, r)))
                continue
            tap = c["bws"][bid].get("tap_m", 0.8)
            d = Point(r["x"], r["y"]).distance(c["bwf"][bid])
            if d > tap + EPS:
                out.append(_v(f"{r['id']} is {d:.2f} m from {side} busway {bid}; tap whips reach {tap:.2f} m",
                              d - tap, _pair_fix(c, r)))
    return out


def r_power_2n(c):
    loads = {}
    for bid, b in sorted(c["bws"].items()):
        users = [r for r in c["racks"] if bid in (r["feed_a"], r["feed_b"])]
        loads[bid] = (sum(r["kw"] for r in users), _usable(b), users)
    spare = ", ".join(f"{k} {cap - l:.0f} kW" for k, (l, cap, _) in loads.items() if cap - l > 0)
    out = []
    for bid, (l, cap, users) in loads.items():
        if l > cap + EPS:
            b = c["bws"][bid]
            out.append(_v(f"{bid} serves {l:.0f} kW from {len(users)} racks (as feed A or B); if its partner fails it "
                          f"carries all of it; limit {cap:.0f} kW ({b['capacity_kw']} kW x {b.get('derate', 1.0)})",
                          l - cap, f"move at least {_min_moves([r['kw'] for r in users], l - cap)} rack(s) off {bid} "
                                   f"to a busway pair with spare capacity ({spare or 'none'})"))
    return out


def r_air_capacity(c):
    air = [r["kw"] for r in c["racks"] if r["cooling"] == "air"]
    budget = c["hall"].get("air_cooling_kw", 0.0)
    if sum(air) <= budget + EPS:
        return []
    return [_v(f"air-cooled racks total {sum(air):.0f} kW; room air cooling is {budget:.0f} kW", sum(air) - budget,
               f"remove {_min_moves(air, sum(air) - budget)} air-cooled rack(s) or replace them with liquid-cooled "
               "GB200/GB300 racks on a CDU")]


def r_hot_cold_aisle(c):
    out, B, rs = [], c["B"], c["racks"]
    F = {r["id"]: _front(r["rot"]) for r in rs}
    for a in rs:
        ba, fa, k = B[a["id"]], F[a["id"]], _axis(a["rot"])
        x = 1 - k
        s = fa[x]
        best = None
        for b in rs:
            if b is a or _ov(ba, B[b["id"]], k) <= FACE_OV:
                continue
            bb = B[b["id"]]
            g = bb[x] - ba[x + 2] if s > 0 else ba[x] - bb[x + 2]
            if -ABUT <= g <= FACE_MAX and (best is None or g < best[0]):
                best = (g, b)
        if best is None:
            continue
        g, b = best
        fb = F[b["id"]]
        if fa[0] * fb[0] + fa[1] * fb[1] >= 0:
            what = "back" if fa == fb else "side"
            out.append(_v(f"{a['id']} (rot {a['rot']:.0f}) blows its intake air from the {what} of {b['id']} "
                          f"(rot {b['rot']:.0f}) {max(g, 0):.2f} m away: hot exhaust feeds a cold intake", 1.0,
                          f"set {b['id']} rot to {(a['rot'] + 180) % 360:.0f} (front-to-front cold aisle) or set "
                          f"{a['id']} rot to {(a['rot'] + 180) % 360:.0f} (back-to-back hot aisle)"))
    for i, a in enumerate(rs):
        for b in rs[i + 1:]:
            k = _axis(a["rot"])
            if k != _axis(b["rot"]) or F[a["id"]] == F[b["id"]]:
                continue
            ba, bb = B[a["id"]], B[b["id"]]
            if _gap(ba, bb) <= ABUT and _ov(ba, bb, 1 - k) > FACE_OV and _ov(ba, bb, k) <= FACE_OV:
                out.append(_v(f"{a['id']} (rot {a['rot']:.0f}) and {b['id']} (rot {b['rot']:.0f}) abut in one row "
                              "but face opposite ways", 1.0, f"set {b['id']} rot to {a['rot']:.0f}"))
    return out


RULE_FNS = [("no_collision", r_no_collision), ("in_bounds", r_in_bounds), ("zones_clear", r_zones_clear),
            ("aisle_width", r_aisle_width), ("cooling_type", r_cooling_type), ("cdu_capacity", r_cdu_capacity),
            ("hose", r_hose), ("dual_feed", r_dual_feed), ("power_2n", r_power_2n),
            ("air_capacity", r_air_capacity), ("hot_cold_aisle", r_hot_cold_aisle)]


# --------------------------------------------------------------------------- scored terms

def _rehome_exact(kws, opts, free):
    free, dead = list(free), set()
    rem = [0.0] * (len(kws) + 1)
    for i in range(len(kws) - 1, -1, -1):
        rem[i] = rem[i + 1] + kws[i]

    def go(i):
        if i == len(kws):
            return True
        key = (i, tuple(free))
        if key in dead or rem[i] > sum(f for f in free if f > 0) + EPS:
            dead.add(key)
            return False
        for j in opts[i]:
            if free[j] + EPS >= kws[i]:
                free[j] -= kws[i]
                if go(i + 1):
                    return True
                free[j] += kws[i]
        dead.add(key)
        return False
    return go(0)


def _rehome_ffd(kws, opts, free):
    free = list(free)
    for kw, js in zip(kws, opts):
        j = next((j for j in js if free[j] + EPS >= kw), None)
        if j is None:
            return False
        free[j] -= kw
    return True


def _resilience(c):
    loads, ids = _cdu_loads(c), sorted(c["cdus"])
    ok, viol, notes = 0, [], []
    for cid in ids:
        rs = sorted((r for r in c["racks"] if r["cooling"] == "liquid" and r["cdu"] == cid),
                    key=lambda r: (-r["kw"], r["id"]))
        if not rs:
            ok += 1
            continue
        surv = [k for k in ids if k != cid]
        free = [c["cdus"][k]["capacity_kw"] - loads[k] for k in surv]
        opts = [[j for j, k in enumerate(surv) if _hose(c, r, k) <= c["cdus"][k]["hose_m"] + EPS] for r in rs]
        kws = [r["kw"] for r in rs]
        if len(rs) > EXACT_MAX:
            notes.append(f"{cid}: {len(rs)} racks > {EXACT_MAX}, first-fit-decreasing used")
            good = _rehome_ffd(kws, opts, free)
        else:
            good = _rehome_exact(kws, opts, free)
        if good:
            ok += 1
            continue
        stranded = [r["id"] for r, o in zip(rs, opts) if not o]
        sp = ", ".join(f"{k} {f:.0f} kW" for k, f in zip(surv, free)) or "none"
        fix = (f"move {', '.join(stranded[:6])} within hose reach of a second CDU" if stranded else
               f"free {sum(kws):.0f} kW of reachable spare on surviving CDUs, e.g. move a rack off {cid}")
        viol.append(_v(f"loss of {cid} strands {sum(kws):.0f} kW on {len(rs)} rack(s); survivor spare: {sp}"
                       + (f"; no survivor in hose reach for {', '.join(stranded[:6])}" if stranded else ""),
                       sum(kws), fix))
    return (ok / len(ids) if ids else 1.0), ok, len(ids), viol, notes


def score(hall, payload):
    data, errs = _rebuild(hall, payload)
    scene = Scene(data)
    racks = [o for o in data["objects"] if o["kind"] == "rack"]
    it_kw = float(sum(r["kw"] for r in racks))
    ub = upper_bound_kw(hall)
    zero = {"density": 0.0, "resilience": 0.0, "hose": 0.0}
    if errs:
        checks = [{"name": "payload", "passed": False, "hard": True, "violations": [_v(d, 1.0, f) for d, f in errs]}]
        return {"score": 0.0, "feasible": False, "it_kw": it_kw, "upper_bound_kw": ub, "components": zero,
                "checks": checks, "feedback": _feedback(checks)}
    cdus = {o["id"]: o for o in data["objects"] if o["kind"] == "cdu"}
    c = {"hall": hall, "racks": racks, "hall_objs": [o for o in data["objects"] if o["kind"] != "rack"],
         "B": {r["id"]: footprint(r).bounds for r in racks}, "cdus": cdus,
         "ports": {k: scene.point(f"{k}.port") if "port" in o.get("points", {}) else (o["x"], o["y"])
                   for k, o in cdus.items()},
         "bws": {o["id"]: o for o in data["objects"] if o["kind"] == "busway"}}
    c["bwf"] = {k: footprint(o) for k, o in c["bws"].items()}
    checks = [{"name": "payload", "passed": True, "hard": True, "violations": []}]
    for name, fn in RULE_FNS:
        v = fn(c)
        checks.append({"name": name, "passed": not v, "hard": True, "violations": v})
    feasible = all(ch["passed"] for ch in checks)
    if not feasible:
        return {"score": 0.0, "feasible": False, "it_kw": it_kw, "upper_bound_kw": ub, "components": zero,
                "checks": checks, "feedback": _feedback(checks)}
    density = min(1.0, it_kw / ub) if ub > 0 else 0.0
    res, ok, n, rviol, notes = _resilience(c)
    liquid = [r for r in racks if r["cooling"] == "liquid"]
    lim = sum(cdus[r["cdu"]]["hose_m"] for r in liquid)
    hose = 1.0 - sum(_hose(c, r, r["cdu"]) for r in liquid) / lim if liquid and lim > 0 else 1.0
    checks.append({"name": "resilience", "passed": not rviol, "hard": False, "violations": rviol,
                   "detail": f"{ok}/{n} CDU failures survivable (exact search"
                             + (f"; {'; '.join(notes)}" if notes else "") + ")"})
    comps = {"density": round(density, 4), "resilience": round(res, 4), "hose": round(hose, 4)}
    total = round(70 * density + 20 * res + 10 * hose, 3)
    fb = (f"feasible: {it_kw:.0f} kW / {ub:.0f} kW bound, resilience {ok}/{n} CDUs, hose {hose:.2f}, score {total}"
          + (" (FFD fallback used)" if notes else ""))
    return {"score": total, "feasible": True, "it_kw": it_kw, "upper_bound_kw": ub, "components": comps,
            "checks": checks, "feedback": fb}


def _feedback(checks, limit=25):
    rows = [(RULES.index(ch["name"]) if ch["name"] in RULES else 99, -v["magnitude"], ch["name"], v)
            for ch in checks if not ch["passed"] for v in ch["violations"]]
    rows.sort(key=lambda t: (t[0], t[1]))
    lines = [f"[{name}] {v['detail']} -> fix: {v['fix']}" for _, _, name, v in rows]
    if len(lines) > limit:
        lines = lines[:limit - 1] + [f"... and {len(lines) - limit + 1} more violations"]
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m donetwin.score <hall.json> <payload.json>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        hall = json.load(f)
    with open(argv[1], encoding="utf-8") as f:
        payload = json.load(f)
    print(json.dumps(score(hall, payload), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
