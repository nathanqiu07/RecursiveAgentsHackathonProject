"""Deterministic greedy baseline for the data-hall layout benchmark (no LLM).

solve(hall) -> {"racks": [...]}: fill each busway row left-to-right with abutting racks,
GB200-NVL72 first (CDU with most spare capacity in hose reach, within 2N busway budget),
then HGX-B200 air racks while air cooling and 2N allow. Rows alternate rot 0/180 so that
neighbours face front-to-front (cold aisle) / back-to-back (hot aisle). Finally the layout
is scored and any racks named in violations are dropped until it is feasible.
"""
import re

from shapely.geometry import Point, box

from donetwin.checks import footprint
from donetwin.scene import RACK_CATALOG, Scene

LIQUID, AIR = "GB200-NVL72", "HGX-B200"
W, D = RACK_CATALOG[LIQUID]["w"], RACK_CATALOG[LIQUID]["d"]
STEP, NUDGE, EPS = W, 0.1, 1e-6
OBSTACLE_KINDS = {"pillar", "panel", "cdu", "walkway", "keepout", "rack", "conveyor", "bin", "fence", "pallet", "robot"}


def _rows(hall):
    """[(row_id, row_y, busway_A, busway_B)] sorted by y."""
    rows = {}
    for o in hall["objects"]:
        if o.get("kind") == "busway" and o.get("row") is not None:
            rows.setdefault(o["row"], {})[o.get("feed", "A")] = o
    out = []
    for rid, f in rows.items():
        if "A" in f and "B" in f:
            out.append((rid, (f["A"]["y"] + f["B"]["y"]) / 2, f["A"], f["B"]))
    return sorted(out, key=lambda r: r[1])


def _free(fp, obstacles, floor):
    return floor.contains(fp) and all(fp.distance(o) > EPS for o in obstacles)


def _in_tap(x, y, bws):
    return all(Point(x, y).distance(footprint(b)) <= b.get("tap_m", 0.8) - 0.01 for b in bws)


def _slots(hall, rows):
    """Candidate rack centres per row: abutting runs, split by obstacles."""
    floor = box(0, 0, hall["floor"]["w"], hall["floor"]["d"])
    obstacles = [footprint(o) for o in hall["objects"] if o.get("kind") in OBSTACLE_KINDS]
    slots = {}
    for rid, y, a, b in rows:
        xs, x = [], W / 2
        while x <= hall["floor"]["w"] - W / 2 + EPS:
            fp = box(x - W / 2, y - D / 2, x + W / 2, y + D / 2)
            if _in_tap(x, y, (a, b)) and _free(fp, obstacles, floor):
                xs.append(round(x, 4))
                x = round(x + STEP, 4)
            else:
                x = round(x + NUDGE, 4)
        slots[rid] = xs
    return slots


def _build(hall, banned=frozenset()):
    rows = _rows(hall)
    slots = _slots(hall, rows)
    scene = Scene(hall)
    cdus = [o for o in hall["objects"] if o.get("kind") == "cdu"]
    ports = {c["id"]: (scene.point(f"{c['id']}.port") if "port" in c.get("points", {}) else (c["x"], c["y"]))
             for c in cdus}
    cdu_load = {c["id"]: 0.0 for c in cdus}
    air_cap = float(hall.get("air_cooling_kw", 0))
    air_load = 0.0
    racks = {}  # (row, x) -> rack dict
    row_load = {}
    budget = {rid: min(a["capacity_kw"] * a.get("derate", 1.0), b["capacity_kw"] * b.get("derate", 1.0))
              for rid, _, a, b in rows}
    kw_l, kw_a = RACK_CATALOG[LIQUID]["kw"], RACK_CATALOG[AIR]["kw"]

    # pass 1: liquid racks
    for rid, y, a, b in rows:
        row_load[rid] = 0.0
        for x in slots[rid]:
            if (rid, x) in banned:
                continue
            if row_load[rid] + kw_l > budget[rid] + EPS:
                break
            fits = [(c["capacity_kw"] - cdu_load[c["id"]], c["id"]) for c in cdus
                    if cdu_load[c["id"]] + kw_l <= c["capacity_kw"] + EPS
                    and abs(x - ports[c["id"]][0]) + abs(y - ports[c["id"]][1]) <= c["hose_m"] - 0.01]
            if not fits:
                continue
            cid = max(fits, key=lambda t: (t[0], [-ord(ch) for ch in t[1]]))[1]
            cdu_load[cid] += kw_l
            row_load[rid] += kw_l
            racks[(rid, x)] = {"model": LIQUID, "x": x, "y": round(y, 4), "cdu": cid,
                               "feed_a": a["id"], "feed_b": b["id"]}
    # pass 2: air racks in remaining slots
    for rid, y, a, b in rows:
        for x in slots[rid]:
            if (rid, x) in racks or (rid, x) in banned:
                continue
            if row_load[rid] + kw_a > budget[rid] + EPS or air_load + kw_a > air_cap + EPS:
                continue
            row_load[rid] += kw_a
            air_load += kw_a
            racks[(rid, x)] = {"model": AIR, "x": x, "y": round(y, 4), "cdu": None,
                               "feed_a": a["id"], "feed_b": b["id"]}
    # orientation: alternate over non-empty rows (row k even -> rot 0 faces +y, odd -> 180)
    used = [rid for rid, *_ in rows if any(k[0] == rid for k in racks)]
    rot = {rid: (0 if i % 2 == 0 else 180) for i, rid in enumerate(used)}
    out, keys = [], []
    for rid, *_ in rows:
        for n, k in enumerate(sorted(k for k in racks if k[0] == rid), 1):
            r = racks[k]
            out.append({"id": f"R_{rid}_{n:02d}"[:32], "model": r["model"], "x": r["x"], "y": r["y"],
                        "rot": rot[rid], "cdu": r["cdu"], "feed_a": r["feed_a"], "feed_b": r["feed_b"]})
            keys.append(k)
    return out[:300], keys[:300]


def _named(checks, ids, found):
    """Collect rack ids mentioned in the violations of failed hard checks."""
    text = " ".join(f"{v.get('detail', '')} {v.get('objects', '')}" for ch in (checks or [])
                    if ch.get("hard", True) and not ch.get("passed") for v in ch.get("violations", []))
    found.update(i for i in ids if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(i)}(?![A-Za-z0-9_-])", text))
    return found


def solve(hall, passes=6):
    banned = set()
    racks, keys = _build(hall, banned)
    try:
        from donetwin import score as scoring
    except ImportError:
        return {"racks": racks}
    for _ in range(passes):
        res = scoring.score(hall, {"racks": racks})
        if res.get("feasible"):
            break
        ids = {r["id"]: k for r, k in zip(racks, keys)}
        bad = _named(res.get("checks"), ids, set())
        if not bad:  # nothing attributable: fall back to dropping the last rack
            bad = {racks[-1]["id"]} if racks else set()
        if not bad:
            break
        banned |= {ids[i] for i in bad}
        racks, keys = _build(hall, banned)
    return {"racks": racks}
