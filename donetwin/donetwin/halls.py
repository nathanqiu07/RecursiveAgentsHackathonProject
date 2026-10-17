"""Seeded generator of AI data-center halls for the layout benchmark.

    generate(seed, tier) -> hall dict (same shape as scenarios/data_hall.json, plus
    air_cooling_kw / aisle_min_m and 2N A/B busway pairs per row).

Seeds: train seeds are 0..9999 (TRAIN_SEEDS); held-out seeds are 10000..10049
(heldout_seeds()) and must never be used for tuning. Output is deterministic per
(seed, tier): the RNG is random.Random(f"{tier}:{seed}") and, if a draw fails
validation, random.Random(f"{tier}:{seed}:retry{k}").

CLI:
    python -m donetwin.halls --seed N --tier T [--out file.json]
    python -m donetwin.halls --write-heldout DIR --tier T
"""
import argparse
import json
import os
import random

from shapely.geometry import box

from .checks import footprint

TRAIN_SEEDS = range(0, 10000)
TIERS = {
    #         floor w, d   rows cdus
    "small": (14.0, 10.0, 2, 2),
    "medium": (20.0, 13.0, 3, 3),
    "large": (24.0, 16.0, 4, 4),
}
ROW_IDS = "ABCD"
WALK = 1.4          # egress corridor width (south + west walls)
RACK_D = 1.2        # rack depth (y)
AISLE = 1.2
MIN_FREE_M = 4.0    # every row must keep >= this much contiguous rack length
OVERHEAD = {"busway", "sensor"}
GOAL = "Deploy as much IT load as possible: 2N power (two feeds per rack), N+1 cooling, hot/cold aisles."
DOMAIN_NOTES = (
    "AI data hall fit-out with 2N power. New racks (0.6 m wide x 1.2 m deep, rot 0) go in rows centered "
    "under an A/B busway pair: BW_<row>1 is feed A at row_y-0.25, BW_<row>2 is feed B at row_y+0.25. "
    "2N: every rack takes two feeds, one from its row's A busway and one from its B busway, each within tap_m; "
    "either feed alone must carry the rack's full load, so each busway's assigned load <= capacity_kw*derate "
    "(continuous). Liquid-cooled racks (132-142 kW) need a CDU: assigned load <= capacity_kw, total liquid load "
    "must survive loss of the largest CDU (N+1), and the Manhattan hose run from rack center to CDU port <= hose_m. "
    "Air-cooled racks (~15 kW) need no CDU but their total load must fit in the room's air_cooling_kw. "
    "Hot/cold aisles: adjacent rows alternate orientation (fronts face fronts across a cold aisle, backs face "
    "backs across a hot aisle); racks in a row abut (gap 0) and aisles between rows are >= aisle_min_m. "
    "Pillars split rows; keepouts (slab limits, NEC 110.26 working space) and egress corridors must stay clear."
)
RESEARCH_QUERIES = [
    "NEC 110.26 working space clearance in front of electrical equipment",
    "NFPA 101 minimum egress corridor width",
    "data center hot aisle cold aisle minimum width",
    "NVIDIA GB200 NVL72 rack power kW weight liquid cooling CDU",
    "busway continuous load 80 percent NEC",
    "2N power distribution A B feed data center rack",
]


def heldout_seeds():
    return list(range(10000, 10050))


def _r(v):
    return round(v + 0.0, 2)


def _obj(id_, kind, x, y, w, d, h, **extra):
    o = {"id": id_, "kind": kind, "x": _r(x), "y": _r(y), "w": _r(w), "d": _r(d), "h": h, "rot": 0, "fixed": True}
    o.update(extra)
    return o


def _overlaps(a, b):
    return footprint(a).intersection(footprint(b)).area > 1e-6


def _clashes(o, objs):
    return any(_overlaps(o, p) for p in objs)


def _draw(rng, tier):
    bw, bd, n_rows, n_cdu = TIERS[tier]
    W = _r(bw * rng.uniform(0.9, 1.1))
    D = _r(bd * rng.uniform(0.9, 1.1))
    objs = [
        _obj("egress_W", "walkway", WALK / 2, D / 2, WALK, D, 0.0, note="egress corridor to west exit door"),
        _obj("egress_S", "walkway", W / 2, WALK / 2, W, WALK, 0.0, note="egress corridor to south exit doors"),
    ]

    # rows: first row clear of the south corridor, last row leaves an aisle in front of the CDUs
    start = rng.uniform(WALK + RACK_D / 2 + 0.6, WALK + RACK_D / 2 + 1.4)
    top = D - 0.8 - 0.4 - AISLE - RACK_D / 2       # CDU front face minus aisle minus half rack
    pitch = rng.uniform(2.6, 3.6)
    if n_rows > 1 and start + (n_rows - 1) * pitch > top:
        pitch = (top - start) / (n_rows - 1)
    if pitch < 2.6 or start > top:
        return None
    rows_y = [_r(start + i * pitch) for i in range(n_rows)]

    # CDUs spread along the north wall, port facing into the room
    cy = D - 0.8
    lo, hi = WALK + 0.8, W - 0.8
    seg = (hi - lo) / n_cdu
    for i in range(n_cdu):
        cap = rng.randrange(600, 901, 50)
        hose = round(rng.uniform(10.0, 16.0) * 2) / 2
        x = lo + seg * (i + 0.5) + rng.uniform(-0.25, 0.25) * max(0.0, seg - 1.4)
        objs.append(_obj(f"CDU_{i + 1}", "cdu", x, cy, 1.2, 0.8, 2.0, capacity_kw=cap, hose_m=hose,
                         points={"port": [0.0, -0.4]},
                         note="liquid-cooling CDU; rack hoses run orthogonally (Manhattan) to the port, max hose_m"))

    # main switchboard on the east wall + NEC 110.26 working space in front of it
    py = rng.uniform(WALK + 1.0, D - 1.2 - 1.2)
    objs.append(_obj("PNL_1", "panel", W - 0.3, py, 0.6, 1.6, 2.2, note="main switchboard"))
    objs.append(_obj("nec_PNL_1", "keepout", W - 0.6 - 0.65, py, 1.3, 2.2, 0.0,
                     reason="NEC 110.26 working space in front of PNL_1 (keep clear)"))

    # pillar grid; some grid lines snap onto a row line (splits that row)
    sx, sy = rng.uniform(6.0, 9.0), rng.uniform(6.0, 9.0)
    ox, oy = rng.uniform(0.35, 0.85) * sx, rng.uniform(0.35, 0.85) * sy
    xs = [ox + i * sx for i in range(20) if ox + i * sx < W - 1.0]
    ys = []
    for j in range(20):
        y = oy + j * sy
        if y >= D - 1.0:
            break
        near = min(rows_y, key=lambda r: abs(r - y))
        if abs(near - y) < 1.8 and rng.random() < 0.5:
            y = near
        ys.append(y)
    k = 0
    for y in ys:
        for x in xs:
            p = _obj(f"col_C{k + 1}", "pillar", x, y, 0.6, 0.6, 4.5)
            if 0.3 < p["x"] < W - 0.3 and 0.3 < p["y"] < D - 0.3 and not _clashes(p, objs):
                objs.append(p)
                k += 1

    # optional slab keepout (cable vault)
    if rng.random() < 0.6:
        for _ in range(20):
            w, d = rng.uniform(2.5, 6.0), rng.uniform(2.0, 4.5)
            x = rng.uniform(WALK + w / 2 + 0.2, W - 1.3 - w / 2)
            y = rng.uniform(WALK + d / 2 + 0.2, D - 1.2 - d / 2)
            s = _obj("slab_vault", "keepout", x, y, w, d, 0.0,
                     reason="Slab over cable vault: 1,200 kg/m2 max (GB200 NVL72 rack ~1,890 kg/m2)")
            if not _clashes(s, objs):
                objs.append(s)
                break

    # A/B busway pairs per row (overhead)
    for rid, ry in zip(ROW_IDS, rows_y):
        x0 = rng.uniform(WALK + 0.2, WALK + 1.6)
        x1 = W - rng.uniform(0.8, 2.8)
        if rng.random() < 0.25:                   # occasional short busway run
            x1 = x0 + (x1 - x0) * rng.uniform(0.6, 0.85)
        x0, x1 = _r(x0), _r(x1)
        cap = rng.randrange(600, 901, 50)
        for n, feed, dy in ((1, "A", -0.25), (2, "B", 0.25)):
            objs.append(_obj(f"BW_{rid}{n}", "busway", (x0 + x1) / 2, ry + dy, x1 - x0, 0.25, 0.15,
                             z=3.0, capacity_kw=cap, derate=0.8, tap_m=0.8, feed=feed, row=rid,
                             note=f"overhead power busway row {rid} feed {feed}, x {x0:.2f}..{x1:.2f}; "
                                  "continuous load <= capacity*derate; racks must sit within tap_m of it"))
    # keep busways next to the other infrastructure in a stable order
    objs.sort(key=lambda o: ["walkway", "pillar", "cdu", "busway", "panel", "keepout"].index(o["kind"]))

    hall = {
        "name": f"{tier}-{{seed}}", "title": "", "goal": GOAL,
        "floor": {"w": W, "d": D}, "objects": objs, "domain_notes": DOMAIN_NOTES,
        "research_queries": list(RESEARCH_QUERIES),
        "air_cooling_kw": rng.randrange(40, 241, 10), "aisle_min_m": AISLE,
    }
    return hall


def row_free_segments(hall, row):
    """Free x-intervals (x0, x1) of the rack strip under a row's busway pair."""
    bws = [o for o in hall["objects"] if o["kind"] == "busway" and o.get("row") == row]
    a = next(o for o in bws if o["feed"] == "A")
    ry = a["y"] + 0.25
    x0, x1 = max(0.0, a["x"] - a["w"] / 2), min(hall["floor"]["w"], a["x"] + a["w"] / 2)
    strip = box(x0, ry - RACK_D / 2, x1, ry + RACK_D / 2)
    blocks = []
    for o in hall["objects"]:
        if o["kind"] in OVERHEAD:
            continue
        inter = footprint(o).intersection(strip)
        if inter.area > 1e-6:
            bx0, _, bx1, _ = inter.bounds
            blocks.append((bx0, bx1))
    segs, cur = [], x0
    for bx0, bx1 in sorted(blocks):
        if bx0 > cur:
            segs.append((cur, bx0))
        cur = max(cur, bx1)
    if x1 > cur:
        segs.append((cur, x1))
    return segs


def validate(hall):
    """Return a list of problems (empty == valid)."""
    errs = []
    W, D = hall["floor"]["w"], hall["floor"]["d"]
    floor = box(0, 0, W, D)
    objs = hall["objects"]
    ids = [o["id"] for o in objs]
    if len(set(ids)) != len(ids):
        errs.append("duplicate ids")
    for o in objs:
        if not floor.buffer(1e-6).contains(footprint(o)):
            errs.append(f"{o['id']} out of bounds")
        if not o.get("fixed"):
            errs.append(f"{o['id']} not fixed")
    ground = [o for o in objs if o["kind"] not in OVERHEAD]
    for i, a in enumerate(ground):
        for b in ground[i + 1:]:
            if a["kind"] == b["kind"] == "walkway":
                continue
            if _overlaps(a, b):
                errs.append(f"{a['id']} overlaps {b['id']}")
    rows = sorted({o["row"] for o in objs if o["kind"] == "busway"})
    walks = [o for o in objs if o["kind"] == "walkway"]
    for r in rows:
        pair = {o["feed"]: o for o in objs if o["kind"] == "busway" and o.get("row") == r}
        if set(pair) != {"A", "B"} or pair["A"]["id"] != f"BW_{r}1" or pair["B"]["id"] != f"BW_{r}2":
            errs.append(f"row {r}: bad busway pair")
            continue
        a, b = pair["A"], pair["B"]
        if abs((b["y"] - a["y"]) - 0.5) > 1e-6 or a["x"] != b["x"] or a["w"] != b["w"]:
            errs.append(f"row {r}: A/B busways misaligned")
        ry = a["y"] + 0.25
        strip = box(a["x"] - a["w"] / 2, ry - RACK_D / 2, a["x"] + a["w"] / 2, ry + RACK_D / 2)
        if any(footprint(w).contains(strip) for w in walks):
            errs.append(f"row {r}: rack strip inside a walkway")
        best = max((s1 - s0 for s0, s1 in row_free_segments(hall, r)), default=0.0)
        if best < MIN_FREE_M - 1e-6:
            errs.append(f"row {r}: longest free run {best:.2f} m < {MIN_FREE_M} m")
    for o in objs:
        if o["kind"] == "cdu" and "port" not in o.get("points", {}):
            errs.append(f"{o['id']} has no port")
    return errs


def generate(seed: int, tier: str = "medium"):
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {sorted(TIERS)}")
    for k in range(200):
        rng = random.Random(f"{tier}:{seed}" if k == 0 else f"{tier}:{seed}:retry{k}")
        hall = _draw(rng, tier)
        if hall is None:
            continue
        hall["name"] = f"{tier}-{seed}"
        n_rows = TIERS[tier][2]
        hall["title"] = f"{tier.capitalize()} hall #{seed} · {n_rows}-row 2N AI fit-out"
        if not validate(hall):
            return hall
    raise RuntimeError(f"could not generate a valid hall for {tier}:{seed}")


def summary(hall):
    ob = hall["objects"]
    cnt = lambda k: sum(o["kind"] == k for o in ob)
    return (f"{hall['name']}: floor {hall['floor']['w']}x{hall['floor']['d']} m, {cnt('busway') // 2} rows, "
            f"{cnt('cdu')} CDUs, {cnt('pillar')} pillars, {cnt('keepout')} keepouts, "
            f"air {hall['air_cooling_kw']} kW")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--tier", default="medium", choices=sorted(TIERS))
    ap.add_argument("--out")
    ap.add_argument("--write-heldout", metavar="DIR")
    a = ap.parse_args(argv)
    if a.write_heldout:
        os.makedirs(a.write_heldout, exist_ok=True)
        for s in heldout_seeds():
            path = os.path.join(a.write_heldout, f"{a.tier}-{s}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(generate(s, a.tier), f, indent=2)
        print(f"wrote {len(heldout_seeds())} {a.tier} halls to {a.write_heldout}")
        return
    if a.seed is None:
        ap.error("--seed is required (or use --write-heldout DIR)")
    hall = generate(a.seed, a.tier)
    text = json.dumps(hall, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(summary(hall))
    else:
        print(text)


if __name__ == "__main__":
    main()
