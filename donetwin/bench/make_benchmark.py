"""Write bench/hall_b_benchmark.json: benchmark definition for the team dashboard (POST /api/benchmarks).

  python bench/make_benchmark.py [--hall scenarios/data_hall_2n.json] [--out bench/hall_b_benchmark.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from donetwin.scene import RACK_CATALOG, Scene  # noqa: E402
from donetwin.score import upper_bound_kw  # noqa: E402

KEEP = ["id", "kind", "x", "y", "w", "d", "rot", "capacity_kw", "derate", "tap_m", "hose_m", "feed", "row", "reason"]


def hall_summary(hall):
    sc = Scene(hall)
    objs = []
    for o in hall["objects"]:
        s = {k: o[k] for k in KEEP if k in o}
        if o.get("kind") == "cdu" and "port" in o.get("points", {}):
            s["port_xy"] = [round(v, 3) for v in sc.point(f"{o['id']}.port")]
        if o.get("kind") == "busway":
            s["x_range"] = [round(o["x"] - o["w"] / 2, 3), round(o["x"] + o["w"] / 2, 3)]
        objs.append(s)
    return {"name": hall.get("name"), "floor": hall["floor"], "air_cooling_kw": hall.get("air_cooling_kw"),
            "aisle_min_m": hall.get("aisle_min_m", 1.2), "objects": objs}


def build(hall):
    summ = hall_summary(hall)
    cdus = [o["id"] for o in hall["objects"] if o.get("kind") == "cdu"]
    bws = [o["id"] for o in hall["objects"] if o.get("kind") == "busway"]
    models = list(RACK_CATALOG)
    catalog = {m: {"kw": v["kw"], "cooling": "air" if m == "HGX-B200" else "liquid", "w": v["w"], "d": v["d"]}
               for m, v in RACK_CATALOG.items()}
    prompt = f"""Fit out data hall "{hall.get('name')}" with AI racks: maximize deployed IT load (kW) under 2N power and N+1 cooling.

HALL (units: metres, kW; x/y are object centres; w is the x size, d the y size at rot 0; floor spans x 0..{hall['floor']['w']}, y 0..{hall['floor']['d']}):
{json.dumps(summ, separators=(',', ':'))}

RACK CATALOG (footprint 0.6 m in x by 1.2 m in y at rot 0; at rot 90/270 the footprint swaps to 1.2 x 0.6):
{json.dumps(catalog, separators=(',', ':'))}

HARD RULES (any violation makes the score 0):
1. No collision: rack footprints must not overlap each other or any pillar, panel or CDU.
2. In bounds: every rack footprint lies inside the floor.
3. Zones clear: rack footprints must not overlap walkways (egress) or keepouts.
4. Aisles: between rows of racks that face each other across an aisle, the clear gap is >= aisle_min_m ({summ['aisle_min_m']} m). Racks within a row abut (gap 0).
5. Cooling type: liquid racks (GB200-NVL72, GB300-NVL72) must name a CDU in "cdu"; air racks (HGX-B200) must have "cdu": null.
6. CDU capacity: the sum of kW of racks assigned to a CDU <= its capacity_kw.
7. Hose: Manhattan distance from rack centre to the CDU's port (port_xy) <= that CDU's hose_m.
8. Dual feed: feed_a and feed_b are two different busway ids, and the rack centre is within tap_m of each busway's footprint. Each row has an A/B pair: BW_<row>1 (feed A) at row_y-0.25 and BW_<row>2 (feed B) at row_y+0.25.
9. 2N power: for every busway, the sum of kW of all racks that use it as either feed <= capacity_kw * derate (either feed alone must carry the full rack load).
10. Air capacity: the total kW of air racks <= air_cooling_kw ({hall.get('air_cooling_kw')} kW).
11. Hot/cold aisle: racks that abut in a row face the same way; rows that face each other across an aisle must have opposite fronts (front-to-front cold aisle or back-to-back hot aisle).

ORIENTATION: rot is in degrees, one of 0, 90, 180, 270. The rack's front (cold-air intake) direction is (-sin(rot), cos(rot)): rot 0 faces +y, rot 180 faces -y, rot 90 faces -x, rot 270 faces +x. A common pattern: alternate rows rot 0 / rot 180 so neighbouring rows face front-to-front, then back-to-back.

SUBMISSION FORMAT (JSON, no other keys):
{{"racks": [{{"id": "R1", "model": "GB200-NVL72", "x": 3.0, "y": 4.2, "rot": 0, "cdu": "{cdus[0] if cdus else 'CDU_1'}", "feed_a": "{bws[0] if bws else 'BW_A1'}", "feed_b": "{bws[1] if len(bws) > 1 else 'BW_A2'}"}}, ...]}}
Rack kW comes from the catalog by model; any extra key (for example "kw") is rejected.

SCORING (0..100, higher is better; 0 if any hard rule fails):
score = 70 * density + 20 * resilience + 10 * hose
- density = min(1, deployed IT kW / upper_bound_kw), where upper_bound_kw = min(sum over busways of capacity_kw*derate / 2, sum of CDU capacity_kw + air_cooling_kw) = {upper_bound_kw(hall):.0f} kW for this hall. N+1 cooling is not a hard rule; it is rewarded through resilience.
- resilience = fraction of CDUs whose single failure is survivable: every liquid rack on the failed CDU can be re-homed onto the surviving CDUs' spare capacity (capacity_kw minus assigned load) within those CDUs' hose reach. A CDU with no racks counts as survivable.
- hose = 1 - (sum of Manhattan hose runs of liquid racks) / (sum of hose_m of their assigned CDUs); 1 if there are no liquid racks.
The verifier returns the score plus feedback listing every violated rule with the offending rack ids.
"""
    instructions = ("Submit one JSON object {\"racks\": [...]} exactly as described in the prompt (no other top-level keys, "
                    "no extra rack keys). Each rack: id (unique, letters/digits/_/-), model from the catalog, x and y "
                    "(rack centre, metres), rot in {0,90,180,270}, cdu (a CDU id for liquid racks, null for air racks), "
                    "feed_a and feed_b (two different busway ids, normally the A/B pair of the rack's row). At most 300 racks. "
                    "Any hard-rule violation scores 0; read the feedback to see which racks broke which rule.")
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Hall layout submission",
        "type": "object",
        "additionalProperties": False,
        "required": ["racks"],
        "properties": {"racks": {"type": "array", "maxItems": 300, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "model", "x", "y", "rot", "cdu", "feed_a", "feed_b"],
            "properties": {
                "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,32}$"},
                "model": {"enum": models},
                "x": {"type": "number", "minimum": 0, "maximum": hall["floor"]["w"]},
                "y": {"type": "number", "minimum": 0, "maximum": hall["floor"]["d"]},
                "rot": {"enum": [0, 90, 180, 270]},
                "cdu": {"enum": cdus + [None]},
                "feed_a": {"enum": bws},
                "feed_b": {"enum": bws},
            }}}},
    }
    return {
        "title": "Hall B max IT load (2N power, N+1 cooling)",
        "prompt": prompt,
        "submission_instructions": instructions,
        "submission_schema": schema,
        "score_name": "Hall score",
        "score_direction": "higher",
        "score_min": 0,
        "score_max": 100,
        "verifier": {"url": "http://127.0.0.1:9000/evaluations", "method": "POST", "mode": "synchronous",
                     "score_path": "score", "feedback_path": "feedback"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", default="scenarios/data_hall_2n.json")
    ap.add_argument("--out", default="bench/hall_b_benchmark.json")
    a = ap.parse_args()
    hp, op = Path(a.hall), Path(a.out)
    hp, op = (hp if hp.is_absolute() else ROOT / hp), (op if op.is_absolute() else ROOT / op)
    hall = json.loads(hp.read_text(encoding="utf-8"))
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(build(hall), indent=2), encoding="utf-8")
    print(f"wrote {op.relative_to(ROOT)} (hall {hall.get('name')})")


if __name__ == "__main__":
    main()
