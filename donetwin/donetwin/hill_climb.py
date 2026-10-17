"""LLM hill-climbing solver for the data-hall layout benchmark.

solve(hall, llm=...) keeps a verified incumbent (greedy baseline by default), asks the LLM for small
edits to it per iteration (remove / update / add racks; a complete payload is also accepted), scores
the result with donetwin.score, and accepts only strict improvements. Feasible bests are written to a small on-disk library that seeds future prompts
(nearest examples by simple hall features).
"""
import json
import math
import os
import re

RULES = """HARD RULES (any violation => score 0):
- Racks must not collide with each other or with pillars/panels/CDUs; stay fully inside the floor.
- Racks must not overlap walkway/keepout zones.
- Aisles between facing rows >= aisle_min_m (at least 1.2 m).
- Rack footprint 0.6 x 1.2 m at rot 0 (x-width 0.6, y-depth 1.2); rot 90/270 swaps them.
- Front direction = (-sin(rot), cos(rot)): rot 0 faces +y, rot 180 faces -y. Racks in a row abut and
  face the same way; rows facing across an aisle must be front-to-front (cold) or back-to-back (hot).
- Liquid racks (GB200-NVL72 132 kW, GB300-NVL72 142 kW) need "cdu" set; air racks (HGX-B200 15 kW) need "cdu": null.
- CDU load (sum kW of its racks) <= capacity_kw. Hose = Manhattan distance rack center -> CDU port <= hose_m.
- Each rack has feed_a and feed_b = the two busways of ONE row pair (BW_<row>1 and BW_<row>2); the rack
  center must be within tap_m of both busway footprints (distance from center to busway rectangle).
- 2N power: for every busway, sum kW of racks that use it as feed_a OR feed_b <= capacity_kw*derate.
- Total air-cooled kW <= air_cooling_kw.
SCORE 0..100 = 70*density (IT kW / upper bound) + 20*resilience (survive any single CDU failure:
spread liquid load so remaining CDUs have headroom) + 10*hose shortness."""

FORMAT = ('Reply with ONE compact JSON object on a single line, no prose, listing EDITS to the incumbent: '
          '{"remove": ["R_A_03"], "update": [{"id": "R_B_01", "cdu": "CDU_4"}], '
          '"add": [{"id": "N1", "model": "GB300-NVL72", "x": 3.3, "y": 4.0, "rot": 0, "cdu": "CDU_1", '
          '"feed_a": "BW_A1", "feed_b": "BW_A2"}]}. Any of the three keys may be omitted. "update" changes '
          "model, x, y, rot, cdu, feed_a or feed_b of an existing id; new ids must be unique. Rack keys "
          "exactly: id, model, x, y, rot (0|90|180|270), cdu, feed_a, feed_b. Keep edits small and targeted.")

# Nemotron 3 otherwise spends its whole output budget on hidden reasoning before writing any JSON.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


# ---------------------------------------------------------------- hall summary / features
def _objs(hall, kind):
    return [o for o in hall.get("objects", []) if o.get("kind") == kind]


def _r(v):
    return round(float(v), 3)


def _port_xy(cdu):
    px, py = (cdu.get("points") or {}).get("port", [0.0, 0.0])
    a = math.radians(float(cdu.get("rot", 0) or 0))
    return (_r(cdu["x"] + px * math.cos(a) - py * math.sin(a)),
            _r(cdu["y"] + px * math.sin(a) + py * math.cos(a)))


def _rows(hall):
    rows = {}
    for b in _objs(hall, "busway"):
        key = b.get("row") or b["id"]
        rows.setdefault(str(key), []).append(b)
    return rows


def features(hall):
    f = hall.get("floor", {})
    return {"n_rows": len(_rows(hall)), "n_cdus": len(_objs(hall, "cdu")),
            "area": _r(float(f.get("w", 0)) * float(f.get("d", 0))),
            "air_cooling_kw": float(hall.get("air_cooling_kw", 0) or 0)}


def hall_summary(hall, upper_bound=None):
    f = hall.get("floor", {})
    lines = [f"HALL {hall.get('name', '?')}: floor {f.get('w')} x {f.get('d')} m (x right, y up, origin bottom-left).",
             f"air_cooling_kw={hall.get('air_cooling_kw', 0)}  aisle_min_m={hall.get('aisle_min_m', 1.2)}"
             + (f"  density upper bound={_r(upper_bound)} kW" if upper_bound else "")]
    lines.append("ROWS (busway pairs; rack feed_a/feed_b must be the pair of one row):")
    for key, bws in sorted(_rows(hall).items()):
        parts = []
        for b in sorted(bws, key=lambda o: o["y"]):
            x0, x1 = b["x"] - b.get("w", 0) / 2, b["x"] + b.get("w", 0) / 2
            usable = float(b.get("capacity_kw", 0)) * float(b.get("derate", 1))
            parts.append(f"{b['id']}(feed {b.get('feed', '?')}) y={_r(b['y'])} x {_r(x0)}..{_r(x1)} "
                         f"usable {_r(usable)} kW tap {b.get('tap_m', 0.8)} m")
        ys = [b["y"] for b in bws]
        lines.append(f"  row {key}: row_y={_r(sum(ys) / len(ys))}; " + "; ".join(parts))
    lines.append("CDUs:")
    for c in _objs(hall, "cdu"):
        lines.append(f"  {c['id']}: port {_port_xy(c)} capacity {c.get('capacity_kw')} kW hose_m {c.get('hose_m')}"
                     f" body at ({_r(c['x'])},{_r(c['y'])}) {c.get('w')}x{c.get('d')}")
    for kind, label in (("pillar", "PILLARS"), ("panel", "PANELS"), ("walkway", "WALKWAYS (keep clear)"),
                        ("keepout", "KEEPOUTS (keep clear)")):
        objs = _objs(hall, kind)
        if objs:
            lines.append(f"{label}: " + "; ".join(
                f"{o['id']} center ({_r(o['x'])},{_r(o['y'])}) {_r(o.get('w', 0))}x{_r(o.get('d', 0))}" for o in objs))
    return "\n".join(lines)


# ---------------------------------------------------------------- library
def _lib_path(library_dir, hall):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(hall.get("name") or "hall"))
    return os.path.join(library_dir, f"{name}.json")


def load_library(library_dir="library"):
    out = []
    if not library_dir or not os.path.isdir(library_dir):
        return out
    for fn in sorted(os.listdir(library_dir)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(library_dir, fn), encoding="utf-8") as fh:
                    e = json.load(fh)
                if isinstance(e, dict) and "payload" in e and "hall" in e:
                    out.append(e)
            except (OSError, json.JSONDecodeError):
                pass
    return out


def save_to_library(library_dir, hall, payload, score):
    """Write library/<hall name>.json only if it beats the existing entry. Returns True if saved."""
    if not library_dir:
        return False
    path = _lib_path(library_dir, hall)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                if float(json.load(fh).get("score", -1)) >= float(score):
                    return False
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    os.makedirs(library_dir, exist_ok=True)
    entry = {"hall": hall, "payload": payload, "score": float(score), "features": features(hall)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=1)
    return True


def nearest_examples(hall, library, k=2):
    f = features(hall)

    def dist(e):
        g = e.get("features") or features(e["hall"])
        return (abs(f["n_rows"] - g.get("n_rows", 0)) / max(1, f["n_rows"])
                + abs(f["n_cdus"] - g.get("n_cdus", 0)) / max(1, f["n_cdus"])
                + abs(f["area"] - g.get("area", 0)) / max(1.0, f["area"])
                + abs(f["air_cooling_kw"] - g.get("air_cooling_kw", 0)) / max(1.0, f["air_cooling_kw"]))
    return sorted(library, key=dist)[:k]


# ---------------------------------------------------------------- prompt
def _cj(obj):
    return json.dumps(obj, separators=(",", ":"))


def _feedback_lines(res, n=25):
    fb = (res or {}).get("feedback") or ""
    lines = [ln for ln in str(fb).splitlines() if ln.strip()]
    return lines[:n]


def build_prompt(hall, inc, inc_res, examples, last=None):
    parts = [hall_summary(hall, inc_res.get("upper_bound_kw")), "", RULES, "", FORMAT, ""]
    for i, e in enumerate(examples, 1):
        racks = e["payload"].get("racks", [])
        parts.append(f"VERIFIED EXAMPLE {i} (other hall {e['hall'].get('name')}, features {_cj(e.get('features'))}, "
                     f"score {_r(e.get('score', 0))}) - {len(racks)} racks, first 40 shown:")
        parts.append(_cj({"racks": racks[:40]}))
        parts.append(f"  its hall: {hall_summary(e['hall']).splitlines()[0]}")
    comps = {k: (_r(v) if isinstance(v, (int, float)) else v) for k, v in (inc_res.get("components") or {}).items()}
    parts += ["", f"INCUMBENT (score {_r(inc_res.get('score', 0))}, feasible {inc_res.get('feasible')}, "
                  f"it_kw {inc_res.get('it_kw')}, components {_cj(comps)}):", _cj(inc)]
    fb = _feedback_lines(inc_res)
    parts.append("VERIFIER FEEDBACK ON INCUMBENT:\n" + ("\n".join(fb) if fb else "(no violations)"))
    if last:
        parts.append(f"YOUR PREVIOUS ATTEMPT WAS REJECTED: {last}")
    parts.append("\nTASK: output edits that make the incumbent score strictly higher while keeping every "
                 "hard rule. Typical wins: add racks in free row length, swap HGX for GB300 where power/CDU "
                 "budget allows, reassign racks across CDUs so any single CDU failure is survivable, shorten "
                 "hoses. JSON only.")
    return "\n".join(parts)


def _parse(text):
    from donetwin.llm import extract_json
    try:
        obj = json.loads(text.strip())  # whole reply is JSON (object or bare list)
    except ValueError:
        obj = extract_json(text)  # fenced / embedded object, or ValueError
    if isinstance(obj, list):
        obj = {"racks": obj}
    if isinstance(obj, dict) and "racks" not in obj and isinstance(obj.get("payload"), dict):
        obj = obj["payload"]
    if isinstance(obj, dict) and "racks" in obj:
        obj = {"racks": obj["racks"]}
    return obj


def apply_edits(inc, obj):
    """A full {"racks": [...]} payload as-is, or {"remove", "update", "add"} edits applied to the incumbent.
    Field validation is left to donetwin.score, which rejects unknown keys and values."""
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    keys = ("racks", "remove", "update", "add")
    if not any(k in obj for k in keys):  # unwrap one level, e.g. {"edits": {...}}
        inner = [v for v in obj.values() if isinstance(v, dict) and any(k in v for k in keys)]
        if len(inner) == 1:
            obj = inner[0]
    if "racks" in obj:
        return {"racks": obj["racks"]}
    if not any(k in obj for k in ("remove", "update", "add")):
        raise ValueError("reply has none of racks/remove/update/add")
    gone = set(obj.get("remove") or [])
    racks = [dict(r) for r in inc.get("racks", []) if r.get("id") not in gone]
    by_id = {r.get("id"): r for r in racks}
    for u in obj.get("update") or []:
        if isinstance(u, dict) and u.get("id") in by_id:
            by_id[u["id"]].update({k: v for k, v in u.items() if k != "id"})
    racks += [dict(a) for a in obj.get("add") or [] if isinstance(a, dict)]
    return {"racks": racks}


# ---------------------------------------------------------------- solver
def solve(hall, llm=None, iters=6, model=None, library_dir="library", start="greedy", log=print,
          max_tokens=3000, temperature=0.4, patience=3):
    """Strict hill climbing over LLM proposals. Returns (best_payload, history).

    start: "greedy" | "empty" | a payload dict to use as the incumbent.
    """
    from donetwin import score as scoring
    from donetwin import llm as llm_mod
    if llm is None:
        llm = llm_mod.LLM()
    model = model or llm_mod.PLANNER_MODEL
    log = log or (lambda *a, **k: None)

    if isinstance(start, dict):
        inc = {"racks": list(start.get("racks", []))}
    elif start == "empty":
        inc = {"racks": []}
    else:
        from donetwin import greedy
        inc = greedy.solve(hall)
    inc_res = scoring.score(hall, inc)
    inc_s = float(inc_res.get("score", 0))
    history = [{"iter": 0, "score": inc_s, "feasible": bool(inc_res.get("feasible")), "accepted": True,
                "feedback_head": "; ".join(_feedback_lines(inc_res, 2)), "start": start if isinstance(start, str) else "payload"}]
    log(f"[hill] start={history[0]['start']} score={inc_s:.2f} feasible={inc_res.get('feasible')}")

    examples = nearest_examples(hall, load_library(library_dir)) if library_dir else []
    stale, last = 0, None
    for it in range(1, iters + 1):
        prompt = build_prompt(hall, inc, inc_res, examples, last)
        messages = [{"role": "system", "content": "You are an expert data-center layout engineer. "
                                                  "You output only valid JSON payloads."},
                    {"role": "user", "content": prompt}]
        rec = {"iter": it, "score": None, "feasible": False, "accepted": False, "feedback_head": ""}
        try:
            text = llm.chat(model, messages, max_tokens=max_tokens, temperature=temperature, retries=2,
                            extra_body=NO_THINK)
        except Exception as e:  # network / API failure: record and keep going
            rec.update(error="llm", feedback_head=f"llm error: {e}"[:200])
            text = None
        cand = None
        if text is not None:
            try:
                cand = apply_edits(inc, _parse(text))
            except (ValueError, json.JSONDecodeError) as e:
                rec.update(error="parse", feedback_head=f"parse error: {e}"[:200])
                last = "your reply had no parseable JSON object; reply with JSON only."
        if cand is not None:
            try:
                res = scoring.score(hall, cand)
            except Exception as e:  # verifier rejected malformed structure outright
                res = {"score": 0.0, "feasible": False, "feedback": f"malformed payload: {e}"}
            s = float(res.get("score", 0) or 0)
            fb = _feedback_lines(res)
            rec.update(score=s, feasible=bool(res.get("feasible")), feedback_head="; ".join(fb[:2]))
            if s > inc_s:
                rec["accepted"] = True
                inc, inc_res, inc_s, last = cand, res, s, None
            else:
                last = (f"it scored {s:.2f} (feasible {res.get('feasible')}) vs incumbent {inc_s:.2f}. "
                        f"Its feedback:\n" + ("\n".join(fb[:12]) or "(none)"))
        history.append(rec)
        log(f"[hill] iter {it}: score={rec['score']} accepted={rec['accepted']} {rec['feedback_head'][:120]}")
        stale = 0 if rec["accepted"] else stale + 1
        if stale >= patience:
            log(f"[hill] stopping: {patience} consecutive non-improving iterations")
            break

    if inc_res.get("feasible") and library_dir:
        if save_to_library(library_dir, hall, inc, inc_s):
            log(f"[hill] saved {_lib_path(library_dir, hall)} (score {inc_s:.2f})")
    return inc, history
