"""Prompts and calls for the Spec agent, Layout agent and Vision inspector."""
import json

from .checks import CHECK_LIBRARY, MANDATORY
from .llm import PLANNER_MODEL, VISION_MODEL, extract_code, extract_json, image_part
from .scene import RACK_CATALOG, ROBOT_CATALOG

SCENE_API_DOC = """Scene API (the variable `scene` is pre-bound; only `math` may be imported):
  scene.get(id) -> dict with x, y, w, d, h, rot, kind, fixed, ...   (x, y = footprint center, meters)
  scene.ids(kind=None) -> list of ids
  scene.point("obj.point") -> (x, y) world coords of a named interaction point; scene.point("obj") -> center
  scene.add_robot(id, model, x, y, rot=0)     model from the robot catalog; footprint/reach come from the catalog
  scene.add_rack(id, model, x, y, rot=0, cdu=None, busway=None)   model from the rack catalog (w 0.6 along x, d 1.2 along y)
  scene.assign(id, cdu=None, busway=None)     set a rack's cooling / power feed
  scene.move(id, x=None, y=None, rot=None)
  scene.translate(id, dx=0, dy=0)
  scene.add_object(id, kind, x, y, w, d, h, rot=0)
  scene.remove(id)
  print(...) to leave a short note for the inspector."""


def _scene_brief(scene_dict):
    rows = []
    for o in scene_dict["objects"]:
        r = {k: o[k] for k in ("id", "kind", "x", "y", "w", "d", "h", "rot", "fixed") if k in o}
        for k in ("z", "points", "note", "model", "reach", "min_reach", "reason", "capacity_kw", "derate",
                  "hose_m", "tap_m", "kw", "kg", "cdu", "busway"):
            if k in o:
                r[k] = o[k]
        rows.append(r)
    return json.dumps({"floor_m": scene_dict["floor"], "objects": rows}, indent=1)


SPEC_SYSTEM = """You are the SPEC AGENT of an industrial layout team. A human gives a one-line goal for a factory cell.
Agents usually fill vague goals with silent assumptions and then report success. You do the opposite: you turn the
goal into an explicit, machine-checkable DEFINITION OF DONE that other agents cannot argue with.

Rules:
- Make every assumption explicit (e.g. what "heavy-duty" means in kg, which robot model, safety margins and why).
- Compose checks ONLY from the check library. Use real object ids and point refs ("obj.point") from the scene.
- Cover everything a site engineer would sign off: the requested capacity/equipment actually exists, every
  physical interaction works (reach, swing, alignment), power and cooling limits incl. redundancy, maintenance
  clearance, egress and pedestrian protection, code-mandated working space and structural keep-outs.
- If CODE REFERENCES are provided, ground your thresholds in them and attach the matching one to a check as
  "source": {"title", "url", "snippet"}. Never invent URLs.
- Choose numbers like a plant safety engineer (clearances 0.4-1.0 m, reach margin 0.85-0.95).
- Fixed objects can never move. Decide which non-fixed objects may be moved to satisfy the goal.
Respond with ONLY a JSON object:
{"interpretation": str,
 "assumptions": [str],
 "new_objects": [{"id": str, "kind": "robot", "model": str, "pick": "obj.point", "place": "obj.point"}],
 "movable": [ids],
 "checks": [{"type": str, "params": {...}, "why": str}],
 "escalate_if": [str]}"""


def spec_agent(llm, goal, scene_dict, references=None):
    user = (f"GOAL: {goal}\n\nDOMAIN NOTES: {scene_dict.get('domain_notes', '-')}\n\n"
            f"SCENE:\n{_scene_brief(scene_dict)}\n\n"
            f"ROBOT CATALOG:\n{json.dumps(ROBOT_CATALOG, indent=1)}\n\n"
            f"RACK CATALOG:\n{json.dumps(RACK_CATALOG, indent=1)}\n\n"
            f"CHECK LIBRARY:\n{json.dumps(CHECK_LIBRARY, indent=1)}")
    if references:
        user += "\n\nCODE REFERENCES (live web search):\n" + json.dumps(references, indent=1)
    text = llm.chat(PLANNER_MODEL, [{"role": "system", "content": SPEC_SYSTEM},
                                    {"role": "user", "content": user}], max_tokens=6000, temperature=0.2)
    spec = extract_json(text)
    return normalize_spec(spec, scene_dict)


def normalize_spec(spec, scene_dict=None):
    """Drop unknown check types and enforce the non-negotiable invariants."""
    checks = [c for c in spec.get("checks", []) if c.get("type") in CHECK_LIBRARY]
    dropped = [c.get("type") for c in spec.get("checks", []) if c.get("type") not in CHECK_LIBRARY]
    have = {c["type"] for c in checks}
    for m in MANDATORY:
        if m["type"] not in have:
            checks.insert(0, dict(m))
    for n in spec.get("new_objects", []):
        if n.get("kind") == "rack":
            continue  # fleets of racks are covered by total_power / capacity checks
        if not any(c["type"] == "exists" and c.get("params", {}).get("id") == n["id"] for c in checks):
            checks.insert(0, {"type": "exists", "params": {"id": n["id"], "kind": n.get("kind", "robot")},
                              "why": "mandatory: the requested equipment must actually be in the layout"})
    # Every walkway / keep-out zone in the site is protected, whether or not the Spec agent remembered it.
    has_robot = any(n.get("kind") == "robot" for n in spec.get("new_objects", []))
    for z in (scene_dict or {}).get("objects", []):
        if z["kind"] in ("walkway", "keepout") and not any(
                c["type"] == "zone_clear" and c.get("params", {}).get("zone") == z["id"] for c in checks):
            checks.append({"type": "zone_clear",
                           "params": {"zone": z["id"], "include_envelopes": has_robot and z["kind"] == "walkway"},
                           "why": f"mandatory: {z.get('reason') or z.get('note') or z['kind']} must stay clear"})
    spec["checks"] = checks
    spec["dropped_checks"] = dropped
    return spec


LAYOUT_SYSTEM = """You are the LAYOUT AGENT. You write a short Python program that edits a factory cell through the Scene API
so that EVERY check in the Definition of Done passes. You cannot see the result directly: a sandbox runs your program,
a geometry verifier measures it, and an inspector reviews drawings. Their findings come back to you.

Rules:
- Never move or remove fixed objects. Only touch new objects and objects listed as movable.
- Use the numeric fixes the verifier gives you; they are exact measurements. Combine fixes for several violations,
  and check that a fix for one check does not break another (reach vs clearance vs swing vs walkway).
- Prefer computing positions with math from scene.point()/scene.get() over hard-coding when it helps.
- Use loops for fleets of equipment (e.g. rows of racks). Keep the program under 60 lines.
- End with print() of a one-line rationale.
Reply with the program in ONE ```python block, and nothing else of substance."""


def layout_agent(llm, goal, spec, base_scene, history, replan_note=None, temperature=0.4):
    parts = [f"GOAL: {goal}",
             f"DEFINITION OF DONE:\n{json.dumps({k: spec[k] for k in ('assumptions', 'new_objects', 'movable', 'checks') if k in spec}, indent=1)}",
             f"DOMAIN NOTES: {base_scene.get('domain_notes', '-')}",
             f"BASE SCENE (your program starts from this every time):\n{_scene_brief(base_scene)}",
             f"ROBOT CATALOG:\n{json.dumps(ROBOT_CATALOG)}",
             f"RACK CATALOG:\n{json.dumps(RACK_CATALOG)}",
             SCENE_API_DOC]
    if history:
        parts.append("PREVIOUS ATTEMPTS (oldest first):")
        for h in history[-3:]:
            parts.append(f"--- iteration {h['iter']} program:\n```python\n{h['code']}\n```")
            if h.get("error"):
                parts.append(f"SANDBOX ERROR: {h['error']}")
            for v in h.get("violations", []):
                parts.append(f"FAIL [{v['type']}] {v['detail']}  ->  FIX: {v['fix']}")
            for c in h.get("vision_concerns", []):
                parts.append(f"INSPECTOR: {c}")
            if h.get("positions"):
                parts.append(f"positions used: {h['positions']}")
    if replan_note:
        parts.append(f"SUPERVISOR: {replan_note}")
    parts.append("Write the complete program now.")
    text = llm.chat(PLANNER_MODEL, [{"role": "system", "content": LAYOUT_SYSTEM},
                                    {"role": "user", "content": "\n\n".join(parts)}],
                    max_tokens=6000, temperature=temperature)
    return extract_code(text)


VISION_SYSTEM = """You are an independent FACTORY SAFETY INSPECTOR. You see only engineering drawings (top plan and
front elevation, meters, 1 m grid) of a proposed cell layout. You have NOT seen any automated check results.
Inspect like a human engineer: footprints that overlap or touch, robots whose dashed reach circle covers the
pedestrian walkway, arm paths blocked by columns, a bin not under its overhead sensor, equipment outside the floor,
anything that would fail a site safety walk. Only report what you can actually see. Name objects by their labels.
Respond with ONLY JSON:
{"verdict": "pass" | "fail",
 "issues": [{"severity": "critical" | "major" | "minor", "objects": [ids], "description": str}]}"""


def vision_inspector(llm, goal, spec, png_path):
    content = [{"type": "text", "text": f"Goal of the change: {goal}\n"
                                        f"The new equipment: {json.dumps(spec.get('new_objects', []))}\n"
                                        "Inspect the drawing and return your JSON verdict."},
               image_part(png_path)]
    text = llm.chat(VISION_MODEL, [{"role": "system", "content": VISION_SYSTEM},
                                   {"role": "user", "content": content}], max_tokens=6000, temperature=0.1)
    out = extract_json(text)
    out.setdefault("issues", [])
    out["verdict"] = "pass" if str(out.get("verdict", "")).lower().startswith("pass") else "fail"
    return out
