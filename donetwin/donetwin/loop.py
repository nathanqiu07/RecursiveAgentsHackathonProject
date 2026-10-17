"""The recursive loop: spec -> (layout -> sandbox -> verify -> inspect -> decide)* -> done | escalate."""
import json
import os
import threading
import time
import traceback

from . import agents
from .checks import feeder_loads, gap_between, swing_hits, verify
from .llm import PLANNER_MODEL, VISION_MODEL, Usage
from .render import render_views
from .research import code_references
from .sandbox import run_layout
from .scene import Scene, layout_signature
from .usd import to_usda

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs")


class Run:
    def __init__(self, run_id, goal, scenario):
        self.run_id = run_id
        self.goal = goal
        self.scenario = scenario
        self.dir = os.path.join(RUNS_DIR, run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.events = []
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.finished = False
        self.status = "running"
        self.listeners = []

    def emit(self, type_, **data):
        with self.lock:
            ev = {"seq": len(self.events), "t": round(time.time() - self.t0, 2), "type": type_, **data}
            self.events.append(ev)
            with open(os.path.join(self.dir, "events.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")
        for fn in self.listeners:
            fn(ev)
        return ev

    def log(self, agent, msg):
        self.emit("log", agent=agent, msg=msg)

    def save(self, rel, content):
        path = os.path.join(self.dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else json.dumps(content, indent=2))
        return path


def _points(scene_dict):
    s = Scene(scene_dict)
    pts = {}
    for o in scene_dict["objects"]:
        for name in o.get("points", {}):
            pts[f"{o['id']}.{name}"] = list(s.point(f"{o['id']}.{name}"))
    return pts


def _robot_geometry(scene_dict, spec):
    s = Scene(scene_dict)
    robots = []
    links = {n["id"]: n for n in spec.get("new_objects", [])}
    for o in scene_dict["objects"]:
        if o["kind"] != "robot":
            continue
        link = links.get(o["id"], {})
        sw = next((c["params"] for c in spec["checks"]
                   if c["type"] == "swing_clear" and c["params"].get("robot") == o["id"]), {})
        pick, place = sw.get("from") or link.get("pick"), sw.get("to") or link.get("place")
        g = {"id": o["id"], "x": o["x"], "y": o["y"], "reach": o["reach"]}
        try:
            g["pick"], g["place"] = list(s.point(pick)), list(s.point(place))
            hits, (a0, a1, rad) = swing_hits(s, o, pick, place, sw.get("against_kinds", ["pillar", "fence", "rack"]),
                                             float(sw.get("arm_width", 0.4)))
            g["sweep"] = {"a0": round(a0, 1), "a1": round(a1, 1), "radius": round(rad, 2), "hits": hits,
                          "dir": "ccw" if not hits["ccw"] else ("cw" if not hits["cw"] else None)}
        except Exception:
            pass
        robots.append(g)
    geo = {"robots": robots}
    racks = [o for o in scene_dict["objects"] if o["kind"] == "rack"]
    if racks:
        geo["loads"] = feeder_loads(scene_dict)
        geo["links"] = [{"rack": r["id"], "cdu": r.get("cdu"), "busway": r.get("busway")} for r in racks]
    return geo


def _adjudicate(scene_dict, issues):
    """Check the inspector's claims against measurement. Returns (rulings, unresolved_issues)."""
    rulings, unresolved = [], []
    for iss in issues:
        objs = [o for o in iss.get("objects", []) if isinstance(o, str)]
        if iss.get("severity") == "minor":
            rulings.append({"claim": iss["description"], "measured": None, "ruling": "noted (minor)"})
            continue
        gaps = []
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                g = gap_between(scene_dict, a, b)
                if g is not None:
                    gaps.append((a, b, g))
        if gaps and all(g >= 0.05 for _, _, g in gaps):
            meas = ", ".join(f"{a}<->{b} gap {g:.2f} m" for a, b, g in gaps)
            rulings.append({"claim": iss["description"], "measured": meas,
                            "ruling": "overruled: geometry shows clear separation"})
        else:
            meas = ", ".join(f"{a}<->{b} gap {g:.2f} m" for a, b, g in gaps) or "not measurable"
            rulings.append({"claim": iss["description"], "measured": meas, "ruling": "upheld"})
            unresolved.append(iss)
    return rulings, unresolved


def execute(run, llm, base_scene, max_iters=6, sandbox_mode="subprocess", vision_mode="final"):
    """vision_mode: 'final' (inspect candidate-final layouts), 'every' (every iteration), 'off'."""
    try:
        _execute(run, llm, base_scene, max_iters, sandbox_mode, vision_mode)
    except Exception as e:
        run.log("supervisor", f"run crashed: {e}")
        run.save("crash.txt", traceback.format_exc())
        run.status = "failed"
        run.emit("done", status="failed", iterations=0, usage=llm.usage.to_dict(), summary=str(e), usd_path=None)
    finally:
        run.finished = True


def _execute(run, llm, base_scene, max_iters, sandbox_mode, vision_mode):
    goal = run.goal
    run.emit("run_start", goal=goal, scenario=base_scene, models={"planner": PLANNER_MODEL, "vision": VISION_MODEL},
             sandbox=sandbox_mode)

    refs = []
    if base_scene.get("research_queries") and os.getenv("YDC_API_KEY"):
        run.log("research", f"Searching {len(base_scene['research_queries'])} code/standard references (You.com)...")
        refs = code_references(base_scene)
        run.save("references.json", refs)
        run.emit("research", references=refs)
        run.log("research", f"{len(refs)} cited references for the Spec agent")
    run.log("spec", "Turning the one-line goal into a machine-checkable Definition of Done...")
    spec = agents.spec_agent(llm, goal, base_scene, refs)
    run.save("spec.json", spec)
    run.emit("spec", spec=spec)
    run.log("spec", f"{len(spec['checks'])} checks, {len(spec.get('assumptions', []))} explicit assumptions"
            + (f"; dropped unknown check types {spec['dropped_checks']}" if spec.get("dropped_checks") else ""))

    history, seen = [], {}
    best = None
    stall = disagreements = 0
    replan_note = None
    temperature = 0.4

    for it in range(1, max_iters + 1):
        run.emit("iteration_start", iter=it)
        run.log("layout", "Writing layout program" + (" (strategy reset)" if replan_note else "") + "...")
        code = agents.layout_agent(llm, goal, spec, base_scene, history, replan_note, temperature)
        replan_note = None
        run.save(f"iter_{it:02d}/layout.py", code)
        run.emit("layout", iter=it, code=code, note="")

        res = run_layout(code, base_scene, os.path.join(run.dir, f"iter_{it:02d}", "sandbox"), mode=sandbox_mode)
        scene = res.get("scene")
        run.emit("sandbox", iter=it, ok=res["ok"], error=res.get("error"), stdout=res.get("stdout", ""),
                 scene=scene, points=_points(scene) if scene else {}, mode=res.get("mode"))
        if not res["ok"]:
            run.log("sandbox", f"Program failed: {res['error'].splitlines()[-1] if res['error'] else '?'}")
            history.append({"iter": it, "code": code, "error": res["error"]})
            run.emit("decision", iter=it, action="continue",
                     reason="The program crashed in the sandbox. Feeding the traceback back to the layout agent.")
            continue

        run.save(f"iter_{it:02d}/scene.json", scene)
        report = verify(scene, base_scene, spec["checks"])
        run.save(f"iter_{it:02d}/verify.json", report)
        run.emit("verify", iter=it, report=report, geometry=_robot_geometry(scene, spec))
        failing = [c for c in report["checks"] if not c["passed"]]
        run.log("verifier", f"{report['passed']}/{report['total']} checks pass"
                + ("" if not failing else f"; failing: {', '.join(sorted({c['type'] for c in failing}))}"))

        png = render_views(scene, os.path.join(run.dir, f"iter_{it:02d}", "views.png"), f"iteration {it}")
        vision = {"verdict": "skipped", "issues": []}
        if vision_mode == "every" or (vision_mode == "final" and report["all_pass"]):
            run.log("vision", "Inspector reviewing plan + elevation drawings...")
            try:
                vision = agents.vision_inspector(llm, goal, spec, png)
            except Exception as e:
                vision = {"verdict": "skipped", "issues": [], "error": str(e)}
                run.log("vision", f"Inspector unavailable: {e}")
        rulings, unresolved = _adjudicate(scene, vision["issues"]) if vision["verdict"] == "fail" else ([], [])
        vision["adjudication"] = rulings
        run.save(f"iter_{it:02d}/vision.json", vision)
        run.emit("vision", iter=it, verdict=vision["verdict"], issues=vision["issues"], adjudication=rulings)

        robot_pos = {o["id"]: (round(o["x"], 2), round(o["y"], 2)) for o in scene["objects"] if not o.get("fixed")}
        entry = {"iter": it, "code": code, "positions": robot_pos,
                 "violations": [{"type": c["type"], **v} for c in failing for v in c["violations"]],
                 "vision_concerns": [i["description"] for i in unresolved]}
        history.append(entry)
        if best is None or report["penalty"] < best["penalty"]:
            if best is not None and report["penalty"] < best["penalty"] - 0.01:
                stall = 0
            best = {"iter": it, "penalty": report["penalty"], "scene": scene, "report": report}
        else:
            stall += 1

        if report["all_pass"] and not unresolved:
            why = "All checks pass" + (" and the inspector signed off." if vision["verdict"] == "pass" else
                                       (" and every inspector claim was overruled by measurement." if rulings else "."))
            run.emit("decision", iter=it, action="done", reason=why)
            return _finish(run, llm, spec, scene, report, vision, it, "verified")

        if report["all_pass"]:
            disagreements += 1
            if disagreements >= 2:
                run.emit("decision", iter=it, action="escalate",
                         reason="Geometry passes but the inspector keeps flagging issues the checks cannot "
                                "measure. A human should decide whether the Definition of Done is missing a rule.")
                return _finish(run, llm, spec, scene, report, vision, it, "escalated")
            run.emit("decision", iter=it, action="continue",
                     reason="Geometry passes but the inspector upheld concerns. Sending them to the layout agent.")
            continue

        sig = layout_signature(scene)
        if sig in seen:
            replan_note = (f"LOOP DETECTED: iteration {it} reproduced the layout of iteration {seen[sig]}. "
                           f"Positions already tried: {[h.get('positions') for h in history if h.get('positions')]}. "
                           "Abandon this region: try the opposite side of the blocking obstacles, and a different "
                           "arm sweep direction.")
            temperature = 0.8
            run.emit("decision", iter=it, action="replan", reason=replan_note)
        elif stall >= 2:
            replan_note = (f"DRIFT: no improvement for {stall} iterations (best penalty {best['penalty']} at "
                           f"iteration {best['iter']}). Re-plan from scratch with a different placement strategy. "
                           f"Positions already tried: {[h.get('positions') for h in history if h.get('positions')]}.")
            stall = 0
            temperature = 0.8
            run.emit("decision", iter=it, action="replan", reason=replan_note)
        else:
            temperature = 0.4
            run.emit("decision", iter=it, action="continue",
                     reason=f"{len(failing)} checks failing (penalty {report['penalty']}). Sending exact fixes back.")
        seen.setdefault(sig, it)

    run.emit("decision", iter=max_iters, action="escalate",
             reason=f"Budget of {max_iters} iterations exhausted. Best attempt: iteration {best['iter'] if best else '-'}.")
    if best:
        return _finish(run, llm, spec, best["scene"], best["report"], {"verdict": "skipped"}, max_iters, "escalated")
    run.status = "escalated"
    run.emit("done", status="escalated", iterations=max_iters, usage=llm.usage.to_dict(),
             summary="No runnable layout produced.", usd_path=None)


def _finish(run, llm, spec, scene, report, vision, iters, status):
    usd_path = run.save("final/layout.usda", to_usda(scene, doc=f"DoneTwin {status}: {run.goal}"))
    run.save("final/scene.json", scene)
    lines = [f"# DoneTwin {'VERIFIED' if status == 'verified' else 'ESCALATED TO HUMAN'}", "",
             f"**Goal:** {run.goal}", "", f"**Iterations:** {iters}  ", f"**Usage:** {llm.usage.to_dict()}", "",
             "## Definition of Done (written by the Spec agent)", ""]
    for c in report["checks"]:
        lines.append(f"- [{'x' if c['passed'] else ' '}] `{c['type']}` {json.dumps(c.get('params', {}))} - {c.get('why', '')}")
        for v in c["violations"]:
            lines.append(f"    - FAIL: {v['detail']}")
    lines += ["", "## Assumptions made explicit", ""] + [f"- {a}" for a in spec.get("assumptions", [])]
    lines += ["", f"## Inspector verdict: {vision.get('verdict')}", ""]
    for r in vision.get("adjudication", []):
        lines.append(f"- {r['claim']} -> {r['ruling']} ({r['measured']})")
    run.save("final/DONE.md", "\n".join(lines) + "\n")
    run.status = status
    summary = (f"{report['passed']}/{report['total']} checks pass after {iters} iteration(s)."
               if status == "verified" else
               f"Stopped at {report['passed']}/{report['total']} checks; human review requested.")
    run.emit("done", status=status, iterations=iters, usage=llm.usage.to_dict(), summary=summary,
             usd_path=os.path.relpath(usd_path, ROOT).replace("\\", "/"))
