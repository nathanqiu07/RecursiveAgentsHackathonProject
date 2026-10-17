"""Hill-climb run on a 2N hall, streamed as the event types the 3D twin already renders.

Iteration 1 is the greedy baseline. Each later iteration is one Nemotron proposal, scored by
donetwin.score (no LLM in the scoring path) and kept only if it beats the incumbent. The run ends by
re-showing the best layout, so the final frame in the UI is always the winner.
"""
import json
import re

from . import greedy, hill_climb, score
from .llm import PLANNER_MODEL

# Nemotron 3 Super spends its whole output budget on hidden reasoning unless thinking is off (ac0c651).
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

RULES = [
    ("payload", "Only rack decisions are accepted; kW, sizes and capacities come from the catalog"),
    ("no_collision", "No two physical footprints overlap"),
    ("in_bounds", "Every rack lies inside the floor"),
    ("zones_clear", "Walkways and keepouts stay clear"),
    ("aisle_width", "Racks in a row abut; facing rows are >= aisle_min_m apart"),
    ("cooling_type", "Liquid racks need a CDU; air-cooled racks must not use one"),
    ("cdu_capacity", "CDU load <= capacity"),
    ("hose", "Manhattan hose run to the CDU port <= hose_m"),
    ("dual_feed", "Two different busway feeds, both within tap reach"),
    ("power_2n", "Either feed alone carries the full rack load"),
    ("air_capacity", "Air-cooled racks fit the room air-cooling budget"),
    ("hot_cold_aisle", "Rows face front-to-front or back-to-back, never front-to-back"),
]


def is_2n_hall(hall):
    return any(o.get("kind") == "busway" and o.get("feed") for o in hall.get("objects", []))


def _ui_scene(hall, payload):
    scene, _ = score.rebuild(hall, payload)
    if scene is None:
        return None
    for o in scene["objects"]:
        if o.get("kind") == "rack":
            o["busway"] = o.get("feed_a")  # the twin's load overlay reads a single busway field
    return scene


def _ui_report(hall, res):
    ids = [o["id"] for o in hall.get("objects", [])]
    checks = []
    for c in res.get("checks", []):
        if not c.get("hard", True):
            continue
        vs = []
        for v in c.get("violations", []):
            text = v.get("detail", "")
            objs = sorted(set(re.findall(r"\bR[\w-]*", text)) | {i for i in ids if i in text})
            vs.append({"objects": objs, "magnitude": v.get("magnitude", 1.0), "detail": text, "fix": v.get("fix")})
        why = next((d for n, d in RULES if n == c["name"]), "")
        checks.append({"type": c["name"], "params": {}, "why": why, "passed": c["passed"], "violations": vs})
    failing = [c for c in checks if not c["passed"]]
    return {"checks": checks, "passed": len(checks) - len(failing), "total": len(checks),
            "all_pass": bool(res.get("feasible")),
            "penalty": round(sum(1 + sum(v["magnitude"] for v in c["violations"]) for c in failing), 3),
            "score": res.get("score"), "components": res.get("components"),
            "it_kw": res.get("it_kw"), "upper_bound_kw": res.get("upper_bound_kw")}


def _summary(res):
    c = res.get("components") or {}
    return (f"score {res.get('score', 0):.2f} · {res.get('it_kw', 0):.0f}/{res.get('upper_bound_kw', 0):.0f} kW · "
            f"resilience {c.get('resilience', 0):.2f} · hose {c.get('hose', 0):.2f}")


def _show(run, hall, it, payload, res, note):
    run.emit("iteration_start", iter=it)
    run.emit("layout", iter=it, code=json.dumps(payload, indent=1), note=note)
    scene = _ui_scene(hall, payload)
    run.emit("sandbox", iter=it, ok=scene is not None, error=None if scene else "payload rejected",
             stdout="", scene=scene, points={}, mode="score")
    run.emit("verify", iter=it, report=_ui_report(hall, res), geometry=None)


def execute_hall(run, llm, hall, iters=6, model=None, library_dir="library", patience=3,
                 max_tokens=16000, temperature=0.4):
    try:
        _execute(run, llm, hall, iters, model or PLANNER_MODEL, library_dir, patience, max_tokens, temperature)
    except Exception as e:  # never leave the UI waiting on a dead run
        run.emit("done", status="failed", iterations=0, usage=llm.usage.to_dict() if llm else {},
                 summary=f"{type(e).__name__}: {e}", usd_path=None)
    finally:
        run.finished = True


def _execute(run, llm, hall, iters, model, library_dir, patience, max_tokens, temperature):
    bound = score.upper_bound_kw(hall)
    run.emit("run_start", goal=hall.get("goal", ""), scenario=hall,
             models={"planner": model if llm else "greedy (no LLM)", "vision": "verifier: donetwin.score"})
    run.emit("spec", spec={
        "interpretation": f"Maximise verified IT load in {hall.get('title', hall.get('name'))}. "
                          f"Upper bound {bound:.0f} kW. Score = 70 density + 20 resilience + 10 hose.",
        "assumptions": ["2N power: each rack has feed_a and feed_b; either alone carries the full load",
                        "N+1 cooling is rewarded, not required: resilience = share of CDU failures survivable",
                        "Air-cooled HGX-B200 racks use room air cooling, not a CDU",
                        "Any hard-rule violation scores 0"],
        "checks": [{"type": n, "params": {}, "why": d} for n, d in RULES],
        "new_objects": []})

    inc = greedy.solve(hall)
    inc_res = score.score(hall, inc)
    base_s = inc_s = float(inc_res["score"])
    run.log("layout", f"Greedy baseline: {_summary(inc_res)}")
    _show(run, hall, 1, inc, inc_res, "greedy baseline (no LLM)")
    if llm is None or iters <= 0:
        run.emit("decision", iter=1, action="done", reason=f"Baseline only. {_summary(inc_res)}")
        return _finish(run, llm, 1, inc_res, base_s)
    run.emit("decision", iter=1, action="continue",
             reason=f"Baseline {inc_s:.2f}. Asking Nemotron for a layout that scores higher.")

    examples = hill_climb.nearest_examples(hall, hill_climb.load_library(library_dir)) if library_dir else []
    if examples:
        run.log("supervisor", f"Using {len(examples)} verified layout(s) from the library as examples")
    stale, last, it = 0, None, 1
    for k in range(1, iters + 1):
        it = k + 1
        run.log("layout", f"Nemotron proposal {k}/{iters} (incumbent {inc_s:.2f})...")
        prompt = hill_climb.build_prompt(hall, inc, inc_res, examples, last)
        messages = [{"role": "system", "content": "You are an expert data-center layout engineer. "
                                                  "You output only valid JSON payloads."},
                    {"role": "user", "content": prompt}]
        try:
            cand = hill_climb._parse(llm.chat(model, messages, max_tokens=max_tokens, temperature=temperature,
                                              retries=2, extra_body=NO_THINK))
        except Exception as e:
            run.emit("iteration_start", iter=it)
            run.emit("sandbox", iter=it, ok=False, error=f"no usable payload: {e}"[:400], stdout="", scene=None,
                     points={}, mode="score")
            last = "your reply had no parseable JSON payload; reply with JSON only."
            run.emit("decision", iter=it, action="continue", reason="Reply was not a usable payload. Retrying.")
            stale += 1
            if stale >= patience:
                break
            continue
        res = score.score(hall, cand)
        s = float(res.get("score", 0) or 0)
        _show(run, hall, it, cand, res, f"Nemotron proposal {k}")
        fb = hill_climb._feedback_lines(res)
        if s > inc_s:
            run.emit("decision", iter=it, action="continue",
                     reason=f"Accepted: {inc_s:.2f} -> {s:.2f} (+{s - inc_s:.2f}). {_summary(res)}")
            inc, inc_res, inc_s, last, stale = cand, res, s, None, 0
        else:
            why = "infeasible: " + "; ".join(fb[:2]) if not res.get("feasible") else _summary(res)
            run.emit("decision", iter=it, action="continue",
                     reason=f"Rejected ({s:.2f} <= {inc_s:.2f}). {why}"[:400])
            last = (f"it scored {s:.2f} (feasible {res.get('feasible')}) vs incumbent {inc_s:.2f}. "
                    f"Its feedback:\n" + ("\n".join(fb[:12]) or "(none)"))
            stale += 1
            if stale >= patience:
                run.log("supervisor", f"No improvement for {patience} proposals; stopping.")
                break

    it += 1
    _show(run, hall, it, inc, inc_res, "best verified layout")
    if inc_res.get("feasible") and library_dir and hill_climb.save_to_library(library_dir, hall, inc, inc_s):
        run.log("supervisor", f"Saved to the verified-layout library (score {inc_s:.2f})")
    run.emit("decision", iter=it, action="done",
             reason=f"Best {inc_s:.2f} vs greedy {base_s:.2f} ({inc_s - base_s:+.2f}). {_summary(inc_res)}")
    return _finish(run, llm, it, inc_res, base_s)


def _finish(run, llm, it, res, base_s):
    status = "verified" if res.get("feasible") else "escalated"
    run.save("final/hall_result.json", {"result": res, "greedy_score": base_s})
    run.emit("done", status=status, iterations=it, usage=llm.usage.to_dict() if llm else {},
             summary=f"Hall {_summary(res)} (greedy {base_s:.2f})", usd_path=None)
