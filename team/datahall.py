"""GPU data-hall domain: Nemotron roles lay out 2N AI halls, scored only by Rikin's DoneTwin scorer.

Tasks are halls from `donetwin.halls.generate(seed, tier)`: train and dev draw disjoint seeds from
0..9999, held-out draws from `halls.heldout_seeds()` (10000..10049), tiers mixed small/medium/large.

Scoring is `donetwin.score.score(hall, payload)` and nothing else, never an LLM: any hard-rule
violation scores 0; a feasible layout scores 0..100 = 70 density + 20 resilience to a single CDU
failure + 10 hose economy. RunResult.score = that / 100 and passed = feasible. There is no repair
step and no slot grid here: the scorer's own rules are the contract. Graded progress comes from
density, resilience and hose, and the team recovers from a 0 by iterating with the scorer's
violation details and fix hints (the verifier_feedback tool) between Layout attempts.

failures are the violated check names (e.g. "hose", "power_2n", "aisle_width", soft "resilience")
so recurring patterns are countable; each violation's detail and fix hint go to RunResult.notes.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DONETWIN = Path(__file__).resolve().parent.parent / "donetwin"
if str(DONETWIN) not in sys.path:
    # Appended, not inserted first: donetwin/ also holds top-level run.py, server.py and a tests package.
    sys.path.append(str(DONETWIN))

from donetwin import greedy, halls  # noqa: E402
from donetwin import score as scoring  # noqa: E402
from donetwin.hill_climb import FORMAT, RULES, hall_summary  # noqa: E402
from donetwin.scene import RACK_CATALOG  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.crusoe import NANO, extract_json  # noqa: E402
from team.evolve import TeamVersion  # noqa: E402
from team.types import LLM, Reply, RunResult, Task  # noqa: E402

TIERS = ("small", "medium", "large")
SPLIT_SIZES = (12, 8, 15)
TOOL_CATALOG = ["verifier_feedback", "capacity_calc", "greedy_seed"]
SEVERITY = scoring.RULES + ["resilience"]
LIQUID_KW = {m: s["kw"] for m, s in RACK_CATALOG.items() if scoring.COOLING[m] == "liquid"}
AIR_KW = {m: s["kw"] for m, s in RACK_CATALOG.items() if scoring.COOLING[m] == "air"}
RACK_W = RACK_CATALOG["GB200-NVL72"]["w"]

CONTRACTS: dict[str, str] = {
    "Planner": ('Reply with JSON only: {"row_racks": {"<row>": <int>}, "cdu_racks": {"<cdu id>": <int>}, '
                '"total_racks": <int>, "notes": "<short>"}'),
    "Layout": FORMAT,
    "Checker": 'Reply with JSON only: {"fix": "<concrete instructions for the next layout attempt>"}',
}
ADVISOR_CONTRACT = 'Reply with JSON only: {"advice": "<concrete guidance for the Layout role>"}'

V0_PROMPTS: dict[str, str] = {
    "Planner": ("You are the power and cooling Planner for a 2N AI data hall. Before any rack is placed, budget how "
                "many racks each busway row and each CDU can carry within every capacity rule, and the hall total."),
    "Layout": ("You are the Layout engineer for a 2N AI data hall. Place racks to maximize the score with zero "
               "hard-rule violations. Follow the Planner's budget, keep each row under its busway pair, and give "
               "each liquid rack a CDU its hose can reach."),
    "Checker": ("You are the Checker. Read the verifier feedback on the last layout and tell the Layout engineer "
                "exactly which racks to move, drop, rotate or reassign so the next attempt is feasible, keeping as "
                "much kW as possible."),
}

_CAPACITY = ("Planner", "Layout", "Checker")
_GEOMETRY = ("Layout", "Checker")
# Generic, hall-independent rules per scorer check; they reach the brain as run proposals.
LESSONS: dict[str, tuple[tuple[str, ...], str]] = {
    "payload": (("Layout",), "The payload is only {\"racks\": [...]}; each rack has exactly id, model, x, y, rot, cdu, "
                "feed_a, feed_b, with ids unique and different from every hall object id."),
    "no_collision": (_GEOMETRY, "Racks are 0.6 m wide at rot 0: in a row, centres sit exactly 0.6 m apart (abutting); "
                     "skip any x where a pillar, panel or CDU footprint overlaps."),
    "in_bounds": (_GEOMETRY, "Keep the whole rack footprint (x +/- 0.3, y +/- 0.6 at rot 0) inside the floor."),
    "zones_clear": (_GEOMETRY, "Keep footprints off walkway and keepout zones (centre +/- half size); egress corridors "
                    "run along the south and west walls."),
    "aisle_width": (_GEOMETRY, "Put every rack of a row exactly at its busway pair's row_y; rows facing across an "
                    "aisle need >= aisle_min_m between footprints."),
    "cooling_type": (("Layout",), "GB200/GB300 racks need cdu set to a CDU id; HGX-B200 air racks need cdu null."),
    "cdu_capacity": (_CAPACITY, "Sum of rack kW per CDU must stay <= capacity_kw; spread liquid racks across CDUs."),
    "hose": (_GEOMETRY, "Hose run is Manhattan |x - port_x| + |y - port_y| <= hose_m; rows far from the CDU ports "
             "reach only x near a port, so give each rack the nearest CDU with spare capacity."),
    "dual_feed": (_GEOMETRY, "Each rack takes feed_a = BW_<row>1 and feed_b = BW_<row>2 of its own row and sits at "
                  "that row's row_y, within tap_m of both busways and inside their x range."),
    "power_2n": (_CAPACITY, "2N: each busway carries the full kW of every rack it feeds, so a row holds at most "
                 "floor(capacity_kw * derate / rack kW) liquid racks; count GB300 at 142 kW."),
    "air_capacity": (_CAPACITY, "Total HGX-B200 air-cooled kW must stay <= the room's air_cooling_kw."),
    "hot_cold_aisle": (_GEOMETRY, "All racks in a row share one rot; adjacent rows alternate rot 0 and 180 so fronts "
                       "face fronts and backs face backs."),
    "resilience": (_CAPACITY, "Resilience is 20 of 100 points: if any one CDU fails, its racks must fit in the spare "
                   "capacity of other CDUs within hose reach, so leave headroom on every CDU."),
}


def load_tasks(seed: int = 0) -> tuple[list[Task], list[Task], list[Task]]:
    """Fixed, seeded splits: train 12 and dev 8 disjoint seeds from 0..9999, held-out 15 from heldout_seeds()."""
    rng = random.Random(seed)
    pool = rng.sample(range(10000), SPLIT_SIZES[0] + SPLIT_SIZES[1])
    held = rng.sample(halls.heldout_seeds(), SPLIT_SIZES[2])
    train, dev = pool[:SPLIT_SIZES[0]], pool[SPLIT_SIZES[0]:]
    return ([make_task(s, TIERS[i % 3], "train") for i, s in enumerate(train)],
            [make_task(s, TIERS[i % 3], "dev") for i, s in enumerate(dev)],
            [make_task(s, TIERS[i % 3], "heldout") for i, s in enumerate(held)])


def make_task(seed: int, tier: str, split: str) -> Task:
    """The greedy layout is the baseline and the greedy_seed tool's input; it reaches a prompt only through that tool."""
    hall = halls.generate(seed, tier)
    baseline = greedy.solve(hall)
    res = scoring.score(hall, baseline)
    return Task(id=hall["name"], split=split, text=describe(hall),
                payload={"hall": hall, "seed": seed, "tier": tier, "upper_bound_kw": res["upper_bound_kw"],
                         "greedy_layout": baseline, "greedy_score": res["score"], "greedy_feasible": res["feasible"]})


def _objs(hall: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [o for o in hall["objects"] if o.get("kind") == kind]


def describe(hall: dict[str, Any]) -> str:
    """Retrieval text using the words brain lessons use (busway, CDU, hose, pillar, keepout, air)."""
    bws, cdus = _objs(hall, "busway"), _objs(hall, "cdu")
    usable = [b["capacity_kw"] * b.get("derate", 1.0) for b in bws]
    caps, hoses = [c["capacity_kw"] for c in cdus], [c["hose_m"] for c in cdus]
    return (f"2N AI data hall ({hall['name'].split('-')[0]}): {len(bws) // 2} busway rows (A/B pairs, usable "
            f"{min(usable):.0f}-{max(usable):.0f} kW), {len(cdus)} CDU (capacity {min(caps)}-{max(caps)} kW, hose "
            f"{min(hoses)}-{max(hoses)} m), {len(_objs(hall, 'pillar'))} pillar, keepout "
            f"{', '.join(o['id'] for o in _objs(hall, 'keepout')) or 'none'}, air cooling {hall['air_cooling_kw']} kW")


def score_payload(hall: dict[str, Any], payload: Any) -> dict[str, Any]:
    """Rikin's scorer; a crash on malformed structure becomes a payload violation instead of killing the run."""
    try:
        return scoring.score(hall, payload)
    except Exception as exc:  # the scorer validates most structure, but a model can still send shapes it never saw
        v = {"detail": f"malformed payload: {type(exc).__name__}: {exc}"[:200], "magnitude": 1.0,
             "fix": 'send {"racks": [...]} with exactly the documented keys'}
        return {"score": 0.0, "feasible": False, "it_kw": 0.0, "upper_bound_kw": None,
                "components": {"density": 0.0, "resilience": 0.0, "hose": 0.0},
                "checks": [{"name": "payload", "passed": False, "hard": True, "violations": [v]}],
                "feedback": f"[payload] {v['detail']} -> fix: {v['fix']}"}


def categories(res: dict[str, Any]) -> list[str]:
    """Violated check names, most severe first; soft 'resilience' marks a feasible layout that a CDU loss breaks."""
    names = [c["name"] for c in res.get("checks", []) if not c["passed"]]
    return sorted(names, key=lambda n: SEVERITY.index(n) if n in SEVERITY else len(SEVERITY))


def violations(res: dict[str, Any], limit: int = 8) -> list[dict[str, str]]:
    rows = [{"check": c["name"], "detail": v["detail"], "fix": v["fix"]}
            for c in res.get("checks", []) if not c["passed"] for v in c["violations"]]
    rows.sort(key=lambda r: SEVERITY.index(r["check"]) if r["check"] in SEVERITY else len(SEVERITY))
    return rows[:limit]


def report_text(res: dict[str, Any], max_lines: int = 15, max_chars: int = 2000) -> str:
    """Scorer verdict for prompts: score and components, then each violation's detail and fix hint."""
    comps = ", ".join(f"{k} {v:.2f}" for k, v in (res.get("components") or {}).items())
    head = (f"feasible, score {res['score']:.2f}/100 ({comps}), {res.get('it_kw', 0):.0f} kW of "
            f"{res.get('upper_bound_kw') or 0:.0f} kW bound" if res["feasible"] else "infeasible (score 0)")
    lines = [head] + [f"[{v['check']}] {v['detail']} -> fix: {v['fix']}" for v in violations(res, max_lines)]
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[:max_chars - 12] + "\n(truncated)"


def capacity_calc(hall: dict[str, Any]) -> str:
    """Tool: deterministic power, cooling and free-length budget, so no role has to do this arithmetic."""
    kw = LIQUID_KW["GB200-NVL72"]
    lines = ["Capacity budget:"]
    for row in sorted({b["row"] for b in _objs(hall, "busway")}):
        pair = [b for b in _objs(hall, "busway") if b["row"] == row]
        budget = min(b["capacity_kw"] * b.get("derate", 1.0) for b in pair)
        segs = halls.row_free_segments(hall, row)
        slots = sum(math.floor((s1 - s0) / RACK_W + 1e-6) for s0, s1 in segs)
        runs = ", ".join(f"{s0:.2f}..{s1:.2f}" for s0, s1 in segs)
        lines.append(f"- row {row} (row_y {sum(b['y'] for b in pair) / len(pair):.2f}): 2N budget {budget:.0f} kW -> "
                     f"max {math.floor(budget / kw + 1e-6)} GB200 or {math.floor(budget / LIQUID_KW['GB300-NVL72'] + 1e-6)}"
                     f" GB300; free x runs {runs} -> {slots} rack slots")
    caps = [c["capacity_kw"] for c in _objs(hall, "cdu")]
    for c in _objs(hall, "cdu"):
        lines.append(f"- {c['id']}: {c['capacity_kw']} kW -> max {int(c['capacity_kw'] // kw)} GB200, hose {c['hose_m']} m")
    lines.append(f"- single-CDU-failure headroom: keep total liquid kW <= {sum(caps) - max(caps)} kW "
                 "(all CDUs minus the largest)")
    lines.append(f"- air cooling {hall['air_cooling_kw']} kW -> max {int(hall['air_cooling_kw'] // AIR_KW['HGX-B200'])} "
                 "HGX-B200")
    lines.append(f"Density upper bound: {scoring.upper_bound_kw(hall):.0f} kW")
    return "\n".join(lines)


def greedy_seed(task: Task) -> str:
    """Tool: the greedy solver's verified layout as a starting point the team must improve."""
    p = task.payload
    return (f"Starting layout from a deterministic greedy solver (feasible={p['greedy_feasible']}, score "
            f"{p['greedy_score']:.2f}/100). Keep it feasible and raise the score (density, single-CDU-failure "
            f"resilience, shorter hoses):\n" + json.dumps(p["greedy_layout"], separators=(",", ":")))


def lesson_proposals(failures: list[str]) -> list[dict[str, Any]]:
    """One generic rule per failure category, for the brain; the Evolver attaches run evidence."""
    out, seen = [], set()
    for f in failures:
        category, _, detail = f.partition(":")
        if category == "bad_json":
            roles, text = (detail or "Layout",), "Reply with exactly one JSON object in the required format, no prose."
        elif category in LESSONS:
            roles, text = LESSONS[category]
        else:
            continue
        if (category, roles) not in seen:
            seen.add((category, roles))
            out.append({"kind": "lesson", "roles": list(roles), "text": text})
    return out


def datahall_v0() -> TeamVersion:
    return TeamVersion(
        version=0, parent=None, prompts=dict(V0_PROMPTS), playbooks={},
        config={"roles_enabled": ["Planner", "Layout", "Checker"], "models": {}, "default_model": NANO,
                "temperature": 0.2, "max_tokens": 8192, "max_iters": 3, "stop_when_feasible": True,
                "brain_budget": 2000, "tool_catalog": list(TOOL_CATALOG),
                "tools": {"verifier_feedback": {"enabled": True}, "capacity_calc": {"enabled": False},
                          "greedy_seed": {"enabled": False}}},
        changelog=["v0: initial data-hall team (Planner, Layout, Checker)"])


@dataclass
class _Attempt:
    index: int
    res: dict[str, Any] | None  # None when the reply was not a usable payload
    failures: list[str]
    payload_json: str = ""


class _Run:
    """State for one team run on one task: usage, brain reads, failures."""

    def __init__(self, version: TeamVersion, brain: SharedBrain, task: Task, llm: LLM):
        self.version, self.brain, self.task, self.llm = version, brain, task, llm
        self.cfg = version.config
        self.hall = task.payload["hall"]
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.cost_usd = 0.0
        self.contexts: dict[str, str] = {}
        self.brain_ids: set[str] = set()
        self.failures: list[str] = []
        self.errors: list[str] = []

    def tool(self, name: str) -> dict[str, Any] | None:
        """The tool's settings when enabled in config["tools"], else None."""
        setting = (self.cfg.get("tools") or {}).get(name)
        return setting if isinstance(setting, dict) and setting.get("enabled") else None

    def problem(self) -> str:
        return f"{hall_summary(self.hall, self.task.payload['upper_bound_kw'])}\n\n{RULES}"

    def system(self, role: str) -> str:
        """Role prompt, playbook, what this role read from the brain, then the fixed output contract."""
        if role not in self.contexts:
            context, ids = self.brain.read(role, self.task.text, int(self.cfg.get("brain_budget", 2000)))
            self.contexts[role] = context
            self.brain_ids.update(ids)
        steps = self.version.playbooks.get(role, [])
        parts = [self.version.prompts.get(role, ""), "Playbook:\n" + "\n".join(f"- {s}" for s in steps) if steps else "",
                 self.contexts[role], CONTRACTS.get(role, ADVISOR_CONTRACT)]
        return "\n\n".join(p for p in parts if p)

    def ask(self, role: str, messages: list[dict[str, str]]) -> Reply | None:
        """One JSON-mode call; a failed call ends that step but never the generation."""
        model = (self.cfg.get("models") or {}).get(role) or self.cfg.get("default_model", NANO)
        try:
            reply = self.llm.chat(model, messages, temperature=float(self.cfg.get("temperature", 0.2)),
                                  max_tokens=int(self.cfg.get("max_tokens", 8192)), json_mode=True, role=role)
        except Exception as exc:  # recorded in failures and notes; one dead call must not sink a 4-worker batch
            self.failures.append(f"llm_error:{role}")
            self.errors.append(f"{role}: {type(exc).__name__}: {str(exc)[:200]}")
            return None
        self.calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens
        self.cost_usd += reply.cost_usd
        return reply

    def advise(self, role: str) -> str:
        reply = self.ask(role, [{"role": "system", "content": self.system(role)},
                                {"role": "user", "content": self.problem()}])
        if reply is None:
            return ""
        try:
            text = json.dumps(extract_json(reply.text), separators=(",", ":"))
        except ValueError:
            self.failures.append(f"bad_json:{role}")
            text = reply.text.strip()
        return f"{role}: {text[:800]}"

    def brief(self, plans: list[str]) -> str:
        parts = [self.problem()]
        if any(plans):
            parts.append("Team notes:\n" + "\n".join(p for p in plans if p))
        if self.tool("capacity_calc"):
            parts.append(capacity_calc(self.hall))
        if self.tool("greedy_seed"):
            parts.append(greedy_seed(self.task))
        parts.append("Produce the complete layout.")
        return "\n\n".join(parts)

    def judge(self, index: int, text: str) -> _Attempt:
        try:
            parsed = extract_json(text)
        except ValueError:
            return _Attempt(index, None, ["bad_json"])
        if isinstance(parsed, dict) and "racks" not in parsed and isinstance(parsed.get("payload"), dict):
            parsed = parsed["payload"]
        racks = parsed if isinstance(parsed, list) else parsed.get("racks")
        if not isinstance(racks, list):
            return _Attempt(index, None, ["bad_json"])
        payload = {"racks": racks}
        res = score_payload(self.hall, payload)
        return _Attempt(index, res, categories(res), json.dumps(payload, separators=(",", ":"))[:6000])

    def check(self, attempt: _Attempt, report: str) -> str:
        reply = self.ask("Checker", [{"role": "system", "content": self.system("Checker")},
                                     {"role": "user", "content": f"Layout attempt:\n{attempt.payload_json[:3000]}\n\n"
                                                                 f"Verifier feedback:\n{report}"}])
        if reply is None:
            return ""
        try:
            fix = extract_json(reply.text)
            text = fix.get("fix", "") if isinstance(fix, dict) else json.dumps(fix)
        except ValueError:
            self.failures.append("bad_json:Checker")
            text = reply.text.strip()
        return str(text)[:600]

    def feedback(self, attempt: _Attempt) -> list[dict[str, str]]:
        """What Layout sees before its next attempt: its last payload, then the verdict and any fix."""
        if attempt.res is None:
            return [{"role": "user", "content": "Your last reply was not a JSON object with a racks list. " + FORMAT}]
        res = attempt.res
        verdict = (f"feasible, score {res['score']:.2f}/100" if res["feasible"]
                   else "infeasible (score 0), violated: " + ", ".join(attempt.failures))
        parts = [f"Attempt {attempt.index + 1}: {verdict}."]
        report = report_text(res)
        if self.tool("verifier_feedback"):
            parts.append("Verifier feedback:\n" + report)
        if "Checker" in self.version.roles():
            fix = self.check(attempt, report)
            if fix:
                parts.append(f"Checker: {fix}")
        parts.append("Raise the score while keeping every hard rule; return the complete layout." if res["feasible"]
                     else "Return the complete corrected layout.")
        return [{"role": "assistant", "content": attempt.payload_json}, {"role": "user", "content": "\n".join(parts)}]

    def layout_loop(self, plans: list[str]) -> list[_Attempt]:
        max_iters = max(1, int(self.cfg.get("max_iters", 3)))
        head = [{"role": "system", "content": self.system("Layout")}, {"role": "user", "content": self.brief(plans)}]
        tail: list[dict[str, str]] = []
        attempts: list[_Attempt] = []
        for i in range(max_iters):
            reply = self.ask("Layout", head + tail)
            if reply is None:
                break
            attempt = self.judge(i, reply.text)
            attempts.append(attempt)
            self.failures.extend(attempt.failures)
            if attempt.res and attempt.res["feasible"] and self.cfg.get("stop_when_feasible", True):
                break
            if i + 1 < max_iters:
                tail = self.feedback(attempt)
        return attempts


def datahall_run_team(version: TeamVersion, brain: SharedBrain, task: Task, llm: LLM) -> RunResult:
    """Advisors (Planner and any added role) plan once; Layout then tries up to max_iters times with scorer feedback."""
    start = time.perf_counter()
    run = _Run(version, brain, task, llm)
    roles = version.roles()
    plans = [run.advise(role) for role in roles if role not in ("Layout", "Checker")]
    attempts = run.layout_loop(plans) if "Layout" in roles else []
    if "Layout" not in roles:
        run.failures.append("no_layout_role")
    scored = [a for a in attempts if a.res]
    best = max(scored, key=lambda a: (a.res["score"], a.res["feasible"]), default=None)
    failures = list(dict.fromkeys(run.failures))
    notes = {"proposals": lesson_proposals(failures),
             "violations": violations(best.res) if best else [],
             "attempts": [{"score": a.res["score"] if a.res else 0.0, "feasible": bool(a.res and a.res["feasible"]),
                           "failures": a.failures} for a in attempts],
             "best_attempt": best.index if best else None, "greedy_score": task.payload["greedy_score"],
             "errors": run.errors}
    return RunResult(task_id=task.id, version=version.version,
                     score=round(best.res["score"] / 100, 4) if best else 0.0,
                     passed=bool(best and best.res["feasible"]), claimed_done=bool(scored),
                     iterations=len(attempts), calls=run.calls, prompt_tokens=run.prompt_tokens,
                     completion_tokens=run.completion_tokens, cost_usd=round(run.cost_usd, 8),
                     wall_s=round(time.perf_counter() - start, 3), failures=failures,
                     brain_ids_read=sorted(run.brain_ids), notes=json.dumps(notes))
