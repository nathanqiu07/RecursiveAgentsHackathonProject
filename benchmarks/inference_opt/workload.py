"""Seeded workload of data-center operations questions whose answers are computed exactly by code.

Every question is drawn from one of Rikin's DoneTwin halls (donetwin/donetwin/halls.py generate(seed, tier)) with
rack loads from donetwin's RACK_CATALOG, so the facts are the same ones the data-hall layer optimizes against.
A task is one hall with QUESTIONS_PER_TYPE questions of each type; the checker grades a numeric answer as correct
when it is within the type's stated tolerance of the exact answer. No LLM judges anything.

Question types, easy to hard:
  cdu_headroom  CDU capacity minus the liquid racks assigned to it (air racks and other CDUs listed as distractors)
  busway_left   2N power budget left on a row: one busway's capacity * derate minus the row's rack load
  hose_slack    hose_m minus the Manhattan run from a rack center to its CDU's port (negative = out of reach)
  rack_fit      how many more racks of a model fit a row's remaining 2N budget (floor)
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DONETWIN = ROOT / "donetwin"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(DONETWIN) not in sys.path:
    # Appended, not inserted first: donetwin/ also holds a top-level run.py, server.py and a tests package.
    sys.path.append(str(DONETWIN))

from donetwin import halls  # noqa: E402
from donetwin.scene import RACK_CATALOG  # noqa: E402
from team.types import Task  # noqa: E402

QTYPES = ("cdu_headroom", "busway_left", "hose_slack", "rack_fit")
TOLERANCE = {"cdu_headroom": 0.5, "busway_left": 0.5, "hose_slack": 0.05, "rack_fit": 0.0}
UNITS = {"cdu_headroom": "kW", "busway_left": "kW", "hose_slack": "m", "rack_fit": "racks"}
QUESTIONS_PER_TYPE = 3
LIQUID = ("GB200-NVL72", "GB300-NVL72")
AIR = ("HGX-B200",)
KW = {m: s["kw"] for m, s in RACK_CATALOG.items()}
# Disjoint seed ranges; held-out uses DoneTwin's own held-out seeds, which are never used for tuning.
SPLIT_SEEDS = {"train": 100, "dev": 200, "heldout": halls.heldout_seeds()[0]}
SPLIT_SIZES = (4, 3, 4)
TIERS = ("small", "medium")

REFERENCE = (
    "Reference facts. Rack IT load: " + ", ".join(f"{m} = {kw} kW" for m, kw in KW.items()) + ". "
    "GB200-NVL72 and GB300-NVL72 are liquid-cooled and draw on their assigned CDU; HGX-B200 is air-cooled and uses no CDU. "
    "2N power: every rack in a row takes feed A from busway BW_<row>1 and feed B from BW_<row>2, and either feed alone "
    "must carry the whole row, so a row's 2N budget is ONE busway's capacity_kw * derate (not the sum of A and B). "
    "A CDU's headroom is its capacity_kw minus the kW of the liquid racks assigned to it. "
    "Hose runs are Manhattan (|dx| + |dy|) from the rack center to the CDU port; hose slack = hose_m minus that run."
)


def _objs(hall: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [o for o in hall["objects"] if o["kind"] == kind]


def _rack_list(rng: random.Random, models: tuple[str, ...], budget: float, n_max: int) -> list[str]:
    """Random racks whose total load stays within `budget`."""
    out: list[str] = []
    load = 0.0
    for _ in range(rng.randint(1, n_max)):
        m = rng.choice(models)
        if load + KW[m] > budget:
            break
        out.append(m)
        load += KW[m]
    return out


def _named(racks: list[str], start: int = 1) -> str:
    return ", ".join(f"R{i} {m}" for i, m in enumerate(racks, start))


def q_busway_left(rng: random.Random, hall: dict[str, Any]) -> tuple[str, float]:
    bw = rng.choice([b for b in _objs(hall, "busway") if b["feed"] == "A"])
    row, cap, derate = bw["row"], bw["capacity_kw"], bw["derate"]
    budget = cap * derate
    racks = _rack_list(rng, LIQUID + AIR + AIR, budget, 6)
    text = (f"Row {row} is fed by busways BW_{row}1 (feed A) and BW_{row}2 (feed B), each capacity_kw={cap}, "
            f"derate={derate}. Racks already in row {row}: {_named(racks)}. "
            f"How many kW of 2N power budget are left on row {row}?")
    return text, round(budget - sum(KW[m] for m in racks), 3)


def q_rack_fit(rng: random.Random, hall: dict[str, Any]) -> tuple[str, float]:
    bw = rng.choice([b for b in _objs(hall, "busway") if b["feed"] == "A"])
    row, cap, derate = bw["row"], bw["capacity_kw"], bw["derate"]
    budget = cap * derate
    racks = _rack_list(rng, LIQUID + AIR, budget * rng.uniform(0.2, 0.7), 4)
    model = rng.choice(LIQUID + AIR)
    left = budget - sum(KW[m] for m in racks)
    text = (f"Row {row} is fed by busways BW_{row}1 and BW_{row}2, each capacity_kw={cap}, derate={derate}. "
            f"Racks already in row {row}: {_named(racks)}. How many more {model} racks fit in row {row} "
            f"without exceeding its 2N power budget? (whole racks only)")
    return text, float(math.floor(left / KW[model] + 1e-9))


def q_cdu_headroom(rng: random.Random, hall: dict[str, Any]) -> tuple[str, float]:
    cdus = _objs(hall, "cdu")
    target = rng.choice(cdus)
    mine = _rack_list(rng, LIQUID, target["capacity_kw"], 5)
    others = [c["id"] for c in cdus if c["id"] != target["id"]]
    entries = [(m, target["id"]) for m in mine]
    entries += [(rng.choice(AIR), "none") for _ in range(rng.randint(1, 3))]
    entries += [(rng.choice(LIQUID), rng.choice(others)) for _ in range(rng.randint(1, 3)) if others]
    rng.shuffle(entries)
    listing = ", ".join(f"R{i} {m} -> {c}" for i, (m, c) in enumerate(entries, 1))
    caps = ", ".join(f"{c['id']} capacity_kw={c['capacity_kw']}" for c in cdus)
    text = (f"CDUs: {caps}. Racks and their CDU assignment: {listing}. "
            f"How many kW of cooling headroom are left on {target['id']}?")
    return text, round(target["capacity_kw"] - sum(KW[m] for m in mine), 3)


def q_hose_slack(rng: random.Random, hall: dict[str, Any]) -> tuple[str, float]:
    cdu = rng.choice(_objs(hall, "cdu"))
    bw = rng.choice([b for b in _objs(hall, "busway") if b["feed"] == "A"])
    ry = round(bw["y"] + 0.25, 2)
    rx = round(rng.uniform(halls.WALK + 0.5, hall["floor"]["w"] - 1.0), 2)
    dx, dy = cdu["points"]["port"]
    px, py = round(cdu["x"] + dx, 2), round(cdu["y"] + dy, 2)
    text = (f"{cdu['id']} is centered at x={cdu['x']}, y={cdu['y']} m; its hose port is at offset ({dx}, {dy}) m "
            f"from that center; hose_m={cdu['hose_m']}. A rack in row {bw['row']} is centered at x={rx}, y={ry} m. "
            f"What is the hose slack in metres (hose_m minus the Manhattan run from the rack center to the port; "
            f"negative if out of reach)?")
    return text, round(cdu["hose_m"] - (abs(rx - px) + abs(ry - py)), 3)


MAKERS = {"cdu_headroom": q_cdu_headroom, "busway_left": q_busway_left, "hose_slack": q_hose_slack,
          "rack_fit": q_rack_fit}


def make_questions(seed: int, tier: str, per_type: int = QUESTIONS_PER_TYPE) -> list[dict[str, Any]]:
    """Deterministic per (seed, tier, per_type): the hall from DoneTwin, the questions from Random(f"inference_opt:...")."""
    hall = halls.generate(seed, tier)
    rng = random.Random(f"inference_opt:{tier}:{seed}")
    qs = []
    for qtype in QTYPES:
        for _ in range(per_type):
            text, answer = MAKERS[qtype](rng, hall)
            qs.append({"qtype": qtype, "text": text, "answer": answer, "tol": TOLERANCE[qtype],
                       "unit": UNITS[qtype]})
    rng.shuffle(qs)
    for i, q in enumerate(qs, 1):
        q["id"] = f"q{i}"
    return qs


def make_task(seed: int, tier: str, split: str, per_type: int = QUESTIONS_PER_TYPE) -> Task:
    hall_name = f"{tier}-{seed}"
    return Task(id=f"inf-{hall_name}", split=split,
                payload={"seed": seed, "tier": tier, "hall": hall_name,
                         "questions": make_questions(seed, tier, per_type)},
                text=f"data-center ops Q&A on DoneTwin hall {hall_name}: " + ", ".join(QTYPES))


def load_tasks(sizes: tuple[int, int, int] = SPLIT_SIZES, per_type: int = QUESTIONS_PER_TYPE
               ) -> tuple[list[Task], list[Task], list[Task]]:
    """Train, dev and held-out tasks from disjoint seed ranges, tiers alternating small/medium."""
    out = []
    for split, n in zip(("train", "dev", "heldout"), sizes):
        base = SPLIT_SEEDS[split]
        out.append([make_task(base + i, TIERS[i % len(TIERS)], split, per_type) for i in range(n)])
    return out[0], out[1], out[2]


def is_correct(q: dict[str, Any], value: Any) -> bool:
    """Exact check in code: a finite number within the question's tolerance of the computed answer."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    return abs(float(value) - q["answer"]) <= q["tol"] + 1e-9


def coerce(value: Any) -> float | None:
    """Model replies carry numbers as numbers or numeric strings ("132", "132 kW"); anything else is None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip().replace(",", "").split(" ")[0]
        try:
            return float(token)
        except ValueError:
            return None
    return None


def grade(questions: list[dict[str, Any]], answers: dict[str, Any]) -> dict[str, Any]:
    """Accuracy over every question; a missing or non-numeric answer counts as wrong."""
    per_q = {q["id"]: is_correct(q, coerce(answers.get(q["id"]))) for q in questions}
    n = max(1, len(questions))
    return {"accuracy": sum(per_q.values()) / n, "correct": sum(per_q.values()), "n": len(questions), "per_q": per_q}
