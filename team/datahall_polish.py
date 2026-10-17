"""Verifier-guided polish for a data-hall layout: a classical, no-LLM local search over single-rack moves.

Ported from Rikin's `local_search` in donetwin/donetwin/hill_climb.py on origin/spec/donetwin, commit f8d337f
(read with `git show`, not merged: that branch changes hill_climb.FORMAT, which team/datahall.py parses).
Only local_search and its helpers (_objs, _rows, _slots, _neighbours) are copied; they depend on main's
donetwin.score (score, COOLING) and donetwin.halls (row_free_segments), which are byte-identical on that branch.
Changes from the original: imports at module level, and local_search also returns the verifier calls it used.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

DONETWIN = Path(__file__).resolve().parent.parent / "donetwin"
if str(DONETWIN) not in sys.path:
    # Appended, not inserted first: donetwin/ also holds top-level run.py, server.py and a tests package.
    sys.path.append(str(DONETWIN))

from donetwin import halls as hall_gen  # noqa: E402
from donetwin import score as scoring  # noqa: E402
from donetwin.score import COOLING  # noqa: E402

MAX_CHECKS = 1500  # Rikin's cap on verifier calls (about 8 ms each); also the upper bound of the polish_checks knob


def _objs(hall: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [o for o in hall.get("objects", []) if o.get("kind") == kind]


def _rows(hall: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for b in _objs(hall, "busway"):
        key = b.get("row") or b["id"]
        rows.setdefault(str(key), []).append(b)
    return rows


def _slots(hall: dict[str, Any], racks: list[dict[str, Any]]) -> list[tuple]:
    """Free rack positions at the ends of each row's runs (or the start of an empty row's free strip)."""
    taken = {(round(float(r["x"]), 2), round(float(r["y"]), 2)) for r in racks
             if isinstance(r.get("x"), (int, float)) and isinstance(r.get("y"), (int, float))}
    out = []
    for row, bws in sorted(_rows(hall).items()):
        a = next((b for b in bws if b.get("feed") == "A"), None)
        b = next((b for b in bws if b.get("feed") == "B"), None)
        if not a or not b:
            continue
        y = round(a["y"] + 0.25, 3)
        mine = [r for r in racks if r.get("feed_a") in (a["id"], b["id"]) and isinstance(r.get("x"), (int, float))]
        if mine:
            xs = {round(r["x"] + d, 3) for r in mine for d in (-0.6, 0.6)}
            rots = [mine[0].get("rot", 0)]
        else:
            try:
                xs = {round(x0 + 0.31, 3) for x0, _ in hall_gen.row_free_segments(hall, row)}
            except Exception:
                xs = set()
            rots = [0, 180]
        for x in sorted(xs):
            if (round(x, 2), round(y, 2)) not in taken:
                out += [(x, y, rot, a["id"], b["id"]) for rot in rots]
    return out


def _neighbours(hall: dict[str, Any], racks: list[dict[str, Any]], cdus: list[str]) -> Iterator[tuple[dict, str]]:
    """Single-rack moves, most valuable first: GB300 upgrades, adds, relocations, CDU reassignments."""

    def with_(i, **kw):
        return {"racks": [dict(r, **kw) if j == i else r for j, r in enumerate(racks)]}
    for i, r in enumerate(racks):
        if r.get("model") == "GB200-NVL72":
            for c in [r.get("cdu")] + [c for c in cdus if c != r.get("cdu")]:
                yield with_(i, model="GB300-NVL72", cdu=c), f"upgrade {r['id']} to GB300 on {c}"
    slots = _slots(hall, racks)
    ids = {r.get("id") for r in racks}
    nid = next(f"S{k}" for k in range(1, 10000) if f"S{k}" not in ids)
    for x, y, rot, fa, fb in slots:
        base = {"id": nid, "x": x, "y": y, "rot": rot, "feed_a": fa, "feed_b": fb}
        for model in ("GB300-NVL72", "GB200-NVL72"):
            for c in cdus:
                yield {"racks": racks + [dict(base, model=model, cdu=c)]}, f"add {model} at x={x:.1f} on {c}"
        yield {"racks": racks + [dict(base, model="HGX-B200", cdu=None)]}, f"add HGX-B200 at x={x:.1f}"
    for i, r in enumerate(racks):
        if COOLING.get(r.get("model")) == "liquid":
            for x, y, rot, fa, fb in slots:
                for c in cdus:
                    yield (with_(i, x=x, y=y, rot=rot, feed_a=fa, feed_b=fb, cdu=c),
                           f"move {r['id']} to x={x:.1f} y={y:.1f} on {c}")
    for i, r in enumerate(racks):
        if COOLING.get(r.get("model")) == "liquid":
            for c in cdus:
                if c != r.get("cdu"):
                    yield with_(i, cdu=c), f"reassign {r['id']} to {c}"


def local_search(hall: dict[str, Any], payload: dict[str, Any], max_evals: int = MAX_CHECKS
                 ) -> tuple[dict[str, Any], dict[str, Any], list[str], int]:
    """Verifier-guided first-improvement search over single-rack moves (no LLM).

    Each accepted move strictly raises the verified score, so the result is never worse than the input; an
    infeasible input is returned unchanged. max_evals caps verifier calls and keeps runs deterministic.
    Returns (payload, score result, accepted moves, verifier calls used)."""
    cur = {"racks": [dict(r) for r in payload.get("racks", []) if isinstance(r, dict)]}
    res = scoring.score(hall, cur)
    if not res.get("feasible"):
        return cur, res, [], 0
    cdus = [o["id"] for o in _objs(hall, "cdu")]
    moves, evals, improved = [], 0, True
    while improved and evals < max_evals:
        improved = False
        for cand, label in _neighbours(hall, cur["racks"], cdus):
            if evals >= max_evals:
                break
            evals += 1
            r = scoring.score(hall, cand)
            if r.get("feasible") and r["score"] > res["score"] + 1e-9:
                cur, res, improved = cand, r, True
                moves.append(label)
                break
    return cur, res, moves, evals
