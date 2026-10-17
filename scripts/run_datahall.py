"""Evolve the data-hall team: Nemotron roles lay out 2N AI halls, Rikin's DoneTwin scorer judges, the team edits itself.

Usage (repo venv):
  .venv/bin/python scripts/run_datahall.py --offline            # scripted LLM, proves plumbing in seconds
  .venv/bin/python scripts/run_datahall.py --live --one         # one train hall with v0, to budget a run
  .venv/bin/python scripts/run_datahall.py --live [--generations 2] [--out DIR]
Every run hill-climbs from the greedy layout, so scores never fall below greedy; the gain above it is the team's.
Exit codes: 0 done, 2 no Crusoe API key configured.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team.brain import SharedBrain  # noqa: E402
from team.crusoe import NANO, SUPER, CrusoeLLM, MissingKeyError  # noqa: E402
from team.datahall import SPLIT_SIZES, datahall_run_team, datahall_v0, load_tasks  # noqa: E402
from team.evolve import Evolver, Gate, TeamVersion, markdown_table  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.sa_library import add_library_args, configure_from_args  # noqa: E402
from team.types import LLM, Task  # noqa: E402

CANDIDATES_PER_GEN = 3  # Reviewer max_mutations
# Measured on 4 train halls (Layout varied, other roles Nano with reasoning off): only Nano with reasoning on and a
# 32000-token budget beat greedy (+2.66 points); at 16000 most replies were cut off, and Super with reasoning ran
# ~340 s per call and was cut off. The scorer rewarded one-at-a-time GB200 -> GB300 swaps and CDU reassignments.
DATAHALL_REVIEWER_NOTE = (
    " Data-hall evidence from earlier measurements: Layout with thinking on and max_tokens 32000 is the only setting "
    "that beat greedy; at 16000 most reasoning replies were cut off (truncated); Super with thinking on stalled "
    "(~340 s per call, cut off). Feasible gains came from swapping single GB200 racks to GB300 and reassigning CDUs, "
    "not from re-laying out rows. Super is currently throttled by Crusoe (HTTP 429 on every call); do not assign it "
    "to a role.")
_HALL_NAME = re.compile(r"HALL (\S+):")


class ReviewerBudget:
    """Gives the Reviewer's call its own token budget and reasoning switch; every other call passes through."""

    def __init__(self, llm: Any, max_tokens: int, thinking: bool | None) -> None:
        self.llm, self.max_tokens, self.thinking = llm, max_tokens, thinking

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        if kwargs.get("role") == "reviewer":
            kwargs["max_tokens"] = self.max_tokens
            if self.thinking is not None:
                kwargs["thinking"] = self.thinking
        return self.llm.chat(model, messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)


def offline_llm(tasks: list[Task]) -> ScriptedLLM:
    """A deterministic stand-in: Layout first sends an infeasible layout, and with a brain lesson or on its retry
    it returns the incumbent (no gain, no loss).

    It finds the hall by name in the prompt and answers from the greedy layout, so every score printed
    offline comes from the real DoneTwin scorer; only the replies are canned.
    """
    by_name = {t.id: t for t in tasks}
    reviews = iter([
        {"mutations": [
            {"type": "config_set", "key": "thinking", "value": {"Layout": True},
             "rationale": "Layout proposals are infeasible; let it reason",
             "claim": {"dev_gain": 0.05, "cost_change_pct": 400.0, "why": "reasoning fixes 2N and CDU overloads"}},
            {"type": "config_set", "key": "max_iters", "value": 1, "rationale": "cheaper runs",
             "claim": {"dev_gain": 0.0, "cost_change_pct": -50.0, "why": "half the Layout calls"}},
            {"type": "tool_set", "tool": "hammer", "enabled": True, "rationale": "invalid on purpose",
             "claim": {"dev_gain": 0.1, "cost_change_pct": 0.0, "why": "unknown tool, must be dropped"}}]},
        {"mutations": [
            {"type": "prompt_edit", "role": "Checker", "text": "You are the Checker. Be brief.", "rationale": "shorter",
             "claim": {"dev_gain": 0.0, "cost_change_pct": -10.0, "why": "fewer Checker tokens"}},
            {"type": "tool_set", "tool": "capacity_calc", "enabled": True, "rationale": "power_2n overloads",
             "claim": {"dev_gain": 0.02, "cost_change_pct": 5.0, "why": "budgets stop busway overloads"}}]},
        {"mutations": [{"type": "playbook_remove", "role": "Layout", "index": 9, "rationale": "invalid index",
                        "claim": {"dev_gain": 0.0, "cost_change_pct": 0.0, "why": "invalid on purpose"}}]},
    ])

    def respond(role: str, messages: list[dict[str, str]]) -> str:
        if role == "reviewer":
            return json.dumps(next(reviews, {"mutations": []}))
        if role == "Checker":
            return json.dumps({"fix": "move or drop the racks the verifier names"})
        found = _HALL_NAME.search(messages[1]["content"])
        task = by_name.get(found.group(1)) if found else None
        if task is None:
            return json.dumps({"advice": "fill every row to its 2N budget"})
        racks = task.payload["greedy_layout"]["racks"]
        if role == "Planner":
            return json.dumps({"row_racks": {}, "cdu_racks": {}, "total_racks": len(racks), "notes": "offline"})
        if role != "Layout":
            return json.dumps({"advice": "fill every row to its 2N budget"})
        if "0.6 m apart" in messages[0]["content"] or len(messages) > 2:
            return json.dumps({"racks": racks})
        return json.dumps({"racks": racks + [{**racks[0], "id": "R_DUP"}]})

    return ScriptedLLM(respond)


def estimate_calls(version: TeamVersion, n_train: int, n_dev: int, n_heldout: int, generations: int) -> dict[str, int]:
    """Worker calls per run range from advisors + max_iters (every attempt accepted) to that plus a Checker
    call after each rejected attempt but the last."""
    roles = version.roles()
    advisors = sum(r not in ("Layout", "Checker") for r in roles)
    iters = int(version.config.get("max_iters", 2))
    per_run_max = advisors + iters + (iters - 1 if "Checker" in roles else 0)
    runs = generations * (n_train + n_dev + CANDIDATES_PER_GEN * n_dev) + 3 * n_heldout
    return {"runs": runs, "worker_min": runs * (advisors + iters), "worker_max": runs * per_run_max,
            "reviewer": generations}


def greedy_row(tasks: list[Task]) -> dict[str, Any]:
    """The no-LLM baseline in the same shape as Evolver.final_report rows."""
    n = max(1, len(tasks))
    return {"config": "greedy baseline (no LLM)", "version": None, "n": len(tasks),
            "mean_score": round(sum(t.payload["greedy_score"] for t in tasks) / 100 / n, 4),
            "pass_rate": round(sum(bool(t.payload["greedy_feasible"]) for t in tasks) / n, 4),
            "mean_iterations": 0.0, "tokens": 0, "cost_usd": 0.0}


def run_one(llm: LLM, task: Task, v0: TeamVersion) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        r = datahall_run_team(v0, SharedBrain(tmp), task, llm)
    notes = json.loads(r.notes)
    print(f"task {r.task_id} ({task.text})")
    print(f"score {r.score * 100:.2f}/100 (greedy {task.payload['greedy_score']:.2f}, gain "
          f"{notes['gain_over_greedy']:+.2f}) feasible={r.passed} "
          f"iterations={r.iterations} calls={r.calls} tokens={r.prompt_tokens}+{r.completion_tokens} "
          f"cost=${r.cost_usd:.5f} wall={r.wall_s:.1f}s")
    print(f"failures: {r.failures}")
    print(f"attempts: {json.dumps(notes['attempts'])}")
    for v in notes["violations"][:5]:
        print(f"  [{v['check']}] {v['detail']} -> fix: {v['fix']}")
    if notes["errors"]:
        print(f"errors: {notes['errors']}")


def evolve(llm: LLM, reviewer_model: str, out: Path, generations: int, seed: int, workers: int,
           v0: TeamVersion, splits: tuple[list[Task], list[Task], list[Task]], min_gain: float = 0.05) -> None:
    train, dev, heldout = splits
    evolver = Evolver(datahall_run_team, llm, reviewer_model, SharedBrain(out / "brain"), out / "versions",
                      max_workers=workers, seed=seed, initial=v0, reports_dir=out / "reports",
                      gate=Gate(min_gain=min_gain))
    evolver.reviewer.system += DATAHALL_REVIEWER_NOTE
    t0 = time.perf_counter()
    final = evolver.run(generations, train, dev, heldout)
    for h in evolver.history:
        print(f"gen {h['generation']}: v{h['parent']} -> v{h['version']} accepted={h['accepted']} "
              f"train={h['train_mean']:.3f} dev={h['dev_mean']:.3f} brain_active={h['brain']['active']}", flush=True)
    print("\nchangelog:\n  " + "\n  ".join(final.changelog))
    rows = evolver.final_report(heldout) + [greedy_row(heldout)]
    table = markdown_table(rows)
    base = rows[-1]["mean_score"]
    vs = "\n".join(f"- {r['config']}: {(r['mean_score'] - base) * 100:+.2f} points vs greedy" for r in rows[:-1])
    print(f"\nHeld-out ({len(heldout)} unseen halls), score = DoneTwin score / 100, pass = feasible:\n")
    print(table)
    print(vs)
    (out / "reports" / "vs_greedy.md").write_text(table + "\n" + vs + "\n")
    print(f"\nwall {time.perf_counter() - t0:.1f}s; artifacts: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="scripted LLM, no network")
    mode.add_argument("--live", action="store_true", help="Crusoe Managed Inference")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--one", action="store_true", help="run one train hall with v0 and print its bill")
    ap.add_argument("--out", type=Path, default=None, help="default runs/datahall-<timestamp>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6, help="Evolver threads; the client still caps CRUSOE_RPM")
    ap.add_argument("--max-iters", type=int, default=2, help="v0 hill-climb attempts per hall")
    ap.add_argument("--max-tokens", type=int, default=6000, help="per call; Nemotron reasoning counts against it")
    ap.add_argument("--split", type=int, nargs=3, default=list(SPLIT_SIZES), metavar=("TRAIN", "DEV", "HELDOUT"),
                    help="halls per split (at most 2 large each)")
    ap.add_argument("--worker-model", default=NANO)
    ap.add_argument("--reviewer-model", default=SUPER)
    ap.add_argument("--min-gain", type=float, default=0.05,
                    help="Gate: dev mean score gain needed to accept (score = DoneTwin/100, so 0.02 = 2 points)")
    ap.add_argument("--reviewer-max-tokens", type=int, default=None,
                    help="Reviewer call budget (client default 2048); a reasoning model can spend it all thinking")
    ap.add_argument("--reviewer-thinking", choices=("on", "off"), default=None,
                    help="force the Reviewer's reasoning switch; default sends nothing (server default)")
    ap.add_argument("--timeout-s", type=float, default=600.0,
                    help="per-request timeout; reasoning-on Layout calls run 100-340 s, so 120 s silently kills them")
    add_library_args(ap)
    args = ap.parse_args()
    configure_from_args(args, "datahall")

    v0 = datahall_v0()
    v0.config.update(max_iters=args.max_iters, max_tokens=args.max_tokens, default_model=args.worker_model)
    if args.offline:
        llm: Any = None
    else:
        try:
            llm = CrusoeLLM(timeout_s=args.timeout_s)
        except MissingKeyError as exc:
            print(f"No Crusoe API key configured: {exc}", file=sys.stderr)
            return 2
        if args.reviewer_max_tokens or args.reviewer_thinking:
            llm = ReviewerBudget(llm, args.reviewer_max_tokens or 2048,
                                 None if args.reviewer_thinking is None else args.reviewer_thinking == "on")
    splits = load_tasks(args.seed, tuple(args.split))
    train, dev, heldout = splits
    if llm is None:
        llm = offline_llm(train + dev + heldout)
    if args.one:
        run_one(llm, train[0], v0)
        if args.live:
            print(json.dumps(llm.ledger.summary()["by_role"], indent=1))
        return 0

    est = estimate_calls(v0, len(train), len(dev), len(heldout), args.generations)
    rpm = getattr(llm, "rpm", None)
    print(f"Estimated LLM calls: ~{est['runs']} team runs -> {est['worker_min']}-{est['worker_max']} worker calls "
          f"({args.worker_model}) + {est['reviewer']} reviewer calls ({args.reviewer_model})"
          + (f"; at {rpm:.0f} RPM per model that is >= {est['worker_min'] / rpm:.0f}-{est['worker_max'] / rpm:.0f} min"
             if rpm else ""), flush=True)
    print(f"greedy baseline: train {mean(t.payload['greedy_score'] for t in train):.2f}, "
          f"dev {mean(t.payload['greedy_score'] for t in dev):.2f}, "
          f"held-out {mean(t.payload['greedy_score'] for t in heldout):.2f} (of 100)", flush=True)
    out = args.out or Path("runs") / f"datahall-{time.strftime('%Y%m%d-%H%M%S')}"
    evolve(llm, args.reviewer_model, out, args.generations, args.seed, args.workers, v0, splits, args.min_gain)
    if args.live:
        usage = llm.ledger.summary()
        (out / "reports" / "usage.json").write_text(json.dumps(usage, indent=1))
        print(f"usage: {json.dumps(usage['total'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
