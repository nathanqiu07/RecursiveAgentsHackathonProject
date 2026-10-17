"""Evolve the data-hall team: Nemotron roles lay out 2N AI halls, Rikin's DoneTwin scorer judges, the team edits itself.

Usage (repo venv):
  .venv/bin/python scripts/run_datahall.py --offline            # scripted LLM, proves plumbing in seconds
  .venv/bin/python scripts/run_datahall.py --live --one         # one train hall with v0, to budget a run
  .venv/bin/python scripts/run_datahall.py --live [--generations 3] [--out DIR]
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
from team.datahall import datahall_run_team, datahall_v0, load_tasks  # noqa: E402
from team.evolve import Evolver, TeamVersion, markdown_table  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.types import LLM, Task  # noqa: E402

CANDIDATES_PER_GEN = 3  # Reviewer max_mutations
_HALL_NAME = re.compile(r"HALL (\S+):")


def offline_llm(tasks: list[Task]) -> ScriptedLLM:
    """A deterministic stand-in whose Layout improves with a brain lesson and again with the greedy_seed tool.

    It finds the hall by name in the prompt and answers from the greedy layout, so every score printed
    offline comes from the real DoneTwin scorer; only the replies are canned.
    """
    by_name = {t.id: t for t in tasks}
    reviews = iter([
        {"mutations": [
            {"type": "tool_set", "tool": "greedy_seed", "enabled": True,
             "rationale": "layouts start from scratch and lose racks; start from the greedy solver"},
            {"type": "config_set", "key": "max_iters", "value": 1, "rationale": "cheaper runs"},
            {"type": "tool_set", "tool": "hammer", "enabled": True, "rationale": "invalid on purpose"}]},
        {"mutations": [
            {"type": "prompt_edit", "role": "Checker", "text": "You are the Checker. Be brief.", "rationale": "shorter"},
            {"type": "tool_set", "tool": "capacity_calc", "enabled": True, "rationale": "power_2n overloads"}]},
        {"mutations": [{"type": "playbook_remove", "role": "Layout", "index": 9, "rationale": "invalid index"}]},
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
        if "Starting layout from a deterministic greedy solver" in messages[1]["content"]:
            return json.dumps({"racks": racks})
        if "0.6 m apart" in messages[0]["content"]:
            return json.dumps({"racks": racks[:-2]})
        if len(messages) > 2:
            return json.dumps({"racks": racks[:-3]})
        return json.dumps({"racks": racks + [{**racks[0], "id": "R_DUP"}]})

    return ScriptedLLM(respond)


def estimate_calls(version: TeamVersion, n_train: int, n_dev: int, n_heldout: int, generations: int) -> dict[str, int]:
    """Worker calls per run range from advisors + 1 (feasible first try) to advisors + max_iters + checks."""
    roles = version.roles()
    advisors = sum(r not in ("Layout", "Checker") for r in roles)
    iters = int(version.config.get("max_iters", 3))
    per_run_max = advisors + iters + (iters - 1 if "Checker" in roles else 0)
    runs = generations * (n_train + n_dev + CANDIDATES_PER_GEN * n_dev) + 3 * n_heldout
    return {"runs": runs, "worker_min": runs * (advisors + 1), "worker_max": runs * per_run_max,
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
    print(f"score {r.score * 100:.2f}/100 (greedy {task.payload['greedy_score']:.2f}) feasible={r.passed} "
          f"iterations={r.iterations} calls={r.calls} tokens={r.prompt_tokens}+{r.completion_tokens} "
          f"cost=${r.cost_usd:.5f} wall={r.wall_s:.1f}s")
    print(f"failures: {r.failures}")
    print(f"attempts: {json.dumps(notes['attempts'])}")
    for v in notes["violations"][:5]:
        print(f"  [{v['check']}] {v['detail']} -> fix: {v['fix']}")
    if notes["errors"]:
        print(f"errors: {notes['errors']}")


def evolve(llm: LLM, reviewer_model: str, out: Path, generations: int, seed: int, workers: int,
           v0: TeamVersion, splits: tuple[list[Task], list[Task], list[Task]]) -> None:
    train, dev, heldout = splits
    evolver = Evolver(datahall_run_team, llm, reviewer_model, SharedBrain(out / "brain"), out / "versions",
                      max_workers=workers, seed=seed, initial=v0, reports_dir=out / "reports")
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
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--one", action="store_true", help="run one train hall with v0 and print its bill")
    ap.add_argument("--out", type=Path, default=None, help="default runs/datahall-<timestamp>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="Evolver threads; the client still caps CRUSOE_RPM")
    ap.add_argument("--max-iters", type=int, default=3, help="v0 layout attempts per hall")
    ap.add_argument("--max-tokens", type=int, default=8192, help="per call; Nemotron reasoning counts against it")
    ap.add_argument("--worker-model", default=NANO)
    ap.add_argument("--reviewer-model", default=SUPER)
    args = ap.parse_args()

    v0 = datahall_v0()
    v0.config.update(max_iters=args.max_iters, max_tokens=args.max_tokens, default_model=args.worker_model)
    if args.offline:
        llm: Any = None
    else:
        try:
            llm = CrusoeLLM()
        except MissingKeyError as exc:
            print(f"No Crusoe API key configured: {exc}", file=sys.stderr)
            return 2
    splits = load_tasks(args.seed)
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
    evolve(llm, args.reviewer_model, out, args.generations, args.seed, args.workers, v0, splits)
    if args.live:
        usage = llm.ledger.summary()
        (out / "reports" / "usage.json").write_text(json.dumps(usage, indent=1))
        print(f"usage: {json.dumps(usage['total'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
