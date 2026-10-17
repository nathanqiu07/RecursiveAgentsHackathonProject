"""Evolve the ALE-Bench team on AHC039: a Nemotron solver writes Python programs, the official contest scorer judges.

Usage (repo venv; fetch the data once with `.venv/bin/python -m benchmarks.alebench.fetch`):
  .venv/bin/python scripts/run_alebench_evolve.py --offline           # scripted LLM, real scorer, proves plumbing
  .venv/bin/python scripts/run_alebench_evolve.py --live --one        # one train task with v0, to budget a run
  .venv/bin/python scripts/run_alebench_evolve.py --live [--generations 2] [--out DIR]
Task score = mean over its seeds of case score / reference greedy case score (1.0 = greedy). The final report
scores the best program on all 150 private Lite seeds and converts the total into an AtCoder rank and performance
against the contest's human entrants. Held-out seeds and their scores never enter a prompt.
Exit codes: 0 done, 2 no Crusoe API key configured.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.alebench import scorer, splits  # noqa: E402
from benchmarks.alebench.team_runner import ROLE, alebench_v0, run_team  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.crusoe import LIGHTNING, NANO, SUPER, CrusoeLLM, MissingKeyError  # noqa: E402
from team.evolve import Evolver, Gate, TeamVersion, markdown_table  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.sa_library import add_library_args, configure_from_args  # noqa: E402
from team.types import LLM, RunResult, Task  # noqa: E402

CANDIDATES_PER_GEN = 3  # Reviewer max_mutations
MODELS = {"super": SUPER, "lightning": LIGHTNING, "nano": NANO}
OFFLINE_RULE = ("Never print a placeholder: bin the fish on a 50x50 grid and output the best rectangle from a 2D "
                "maximum-subarray search (Kadane over column pairs), then refine it.")


def program_reply(technique: str, code: str) -> str:
    return f"TECHNIQUE: {technique}\n```python\n{code.strip()}\n```"


def offline_llm() -> ScriptedLLM:
    """Canned replies over the real scorer: the solver sends the naive 1x1 square until a playbook rule names
    the grid rectangle search, then sends the reference greedy. The first review adds that rule, so the Gate
    sees a real dev gain and accepts it; the second review proposes nothing."""
    naive = program_reply("1x1 placeholder square", scorer.NAIVE.read_text())
    greedy = program_reply("grid max-subarray rectangle", scorer.GREEDY.read_text())
    reviews = iter([
        {"mutations": [
            {"type": "playbook_add", "role": ROLE, "step": OFFLINE_RULE, "rationale": "every attempt scores ~0",
             "claim": {"dev_gain": 0.9, "cost_change_pct": 0.0, "why": "a rectangle catches a mackerel cluster"}},
            {"type": "config_set", "key": "attempts", "value": 1, "rationale": "half the solver calls",
             "claim": {"dev_gain": 0.0, "cost_change_pct": -50.0, "why": "one attempt instead of two"}}]},
    ])

    def respond(role: str, messages: list[dict[str, str]]) -> str:
        if role == "reviewer":
            return json.dumps(next(reviews, {"mutations": []}))
        return greedy if "Kadane" in messages[0]["content"] else naive

    return ScriptedLLM(respond)


def estimate_calls(version: TeamVersion, n_train: int, n_dev: int, n_heldout: int, generations: int) -> dict[str, int]:
    """Every run makes exactly `attempts` solver calls; dev runs once for the parent and once per candidate."""
    attempts = int(version.config.get("attempts", 2))
    runs = generations * (n_train + n_dev + CANDIDATES_PER_GEN * n_dev) + 3 * n_heldout
    return {"runs": runs, "solver": runs * attempts, "reviewer": generations}


def greedy_row(tasks: list[Task]) -> dict[str, Any]:
    """The reference greedy in the same shape as Evolver.final_report rows: ratio 1.0 on every case by definition."""
    return {"config": "reference greedy (no LLM)", "version": None, "n": len(tasks), "mean_score": 1.0,
            "pass_rate": 1.0, "mean_iterations": 0.0, "tokens": 0, "cost_usd": 0.0}


def rating_rows(heldout_results: list[RunResult], workers: int) -> list[dict[str, Any]]:
    """Totals on all 150 private seeds, ranked against the contest's humans.

    The single best program is chosen by its public-seed mean only, then run once on every private seed.
    """
    private = splits.private_seeds()
    ref = scorer.reference_scores(private)
    rows = [{"entry": "naive 1x1 square", **scorer.rank_performance(len(private))},
            {"entry": "reference greedy", **scorer.rank_performance(sum(ref.values()))}]
    notes = [json.loads(r.notes) for r in heldout_results]
    with_program = [n for n in notes if n.get("program")]
    if len(with_program) == len(notes) and notes:
        rows.append({"entry": "team, one program per 30-seed batch", **scorer.rank_performance(
            sum(n["total"] for n in notes))})
    if with_program:
        pick = max(with_program, key=lambda n: n["feedback_mean_ratio"])
        cases = scorer.evaluate(pick["program"], private, scorer.contest_time_limit(), workers)
        failed = sum(c["status"] != "ok" for c in cases)
        rows.append({"entry": f"team, best program ({pick['technique'][:40]})"
                              + (f", {failed} failed cases" if failed else ""),
                     **scorer.rank_performance(sum(c["score"] for c in cases))})
    rows.append({"entry": "top human", **scorer.rank_performance(rows[0]["top_human_total"])})
    return rows


def rating_table(rows: list[dict[str, Any]]) -> str:
    out = ["| entry | total (150 private cases) | mean per case | vs greedy | rank | performance |",
           "|---|---|---|---|---|---|"]
    greedy_total = next(r["total"] for r in rows if r["entry"] == "reference greedy")
    for r in rows:
        out.append(f"| {r['entry']} | {r['total']:,} | {r['total'] / 150:.0f} | {r['total'] / greedy_total:.3f}x | "
                   f"{r['rank']}/{r['entrants']} | {r['performance']} |")
    return "\n".join(out)


def run_one(llm: LLM, task: Task, v0: TeamVersion, out: Path) -> None:
    v0.config["log_dir"] = str(out / "logs")
    with tempfile.TemporaryDirectory() as tmp:
        r = run_team(v0, SharedBrain(tmp), task, llm)
    notes = json.loads(r.notes)
    seeds = task.payload["seeds"]
    ref = scorer.reference_scores(seeds)
    print(f"task {r.task_id} ({task.text}), model {v0.config['models'][ROLE]}, thinking "
          f"{v0.config['thinking'][ROLE]}")
    print(f"score {r.score:.4f} (mean ratio vs greedy) passed={r.passed} iterations={r.iterations} calls={r.calls} "
          f"tokens={r.prompt_tokens}+{r.completion_tokens} cost=${r.cost_usd:.5f} wall={r.wall_s:.1f}s")
    print(f"failures: {r.failures}")
    print(f"attempts: {json.dumps(notes['attempts'])}")
    if notes.get("program"):
        cases = scorer.evaluate(notes["program"], seeds, scorer.contest_time_limit())
        mean_case = mean(c["score"] for c in cases)
        proj = scorer.rank_performance(mean_case * 150)
        greedy_proj = scorer.rank_performance(mean(ref.values()) * 150)
        print(f"per case: {[(c['seed'], c['score'], ref[c['seed']], c['wall_s']) for c in cases]}")
        print(f"projected from these {len(seeds)} train cases x150 (not the private set): rank {proj['rank']}/"
              f"{proj['entrants']} perf {proj['performance']} vs greedy rank {greedy_proj['rank']} perf "
              f"{greedy_proj['performance']}")
    print(f"logs: {out / 'logs'}")


def evolve(llm: LLM, reviewer_model: str, out: Path, generations: int, seed: int, workers: int, min_gain: float,
           v0: TeamVersion, tasks: tuple[list[Task], list[Task], list[Task]]) -> None:
    train, dev, heldout = tasks
    captured: dict[tuple, RunResult] = {}

    def run_and_capture(version: Any, brain: Any, task: Task, llm_: Any) -> RunResult:
        r = run_team(version, brain, task, llm_)
        if task.split == "heldout":
            captured[(version.version, str(brain.dir), task.id)] = r
        return r

    evolver = Evolver(run_and_capture, llm, reviewer_model, SharedBrain(out / "brain"), out / "versions",
                      max_workers=workers, seed=seed, initial=v0, gate=Gate(min_gain=min_gain),
                      reports_dir=out / "reports")
    t0 = time.perf_counter()
    final = evolver.run(generations, train, dev, heldout)
    for h in evolver.history:
        print(f"gen {h['generation']}: v{h['parent']} -> v{h['version']} accepted={h['accepted']} "
              f"train={h['train_mean']:.3f} dev={h['dev_mean']:.3f} brain_active={h['brain']['active']}", flush=True)
    print("\nchangelog:\n  " + "\n  ".join(final.changelog))
    rows = evolver.final_report(heldout) + [greedy_row(heldout)]
    table = markdown_table(rows)
    print(f"\nHeld-out ({len(heldout)} batches x 30 private seeds), score = mean case score / greedy case score:\n")
    print(table)
    final_results = [captured[(final.version, str(evolver.brain.dir), t.id)] for t in heldout]
    rating = rating_rows(final_results, workers=4)
    rtable = rating_table(rating)
    print(f"\nAtCoder rating on all 150 private Lite seeds (v{final.version} + final brain):\n")
    print(rtable)
    (out / "reports" / "vs_greedy.md").write_text(table + "\n\n" + rtable + "\n")
    (out / "reports" / "rating.json").write_text(json.dumps(rating, indent=1))
    print(f"\nwall {time.perf_counter() - t0:.1f}s; artifacts: {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="scripted LLM, no network; real scorer")
    mode.add_argument("--live", action="store_true", help="Crusoe Managed Inference")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--one", action="store_true", help="run one train task with v0 and print its bill")
    ap.add_argument("--out", type=Path, default=None, help="default runs/alebench-<timestamp>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2, help="Evolver threads; Super's 30 RPM is shared, keep <= 2")
    ap.add_argument("--min-gain", type=float, default=0.01, help="Gate: dev mean ratio gain needed to accept")
    ap.add_argument("--attempts", type=int, default=2, help="v0 solver attempts per task")
    ap.add_argument("--max-tokens", type=int, default=8000, help="per call; reasoning counts against it")
    ap.add_argument("--solver-model", choices=sorted(MODELS), default="super")
    ap.add_argument("--reviewer-model", choices=sorted(MODELS), default="super")
    ap.add_argument("--thinking", action="store_true", help="v0 solver reasoning on (default off)")
    ap.add_argument("--rpm", type=float, default=None, help="client cap per model (default CRUSOE_RPM or 25)")
    add_library_args(ap)
    args = ap.parse_args()
    configure_from_args(args, "ale-bench-ahc039")

    scorer.require_data()
    v0 = alebench_v0()
    v0.config.update(models={ROLE: MODELS[args.solver_model]}, thinking={ROLE: args.thinking},
                     attempts=args.attempts, max_tokens=args.max_tokens)
    if args.offline:
        llm: Any = offline_llm()
    else:
        try:
            # dotenv_path=None: the key comes only from the environment of this command.
            llm = CrusoeLLM(rpm=args.rpm, dotenv_path=None)
        except MissingKeyError as exc:
            print(f"No Crusoe API key configured: {exc}", file=sys.stderr)
            return 2
    tasks = splits.load_tasks()
    train, dev, heldout = tasks
    out = args.out or Path("runs") / f"alebench-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    if args.one:
        run_one(llm, train[0], v0, out)
        if args.live:
            usage = llm.ledger.summary()
            (out / "usage.json").write_text(json.dumps(usage, indent=1))
            print(json.dumps(usage["by_model"], indent=1))
        return 0

    est = estimate_calls(v0, len(train), len(dev), len(heldout), args.generations)
    rpm = getattr(llm, "rpm", None)
    print(f"Estimated LLM calls: {est['runs']} team runs -> {est['solver']} solver calls ({MODELS[args.solver_model]})"
          f" + {est['reviewer']} reviewer calls ({MODELS[args.reviewer_model]})"
          + (f"; at {rpm:.0f} RPM that is >= {est['solver'] / rpm:.0f} min" if rpm else ""), flush=True)
    evolve(llm, MODELS[args.reviewer_model], out, args.generations, args.seed, args.workers, args.min_gain, v0, tasks)
    if args.live:
        usage = llm.ledger.summary()
        (out / "reports" / "usage.json").write_text(json.dumps(usage, indent=1))
        print(f"usage: {json.dumps(usage['total'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
