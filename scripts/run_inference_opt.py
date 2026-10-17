"""Evolve the inference-serving team: answer exact data-center ops questions on Crusoe at the lowest measured cost.

Layer 4 of "optimize every layer of an AI data center": the workload is fixed (benchmarks/inference_opt/workload.py,
questions drawn from DoneTwin halls), the verifier is exact-answer checking in code plus Crusoe's billed tokens x
team.crusoe.PRICES plus measured latency. The Reviewer may change model per question type, thinking, max_tokens,
batching and prompts; the Gate keeps a candidate that is no less accurate, loses no pass, and cuts cost >= 15%.

Usage (repo venv):
  .venv/bin/python scripts/run_inference_opt.py --offline            # scripted LLM, tests only
  .venv/bin/python scripts/run_inference_opt.py --live --one         # one train task with v0, to budget a run
  .venv/bin/python scripts/run_inference_opt.py --live [--generations 2] [--out DIR]
Exit codes: 0 done, 2 no Crusoe API key configured.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.inference_opt.team_runner import run_team, v0_config, v0_prompts  # noqa: E402
from benchmarks.inference_opt.workload import QTYPES, SPLIT_SIZES, load_tasks  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.crusoe import LIGHTNING, NANO, SUPER, CrusoeLLM, MissingKeyError  # noqa: E402
from team.evolve import Evolver, Gate, TeamVersion, markdown_table, papers_table  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.sa_library import add_library_args, configure_from_args  # noqa: E402
from team.research import add_research_args, research_from_args  # noqa: E402
from team.types import LLM, RunResult, Task  # noqa: E402

CANDIDATES_PER_GEN = 3  # Reviewer max_mutations
MODELS = {"nano": NANO, "super": SUPER, "lightning": LIGHTNING}
_HALL = re.compile(r"^Hall (\S+)\.", re.MULTILINE)
_QID = re.compile(r"^\[(q\d+)\]", re.MULTILINE)


def inference_v0(model: str = SUPER, thinking: bool = True, max_tokens: int = 4000) -> TeamVersion:
    return TeamVersion(version=0, parent=None, prompts=v0_prompts(), playbooks={},
                       config=v0_config(max_tokens=max_tokens, thinking=thinking, model=model),
                       changelog=["v0: one model for every question type, reasoning on, one question per call"])


def offline_llm(tasks: list[Task]) -> ScriptedLLM:
    """Answers every question exactly (looked up by hall and question id), so offline scores are the real checker's;
    the Reviewer proposes batching (fewer prompt tokens, accepted on cost), a cheaper model (the scripted LLM bills
    no USD and the same tokens, so no cut: rejected) and an invalid knob (dropped)."""
    truth = {t.payload["hall"]: {q["id"]: q["answer"] for q in t.payload["questions"]} for t in tasks}
    reviews = iter([
        {"mutations": [
            {"type": "config_set", "key": "batch_size", "value": 3, "rationale": "one call per type, not per question",
             "claim": {"dev_gain": 0.0, "cost_change_pct": -40.0, "why": "shared prompt sent once per batch"}},
            {"type": "config_set", "key": "default_model", "value": NANO, "rationale": "arithmetic does not need Super",
             "claim": {"dev_gain": 0.0, "cost_change_pct": -90.0, "why": "Nano is 12x cheaper per output token"}},
            {"type": "config_set", "key": "max_tokens", "value": 10, "rationale": "invalid on purpose",
             "claim": {"dev_gain": 0.0, "cost_change_pct": -50.0, "why": "below the 256 floor, must be dropped"}}]},
    ])

    def respond(role: str, messages: list[dict[str, str]]) -> str:
        if role == "reviewer":
            return json.dumps(next(reviews, {"mutations": []}))
        user = messages[-1]["content"]
        hall = _HALL.search(user)
        answers = truth.get(hall.group(1), {}) if hall else {}
        return json.dumps({"answers": {qid: answers.get(qid) for qid in _QID.findall(user)}})

    return ScriptedLLM(respond)


REVIEW_FORMAT = ('\n\nExact reply shape (claim is a nested object): {"mutations": [{"type": "config_set", '
                 '"key": "batch_size", "value": 3, "rationale": "...", "claim": {"dev_gain": 0.0, '
                 '"cost_change_pct": -40.0, "why": "..."}}]}. Propose up to 3 different single-key changes.')


class ReviewerBudget:
    """Passes every call through, but gives the Reviewer's call a reply budget it can finish in.

    team.evolve.Reviewer calls chat() with the default max_tokens (2048) and the model's default reasoning; on
    Nemotron with reasoning on, the reasoning alone used up that budget, the reply came back empty, and the first
    live attempt's generation 0 proposed no mutations at all. Workers' calls are untouched.
    """

    def __init__(self, llm: Any, max_tokens: int = 8000, thinking: bool | None = False) -> None:
        self._llm, self._max_tokens, self._thinking = llm, max_tokens, thinking

    def chat(self, model: str, messages: list[dict[str, str]], **kw: Any) -> Any:
        if kw.get("role") == "reviewer":
            kw["max_tokens"] = max(int(kw.get("max_tokens") or 0), self._max_tokens)
            if kw.get("thinking") is None:
                kw["thinking"] = self._thinking
            # The second attempt's one mutation put dev_gain/cost_change_pct at the top level and was dropped.
            messages = messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + REVIEW_FORMAT}]
        return self._llm.chat(model, messages, **kw)
        return self._llm.chat(model, messages, **kw)

    def __getattr__(self, name: str) -> Any:  # ledger, rpm, calls, calls_for, ...
        return getattr(self._llm, name)


class Recorder:
    """Wraps run_team so the held-out table can show latency, which Evolver.final_report rows do not carry."""

    def __init__(self) -> None:
        self.records: list[tuple[int, str, str, RunResult]] = []

    def __call__(self, version: Any, brain: Any, task: Task, llm: Any) -> RunResult:
        r = run_team(version, brain, task, llm)
        self.records.append((version.version, version.key(), str(getattr(brain, "dir", "")), r))
        return r

    def find(self, version: TeamVersion, brain_dir: str | None, task_ids: set[str]) -> list[RunResult]:
        """Latest result per task for this version; brain_dir None = any brain other than the learned one."""
        out: dict[str, RunResult] = {}
        for _, key, bdir, r in self.records:
            if key != version.key() or r.task_id not in task_ids:
                continue
            if (brain_dir is None and bdir != self.learned) or bdir == brain_dir:
                out[r.task_id] = r
        return [out[i] for i in sorted(out)]

    learned = ""


def latency_row(label: str, rs: list[RunResult]) -> dict[str, Any]:
    n = max(1, len(rs))
    calls = [x for r in rs for x in json.loads(r.notes or "{}").get("latencies_s", [])]
    correct = sum(r.solution["correct"] for r in rs if isinstance(r.solution, dict))
    questions = sum(r.solution["n"] for r in rs if isinstance(r.solution, dict))
    cost = sum(r.cost_usd for r in rs)
    tokens = sum(r.prompt_tokens + r.completion_tokens for r in rs)
    return {"config": label, "n_tasks": len(rs), "questions": questions, "correct": correct,
            "accuracy": round(correct / questions, 4) if questions else 0.0,
            "pass_rate": round(sum(r.passed for r in rs) / n, 4),
            "cost_per_task_usd": round(cost / n, 8),
            "cost_per_correct_usd": round(cost / correct, 8) if correct else None,
            "p50_call_latency_s": round(statistics.median(calls), 3) if calls else None,
            "p50_task_wall_s": round(statistics.median(r.wall_s for r in rs), 3) if rs else None,
            "calls_per_task": round(sum(r.calls for r in rs) / n, 2),
            "tokens_per_task": round(tokens / n, 1), "cost_usd": round(cost, 6)}


def per_type_row(rs: list[RunResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for qtype in QTYPES:
        ts = [json.loads(r.notes or "{}").get("per_type", {}).get(qtype, {}) for r in rs]
        lat = [x for t in ts for x in t.get("latency_s", [])]
        n, c = sum(t.get("n", 0) for t in ts), sum(t.get("correct", 0) for t in ts)
        out[qtype] = {"accuracy": round(c / n, 4) if n else None, "cost_usd": round(sum(t.get("cost_usd", 0) for t in ts), 6),
                      "p50_latency_s": round(statistics.median(lat), 3) if lat else None}
    return out


def latency_table(rows: list[dict[str, Any]]) -> str:
    def f(x: Any, fmt: str) -> str:
        return "n/a" if x is None else format(x, fmt)

    lines = ["| Config | Accuracy | Pass rate | Cost/task (USD) | Cost/correct (USD) | p50 call latency (s) | "
             "p50 task wall (s) | Calls/task | Tokens/task |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['config']} | {r['accuracy']:.3f} ({r['correct']}/{r['questions']}) | {r['pass_rate']:.0%} | "
              f"{r['cost_per_task_usd']:.6f} | {f(r['cost_per_correct_usd'], '.7f')} | "
              f"{f(r['p50_call_latency_s'], '.2f')} | {f(r['p50_task_wall_s'], '.2f')} | {r['calls_per_task']:.1f} | "
              f"{r['tokens_per_task']:.0f} |" for r in rows]
    return "\n".join(lines) + "\n"


def estimate_calls(n_train: int, n_dev: int, n_heldout: int, generations: int, per_task: int) -> dict[str, int]:
    runs = generations * (n_train + n_dev + CANDIDATES_PER_GEN * n_dev) + 3 * n_heldout
    return {"runs": runs, "worker_max": runs * per_task, "reviewer": generations}


def run_one(llm: LLM, task: Task, v0: TeamVersion) -> RunResult:
    with tempfile.TemporaryDirectory() as tmp:
        r = run_team(v0, SharedBrain(tmp), task, llm)
    notes = json.loads(r.notes)
    print(f"task {r.task_id}: accuracy {r.score:.3f} passed={r.passed} calls={r.calls} "
          f"tokens={r.prompt_tokens}+{r.completion_tokens} cost=${r.cost_usd:.6f} wall={r.wall_s:.1f}s")
    print(f"call latencies: {notes['latencies_s']}")
    print(f"failures: {r.failures}")
    for qtype, t in notes["per_type"].items():
        print(f"  {qtype}: {t['correct']}/{t['n']} cost ${t['cost_usd']:.6f} tokens {t['tokens']} "
              f"latency {t['latency_s']}")
    return r


def make_evolver(llm: LLM, args: argparse.Namespace, out: Path, v0: TeamVersion, recorder: Recorder) -> Evolver:
    brain = SharedBrain(out / "brain")
    recorder.learned = str(brain.dir)
    return Evolver(recorder, ReviewerBudget(llm), args.reviewer_model, brain, out / "versions", max_workers=args.workers,
                   seed=args.seed, initial=v0, gate=Gate(min_gain=args.min_gain), reports_dir=out / "reports",
                   research=research_from_args(args, "inference_opt"))


def evolve(llm: LLM, args: argparse.Namespace, out: Path, v0: TeamVersion,
           splits: tuple[list[Task], list[Task], list[Task]]) -> dict[str, Any]:
    train, dev, heldout = splits
    recorder = Recorder()
    evolver = make_evolver(llm, args, out, v0, recorder)
    t0 = time.perf_counter()
    final = evolver.run(args.generations, train, dev, heldout)
    for h in evolver.history:
        print(f"gen {h['generation']}: v{h['parent']} -> v{h['version']} accepted={h['accepted']} "
              f"({h['accepted_reason']}) train={h['train_mean']:.3f} dev={h['dev_mean']:.3f}", flush=True)
    print("\npapers (claimed vs measured on dev):\n" + papers_table(evolver.papers))
    print("changelog:\n  " + "\n  ".join(final.changelog))
    rows = evolver.final_report(heldout)
    ids = {t.id for t in heldout}
    cold = recorder.find(evolver.v0, None, ids)
    final_rs = recorder.find(final, recorder.learned, ids)
    lat_rows = [latency_row("v0 (cold brain)", cold), latency_row(f"v{final.version} + final brain (final)", final_rs)]
    table = latency_table(lat_rows)
    v0r, fr = lat_rows
    ratio = fr["cost_per_task_usd"] / v0r["cost_per_task_usd"] if v0r["cost_per_task_usd"] else None
    headline = (f"Held-out: accuracy {v0r['accuracy']:.3f} -> {fr['accuracy']:.3f} at "
                + (f"{ratio:.1%} of the cost per task" if ratio is not None else "n/a cost")
                + f"; p50 call latency {v0r['p50_call_latency_s']} s -> {fr['p50_call_latency_s']} s.")
    print(f"\nHeld-out ({len(heldout)} unseen halls, {v0r['questions']} questions), Evolver ablation:\n")
    print(markdown_table(rows))
    print("Cost and latency, v0 vs final:\n")
    print(table)
    print(headline)
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "heldout_cost_latency.md").write_text(table + "\n" + headline + "\n")
    (reports / "heldout_cost_latency.json").write_text(json.dumps(
        {"rows": lat_rows, "per_type": {"v0": per_type_row(cold), "final": per_type_row(final_rs)},
         "final_version": final.version, "final_config": final.config, "changelog": final.changelog,
         "headline": headline, "history": evolver.history}, indent=1, default=str))
    print(f"\nwall {time.perf_counter() - t0:.1f}s; artifacts: {out}")
    return {"rows": lat_rows, "final": final, "evolver": evolver}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="scripted LLM, no network (tests only)")
    mode.add_argument("--live", action="store_true", help="Crusoe Managed Inference")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--one", action="store_true", help="run one train task with v0 and print its bill")
    ap.add_argument("--out", type=Path, default=None, help="default runs/inference-opt-<timestamp>/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="Evolver threads; each task also runs up to 4 calls at once")
    ap.add_argument("--split", type=int, nargs=3, default=list(SPLIT_SIZES), metavar=("TRAIN", "DEV", "HELDOUT"))
    ap.add_argument("--per-type", type=int, default=3, help="questions per type per task (4 types)")
    ap.add_argument("--v0-model", choices=sorted(MODELS), default="super")
    ap.add_argument("--v0-reasoning", choices=("on", "off"), default="on")
    ap.add_argument("--max-tokens", type=int, default=4000, help="v0 per-call cap; reasoning counts against it")
    ap.add_argument("--reviewer-model", choices=sorted(MODELS), default="super")
    ap.add_argument("--min-gain", type=float, default=0.05, help="Gate: accuracy gain to accept on score alone")
    add_library_args(ap)
    add_research_args(ap)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_from_args(args, "inference_opt")
    args.reviewer_model = MODELS[args.reviewer_model]
    v0 = inference_v0(MODELS[args.v0_model], args.v0_reasoning == "on", args.max_tokens)
    splits = load_tasks(tuple(args.split), args.per_type)
    train, dev, heldout = splits
    if args.offline:
        llm: Any = offline_llm(train + dev + heldout)
    else:
        try:
            llm = CrusoeLLM()
        except MissingKeyError as exc:
            print(f"No Crusoe API key configured: {exc}", file=sys.stderr)
            return 2
    if args.one:
        run_one(llm, train[0], v0)
        if args.live:
            print(json.dumps(llm.ledger.summary()["by_role"], indent=1))
        return 0
    est = estimate_calls(len(train), len(dev), len(heldout), args.generations, 4 * args.per_type)
    print(f"v0: {args.v0_model} for every type, reasoning {args.v0_reasoning}, max_tokens {args.max_tokens}, "
          f"batch 1; reviewer {args.reviewer_model.split('/')[-1]}", flush=True)
    print(f"Estimated: ~{est['runs']} team runs -> <= {est['worker_max']} worker calls + {est['reviewer']} reviewer "
          f"calls", flush=True)
    out = args.out or Path("runs") / f"inference-opt-{time.strftime('%Y%m%d-%H%M%S')}"
    evolve(llm, args, out, v0, splits)
    if args.live:
        usage = llm.ledger.summary()
        (out / "reports" / "usage.json").write_text(json.dumps(usage, indent=1))
        print(f"usage: {json.dumps(usage['total'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
