"""Evolve the QAPLIB solver team: case study 2 for the same Shared Brain + Version N+1 loop as the data hall.

A solver role writes a heuristic program, the exact verifier scores its permutation, and the Evolver
gates the team's edits to itself on dev instances. Held-out instances run only in the final report.

Usage (repo venv):
  .venv/bin/python scripts/run_qaplib_evolve.py --offline            # scripted LLM, proves the loop in seconds
  .venv/bin/python scripts/run_qaplib_evolve.py --live --one         # one train instance with v0, to budget a run
  .venv/bin/python scripts/run_qaplib_evolve.py --live [--generations 2] [--split 6 4 8] [--out DIR]
  .venv/bin/python scripts/run_qaplib_evolve.py --live --instances dc   # GPU rack placement halls (dc_instances.py)
`--instances dc` swaps QAPLIB for the data-center QAP instances (default split 4 3 6, all of them); there "BKS" is
each instance's stored reference cost. Scores are BKS / cost (1.0 = best known), so they sit near 0.99; --min-gain sets the Gate's bar to match.
Exit codes: 0 done, 2 no Crusoe API key configured.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.qaplib import dc_instances  # noqa: E402
from benchmarks.qaplib.fetch import family  # noqa: E402
from benchmarks.qaplib.load import load_tasks  # noqa: E402
from benchmarks.qaplib.team_runner import DEFAULT_SOLVER_PROMPT, run_team  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.crusoe import SUPER, CrusoeLLM, MissingKeyError  # noqa: E402
from team.evolve import Evolver, Gate, TeamVersion  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.sa_library import add_library_args, configure_from_args  # noqa: E402
from team.research import add_research_args, research_from_args  # noqa: E402
from team.types import LLM, RunResult, Task  # noqa: E402

CANDIDATES_PER_GEN = 3  # Reviewer max_mutations
SPLIT_SIZES = (6, 4, 8)
MAX_N = 50
BASELINE_JSON = ROOT / "reports" / "qaplib_baseline.json"

RANDOM_PROGRAM = """TECHNIQUE: random permutation
```python
import json, random, sys
n = json.load(open(sys.argv[1]))["n"]
p = list(range(n))
random.Random(0).shuffle(p)
print("PERMUTATION:", " ".join(str(x + 1) for x in p))
```"""

# The swap delta is benchmarks/qaplib/baseline.py's, so offline v1 writes the same textbook heuristic as the baseline,
# but one descent per attempt instead of restarts until the time limit.
TWO_SWAP_PROGRAM = """TECHNIQUE: 2-swap descent from a random start
```python
import json, sys, time
import numpy as np
SEED = {seed}
d = json.load(open(sys.argv[1]))
deadline = time.monotonic() + 0.8 * float(sys.argv[2])
a, b, n = np.array(d["A"], dtype=np.int64), np.array(d["B"], dtype=np.int64), d["n"]

def deltas(bp, r):
    diag_a, diag_b = np.diag(a), np.diag(bp)
    s1 = ((a[:, r][:, None] - a) * (bp - bp[:, r][:, None])).sum(axis=0)
    s1 -= (a[r, r] - a[r, :]) * (bp[r, :] - bp[r, r]) + (a[:, r] - diag_a) * (diag_b - bp[:, r])
    s2 = ((a[r, :][None, :] - a) * (bp - bp[r, :][None, :])).sum(axis=1)
    s2 -= (a[r, r] - a[:, r]) * (bp[:, r] - bp[r, r]) + (a[r, :] - diag_a) * (diag_b - bp[r, :])
    delta = (a[r, r] - diag_a) * (diag_b - bp[r, r]) + (a[r, :] - a[:, r]) * (bp[:, r] - bp[r, :]) + s1 + s2
    delta[r] = 0
    return delta

rng = np.random.default_rng(SEED)
perm = rng.permutation(n)
bp = b[perm][:, perm]
improved = True
while improved and time.monotonic() < deadline:
    improved = False
    for r in rng.permutation(n):
        delta = deltas(bp, int(r))
        s = int(delta.argmin())
        if delta[s] < 0:
            perm[[r, s]] = perm[[s, r]]
            bp = b[perm][:, perm]
            improved = True
print("PERMUTATION:", " ".join(str(x + 1) for x in perm))
```"""

LOCAL_SEARCH_STEP = ("Never submit a bare random permutation: start from one and run 2-swap local search with O(n) "
                     "swap deltas until no swap improves or the time limit nears, then print the best permutation.")


def qaplib_v0(max_tokens: int = 6000, attempts: int = 3, time_limit_s: float = 10.0, model: str = SUPER) -> TeamVersion:
    return TeamVersion(
        version=0, parent=None, prompts={"solver": DEFAULT_SOLVER_PROMPT}, playbooks={},
        config={"roles_enabled": ["solver"], "models": {"solver": model}, "default_model": SUPER,
                "thinking": {"solver": False}, "temperature": 0.4, "max_tokens": max_tokens,
                "attempts": attempts, "time_limit_s": time_limit_s},
        changelog=["v0: one solver role writes a QAP heuristic program; the best verified attempt scores"])


def select_tasks(sizes: tuple[int, int, int] = SPLIT_SIZES, seed: int = 0,
                 max_n: int = MAX_N) -> tuple[list[Task], list[Task], list[Task]]:
    """A seeded subset of each frozen split, so held-out stays disjoint from train and dev by construction.

    Within a split, instances with n <= max_n are dealt round-robin over families in a seeded order,
    so a small subset still spans as many families as it can.
    """
    out = []
    for split, k in zip(("train", "dev", "heldout"), sizes):
        rng = random.Random(f"{seed}-{split}")
        by_family: dict[str, list[Task]] = {}
        for t in load_tasks(split):
            if t.payload["n"] <= max_n:
                by_family.setdefault(family(t.payload["name"]), []).append(t)
        queues = [by_family[f] for f in sorted(by_family)]
        rng.shuffle(queues)
        for q in queues:
            rng.shuffle(q)
        dealt = [q[i] for i in range(max(map(len, queues), default=0)) for q in queues if i < len(q)]
        if len(dealt) < k:
            print(f"warning: {split} has only {len(dealt)} instances with n <= {max_n}; using all of them",
                  file=sys.stderr)
        out.append(dealt[:k])
    return tuple(out)  # type: ignore[return-value]


def select_dc_tasks(sizes: tuple[int, int, int] = dc_instances.SPLIT_SIZES, seed: int = 0,
                    max_n: int = MAX_N) -> tuple[list[Task], list[Task], list[Task]]:
    """A seeded subset of each data-center QAP split (all of it at the default sizes); splits are disjoint by seed."""
    out = []
    for split, k in zip(("train", "dev", "heldout"), sizes):
        tasks = [t for t in dc_instances.load_tasks(split) if t.payload["n"] <= max_n]
        random.Random(f"{seed}-dc-{split}").shuffle(tasks)
        if len(tasks) < k:
            print(f"warning: dc {split} has only {len(tasks)} instances with n <= {max_n}; using all of them",
                  file=sys.stderr)
        out.append(tasks[:k])
    return tuple(out)  # type: ignore[return-value]


class Recorder:
    """Wraps run_team to keep each run's verified score under the key the Evolver caches evaluations by,
    and the order tasks ran in, so the report can add gap columns and tests can see what ran when."""

    def __init__(self, inner: Callable[..., RunResult] = run_team):
        self.inner = inner
        self.scores: dict[tuple[str, str, str], float] = {}
        self.order: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, version: TeamVersion, brain: SharedBrain, task: Task, llm: LLM) -> RunResult:
        result = self.inner(version, brain, task, llm)
        with self._lock:
            self.scores[(version.key(), str(brain.dir), task.id)] = result.score
            self.order.append(task.id)
        return result

    def heldout_scores(self, version: TeamVersion, ids: list[str], brain_dir: str, same_brain: bool) -> list[float]:
        found = {tid: s for (key, d, tid), s in self.scores.items()
                 if key == version.key() and (d == brain_dir) == same_brain}
        return [found[t] for t in ids if t in found]


def gap_columns(scores: list[float]) -> dict[str, Any]:
    """Gap follows from score = BKS / cost; a score of 0 means no valid permutation and has no gap."""
    valid = [s for s in scores if s > 0]
    gaps = [(1 / s - 1) * 100 for s in valid]
    return {"mean_gap_pct": round(mean(gaps), 3) if gaps else None, "at_bks": sum(s >= 1.0 for s in valid),
            "invalid": len(scores) - len(valid)}


def baseline_row(heldout: list[Task], path: Path = BASELINE_JSON) -> dict[str, Any]:
    """reports/qaplib_baseline.json's 2-swap result on the same held-out instances, in final_report's row shape."""
    if heldout and all(dc_instances.is_dc(t.payload["name"]) for t in heldout):
        return dc_baseline_row(heldout)
    data = json.loads(path.read_text())
    rows = [data["instances"][t.payload["name"]]["two_swap"] for t in heldout]
    n = max(1, len(rows))
    return {"config": f"2-swap baseline (no LLM, {data['time_limit_s']:g} s restarts)", "version": None,
            "n": len(rows), "mean_score": round(sum(r["score"] for r in rows) / n, 4),
            "pass_rate": round(sum(r["gap_pct"] <= 1.0 for r in rows) / n, 4),
            "mean_gap_pct": round(sum(r["gap_pct"] for r in rows) / n, 3),
            "at_bks": sum(r["gap_pct"] <= 0 for r in rows), "invalid": 0,
            "calls_per_task": 0.0, "tokens_per_task": 0.0, "cost_per_task_usd": 0.0}


def dc_baseline_row(heldout: list[Task]) -> dict[str, Any]:
    """The stored 2-swap result (dc_instances/index.json) on the same data-center held-out instances."""
    index = dc_instances.load_index()
    rows = []
    for t in heldout:
        e = index[t.payload["name"]]
        cost = e["two_swap"]["cost"]
        rows.append({"score": e["bks"] / cost, "gap_pct": (cost - e["bks"]) / e["bks"] * 100})
    n = max(1, len(rows))
    return {"config": f"2-swap baseline (no LLM, {dc_instances.TWO_SWAP_TIME_S:g} s restarts)", "version": None,
            "n": len(rows), "mean_score": round(sum(r["score"] for r in rows) / n, 4),
            "pass_rate": round(sum(r["gap_pct"] <= 1.0 for r in rows) / n, 4),
            "mean_gap_pct": round(sum(r["gap_pct"] for r in rows) / n, 3),
            "at_bks": sum(r["gap_pct"] <= 0 for r in rows), "invalid": 0,
            "calls_per_task": 0.0, "tokens_per_task": 0.0, "cost_per_task_usd": 0.0}


def qap_table(rows: list[dict[str, Any]]) -> str:
    def cell(r: dict, key: str, fmt: str) -> str:
        return "n/a" if r.get(key) is None else format(r[key], fmt)

    lines = ["| Config | Mean score | Mean gap to BKS (%) | At/above BKS | Pass (gap <= 1%) | Calls/task | "
             "Tokens/task | Cost/task (USD) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        gap = cell(r, "mean_gap_pct", ".2f") + (f" ({r['invalid']} invalid)" if r.get("invalid") else "")
        lines.append(f"| {r['config']} | {r['mean_score']:.4f} | {gap} | {r['at_bks']}/{r['n']} | {r['pass_rate']:.0%} | "
                     f"{cell(r, 'calls_per_task', '.2f')} | {cell(r, 'tokens_per_task', '.0f')} | "
                     f"{cell(r, 'cost_per_task_usd', '.6f')} |")
    return "\n".join(lines) + "\n"


def offline_llm() -> ScriptedLLM:
    """A deterministic stand-in: the solver writes a random-permutation program until a playbook rule asks for
    2-swap local search, then a 2-swap descent seeded by the attempt number.

    Only the replies are canned; every program really runs and the exact verifier scores it. The rule is read
    from the playbook part of the system prompt only, so brain lessons cannot trigger it.
    """
    reviews = iter([
        {"mutations": [
            {"type": "playbook_add", "role": "solver", "step": LOCAL_SEARCH_STEP,
             "rationale": "train runs end at random permutations far above BKS",
             "claim": {"dev_gain": 0.2, "cost_change_pct": 0.0, "why": "local search closes most of a random gap"}},
            {"type": "config_set", "key": "thinking", "value": {"solver": True}, "rationale": "let the solver reason",
             "claim": {"dev_gain": 0.02, "cost_change_pct": 300.0, "why": "reasoning writes better programs"}},
            {"type": "config_set", "key": "models", "value": {"solver": "gpt-4o"}, "rationale": "invalid on purpose",
             "claim": {"dev_gain": 0.1, "cost_change_pct": 0.0, "why": "unknown model, must be dropped"}}]},
        {"mutations": [
            {"type": "config_set", "key": "attempts", "value": 1, "rationale": "one descent may be enough",
             "claim": {"dev_gain": 0.0, "cost_change_pct": -60.0, "why": "a third of the solver calls"}}]},
    ])

    def respond(role: str, messages: list[dict[str, str]]) -> str:
        if role == "reviewer":
            return json.dumps(next(reviews, {"mutations": []}))
        playbook = messages[0]["content"].split("Shared memory from earlier runs")[0]
        if "2-swap local search" in playbook:
            return TWO_SWAP_PROGRAM.replace("{seed}", str(sum(m["role"] == "user" for m in messages)))
        return RANDOM_PROGRAM

    return ScriptedLLM(respond)


class ReviewerReasoningOff:
    """Sends reviewer calls with reasoning off and room for the JSON; every other call passes through untouched.

    With reasoning on, Nemotron Nano spent the reviewer's whole default 2048-token budget thinking, so the JSON was
    cut off and parsed to zero mutations with no dropped record (probe 2026-10-17: 2048/2048 tokens, 0 mutations;
    reasoning off: 306 tokens, 3 mutations). With reasoning off, Super then left out every claim (run 4, gen 0: 2 of 2
    mutations dropped as claimless), so the reviewer's user message also restates the claim rule last.
    """

    CLAIM_RULE = ('\n\nEvery mutation object must include "claim": {"dev_gain": <number>, "cost_change_pct": <number>, '
                  '"why": <string>}. A mutation without a numeric claim is dropped untested.')

    def __init__(self, llm: Any, max_tokens: int = 4096):
        self.llm, self.max_tokens = llm, max_tokens

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        if kwargs.get("role") == "reviewer":
            kwargs.setdefault("thinking", False)
            kwargs["max_tokens"] = max(int(kwargs.get("max_tokens") or 0), self.max_tokens)
            messages = [*messages[:-1], {**messages[-1], "content": messages[-1]["content"] + self.CLAIM_RULE}]
        return self.llm.chat(model, messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.llm, name)


def estimate_calls(version: TeamVersion, n_train: int, n_dev: int, n_heldout: int, generations: int) -> dict[str, int]:
    """Solver calls per run range from 1 (first attempt reaches BKS) to `attempts`; dispute re-runs are extra."""
    attempts = int(version.config.get("attempts", 3))
    runs = generations * (n_train + n_dev + CANDIDATES_PER_GEN * n_dev) + 3 * n_heldout
    return {"runs": runs, "solver_min": runs, "solver_max": runs * attempts, "reviewer": generations}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true", help="scripted LLM, no network")
    mode.add_argument("--live", action="store_true", help="Crusoe Managed Inference")
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--one", action="store_true", help="run one train instance with v0 and print its bill")
    ap.add_argument("--out", type=Path, default=None, help="default runs/qaplib-<timestamp>/")
    ap.add_argument("--seed", type=int, default=0, help="picks the instances within each frozen split")
    ap.add_argument("--workers", type=int, default=4, help="Evolver threads; the client still caps CRUSOE_RPM")
    ap.add_argument("--instances", choices=("qaplib", "dc"), default="qaplib",
                    help="qaplib: the 1991 library (default); dc: GPU rack placement in generated data-center halls")
    ap.add_argument("--split", type=int, nargs=3, default=None, metavar=("TRAIN", "DEV", "HELDOUT"),
                    help=f"default {' '.join(map(str, SPLIT_SIZES))} (qaplib) or "
                         f"{' '.join(map(str, dc_instances.SPLIT_SIZES))} (dc, all of them)")
    ap.add_argument("--max-n", type=int, default=MAX_N, help="largest instance size to sample")
    ap.add_argument("--min-gain", type=float, default=0.003, help="Gate: dev mean score gain needed to accept")
    ap.add_argument("--attempts", type=int, default=3, help="v0 programs per instance, each with verifier feedback")
    ap.add_argument("--time-limit", type=float, default=10.0, help="v0 seconds per program run")
    ap.add_argument("--max-tokens", type=int, default=6000, help="per call; reasoning counts against it")
    ap.add_argument("--solver-model", default=SUPER)
    ap.add_argument("--reviewer-model", default=SUPER)
    add_library_args(ap)
    add_research_args(ap)
    return ap


def make_evolver(llm: LLM, args: argparse.Namespace, out: Path, v0: TeamVersion,
                 recorder: Recorder | None = None) -> Evolver:
    label = "qaplib_dc" if getattr(args, "instances", None) == "dc" else "qaplib"
    return Evolver(recorder or Recorder(), llm, args.reviewer_model, SharedBrain(out / "brain"), out / "versions",
                   max_workers=args.workers, seed=args.seed, initial=v0, gate=Gate(min_gain=args.min_gain),
                   reports_dir=out / "reports", research=research_from_args(args, label))


def final_rows(evolver: Evolver, recorder: Recorder, heldout: list[Task]) -> list[dict[str, Any]]:
    """final_report's three rows with gap columns added, then the 2-swap baseline on the same instances."""
    rows = evolver.final_report(heldout)
    ids, brain_dir = [t.id for t in heldout], str(evolver.brain.dir)
    arms = [(evolver.v0, False), (evolver.v0, True), (evolver.current, True)]
    for row, (version, same_brain) in zip(rows, arms):
        row.update(gap_columns(recorder.heldout_scores(version, ids, brain_dir, same_brain)))
    return rows + [baseline_row(heldout)]


def evolve(llm: LLM, args: argparse.Namespace, out: Path, v0: TeamVersion,
           splits: tuple[list[Task], list[Task], list[Task]]) -> list[dict[str, Any]]:
    train, dev, heldout = splits
    recorder = Recorder()
    evolver = make_evolver(llm, args, out, v0, recorder)
    t0 = time.perf_counter()
    final = evolver.run(args.generations, train, dev, heldout)
    for h in evolver.history:
        print(f"gen {h['generation']}: v{h['parent']} -> v{h['version']} accepted={h['accepted']} "
              f"train={h['train_mean']:.4f} dev={h['dev_mean']:.4f} brain_active={h['brain']['active']}", flush=True)
    print("\nchangelog:\n  " + "\n  ".join(final.changelog))
    rows = final_rows(evolver, recorder, heldout)
    table = qap_table(rows)
    base = rows[-1]["mean_score"]
    vs = "\n".join(f"- {r['config']}: {(r['mean_score'] - base) * 100:+.2f} points vs 2-swap" for r in rows[:-1])
    names = ", ".join(t.payload["name"] for t in heldout)
    bks = "reference" if heldout and all(dc_instances.is_dc(t.payload["name"]) for t in heldout) else "BKS"
    print(f"\nHeld-out ({len(heldout)} unseen instances: {names}), score = {bks} / cost, pass = gap <= 1%:\n")
    print(table)
    print(vs)
    (out / "reports" / "vs_2swap.md").write_text(table + "\n" + vs + "\n")
    (out / "reports" / "qaplib_rows.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwall {time.perf_counter() - t0:.1f}s; artifacts: {out} (claims vs measured: reports/papers.md)")
    return rows


def run_one(llm: LLM, task: Task, v0: TeamVersion, out: Path) -> RunResult:
    v0.config["log_dir"] = str(out / "logs")
    with tempfile.TemporaryDirectory() as tmp:
        r = run_team(v0, SharedBrain(tmp), task, llm)
    v0.config.pop("log_dir")
    gap = (1 / r.score - 1) * 100 if r.score > 0 else None
    max_tokens = v0.config["max_tokens"]
    print(f"task {r.task_id} ({task.text}), BKS {task.payload['bks']}")
    print(f"score {r.score:.4f} gap {'n/a' if gap is None else f'{gap:.2f}%'} passed={r.passed} "
          f"attempts={r.iterations} calls={r.calls} tokens={r.prompt_tokens}+{r.completion_tokens} "
          f"cost=${r.cost_usd:.5f} wall={r.wall_s:.1f}s")
    print(f"completion tokens per call {r.completion_tokens / max(1, r.calls):.0f} (max_tokens {max_tokens})")
    attempts = json.loads(Path(r.log_path).read_text())["attempts"] if r.log_path else []
    for k, a in enumerate(attempts, 1):
        detail = f"gap {a['gap_pct']:.2f}%" if a["status"] == "ok" else a.get("error", "")
        print(f"  attempt {k}: {a['status']} ({a['technique']}) {detail}")
    runnable = sum(a["status"] == "ok" for a in attempts)
    print(f"runnable programs: {runnable}/{len(attempts)}; failures: {r.failures}")
    return r


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dc = args.instances == "dc"
    configure_from_args(args, "qaplib_dc" if dc else "qaplib")
    v0 = qaplib_v0(args.max_tokens, args.attempts, args.time_limit, args.solver_model)
    if args.offline:
        llm: Any = offline_llm()
    else:
        try:
            llm = ReviewerReasoningOff(CrusoeLLM())
        except MissingKeyError as exc:
            print(f"No Crusoe API key configured: {exc}", file=sys.stderr)
            return 2
    if dc:
        splits = select_dc_tasks(tuple(args.split or dc_instances.SPLIT_SIZES), args.seed, args.max_n)
    else:
        splits = select_tasks(tuple(args.split or SPLIT_SIZES), args.seed, args.max_n)
    train, dev, heldout = splits
    out = args.out or ROOT / "runs" / f"qaplib-{'dc-' if dc else ''}{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    if args.one:
        run_one(llm, train[0], v0, out)
        if args.live:
            print(json.dumps(llm.ledger.summary()["by_role"], indent=1))
        return 0

    est = estimate_calls(v0, len(train), len(dev), len(heldout), args.generations)
    rpm = getattr(llm, "rpm", None)
    print(f"Estimated LLM calls: ~{est['runs']} team runs -> {est['solver_min']}-{est['solver_max']} solver calls "
          f"({args.solver_model}) + {est['reviewer']} reviewer calls ({args.reviewer_model})"
          + (f"; at {rpm:.0f} RPM per model that is >= {est['solver_min'] / rpm:.0f}-{est['solver_max'] / rpm:.0f} min"
             if rpm else ""), flush=True)
    for split, tasks in zip(("train", "dev", "held-out"), splits):
        print(f"{split}: " + ", ".join(f"{t.payload['name']}(n={t.payload['n']})" for t in tasks), flush=True)
    evolve(llm, args, out, v0, splits)
    if args.live:
        usage = llm.ledger.summary()
        (out / "reports" / "usage.json").write_text(json.dumps(usage, indent=1))
        print(f"usage: {json.dumps(usage['total'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
