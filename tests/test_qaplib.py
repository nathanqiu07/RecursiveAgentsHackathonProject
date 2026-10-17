"""QAPLIB benchmark: loader BKS validation, exact verifier, frozen splits, run_team's best-of-attempts loop,
and case study 2: the same Evolver loop as the data hall, run offline through scripts/run_qaplib_evolve.py."""
from __future__ import annotations

import contextlib
import io
import itertools
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.qaplib.fetch import check_solution  # noqa: E402
from benchmarks.qaplib.splits import FrozenSplitError, freeze, load_splits, make_splits  # noqa: E402
from benchmarks.qaplib.team_runner import run_team  # noqa: E402
from benchmarks.qaplib.verify import INSTANCES, cost, parse_dat, verify  # noqa: E402
from scripts.run_qaplib_evolve import (  # noqa: E402
    BASELINE_JSON, Recorder, build_parser, final_rows, main, make_evolver, offline_llm, qaplib_v0, select_tasks)
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.types import RunResult, Task  # noqa: E402

HAVE_DATA = (INSTANCES / "nug20.dat").exists() and (INSTANCES / "kra30a.dat").exists()
HAVE_SPLITS = BASELINE_JSON.exists() and all(
    (INSTANCES / f"{name}.dat").exists() for split in ("train", "dev", "heldout") for name in load_splits()[split])
A = [[0, 1, 1, 9], [1, 0, 4, 1], [1, 4, 0, 1], [9, 1, 1, 0]]
B = [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]]

WEAK = "TECHNIQUE: identity\n```python\nimport sys, json\nn = json.load(open(sys.argv[1]))['n']\nprint('PERMUTATION:', ' '.join(str(i + 1) for i in range(n)))\n```"
BRUTE = ("TECHNIQUE: brute force\n```python\nimport sys, json, itertools\nd = json.load(open(sys.argv[1]))\n"
         "n, A, B = d['n'], d['A'], d['B']\n"
         "best = min(itertools.permutations(range(n)), key=lambda p: sum(A[i][j] * B[p[i]][p[j]] for i in range(n) for j in range(n)))\n"
         "print('PERMUTATION:', ' '.join(str(x + 1) for x in best))\n```")
CRASH = "TECHNIQUE: broken\n```python\nprint(undefined_name)\n```"
SLEEP = "TECHNIQUE: sleeper\n```python\nimport time\ntime.sleep(60)\n```"
ENV_CHECK = ("TECHNIQUE: env check\n```python\nimport os, sys\nif 'CRUSOE_API_KEY' in os.environ:\n    raise SystemExit(3)\n"
             "print('PERMUTATION: 1 2 3 4')\n```")


class FakeBrain:
    def __init__(self) -> None:
        self.reads: list[tuple[str, str]] = []

    def read(self, role: str, task_text: str, budget_chars: int = 3000) -> tuple[str, list[str]]:
        self.reads.append((role, task_text))
        return "- [lesson] robust tabu search beats 2-opt on tai-a (id abc123)", ["abc123"]


class TinyInstance(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dat = Path(self.tmp.name) / "tiny4.dat"
        rows = lambda m: "\n".join(" ".join(map(str, r)) for r in m)  # noqa: E731
        self.dat.write_text(f"4\n\n{rows(A)}\n\n{rows(B)}\n")
        self.bks = min(cost(A, B, list(p)) for p in itertools.permutations(range(4)))
        self.identity_cost = cost(A, B, [0, 1, 2, 3])
        self.task = Task(id="qaplib-tiny4", split="dev", text="QAP facility layout, family tiny, n=4",
                         payload={"name": "tiny4", "path": str(self.dat), "n": 4, "bks": self.bks})

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def version(self, **config) -> SimpleNamespace:
        return SimpleNamespace(version=7, prompts={}, playbooks={"solver": ["Print a valid permutation early."]},
                               config={"attempts": 3, "time_limit_s": 2, **config})


class VerifierTest(TinyInstance):
    def test_identity_is_worse_than_bks_for_this_instance(self) -> None:
        self.assertGreater(self.identity_cost, self.bks)

    def test_one_based_and_zero_based_give_the_same_exact_cost(self) -> None:
        one = verify("tiny4", "PERMUTATION: 2 1 4 3", bks=self.bks, dat_path=self.dat)
        zero = verify("tiny4", "noise\nPERMUTATION: 1 0 3 2", bks=self.bks, dat_path=self.dat)
        self.assertEqual(one["status"], "ok")
        self.assertEqual(one["cost"], zero["cost"])
        self.assertEqual(one["cost"], cost(A, B, [1, 0, 3, 2]))
        self.assertAlmostEqual(one["score"], self.bks / one["cost"])
        self.assertAlmostEqual(one["gap_pct"], (one["cost"] - self.bks) / self.bks * 100)

    def test_invalid_permutations_are_rejected_with_zero_score(self) -> None:
        for bad in ("PERMUTATION: 1 1 2 3", "PERMUTATION: 1 2 3", "PERMUTATION: 1 2 3 5", "no numbers"):
            v = verify("tiny4", bad, bks=self.bks, dat_path=self.dat)
            self.assertEqual(v["status"], "invalid", bad)
            self.assertEqual(v["score"], 0.0)

    @unittest.skipUnless(HAVE_DATA, "instances not fetched; run python -m benchmarks.qaplib.fetch")
    def test_loader_reproduces_bks_from_sln_permutations(self) -> None:
        index = {e["name"]: e for e in json.loads((INSTANCES / "index.json").read_text())["instances"]}
        for name, convention in (("nug20", "direct"), ("kra30a", "inverse")):
            got, value = check_solution(INSTANCES / f"{name}.dat", INSTANCES / f"{name}.sln")
            self.assertEqual(got, convention, name)
            self.assertEqual(value, index[name]["bks"])
        self.assertEqual(parse_dat(INSTANCES / "nug20.dat")[0], 20)


class SplitTest(unittest.TestCase):
    entries = [{"name": f"{fam}{i}", "family": fam, "n": 10 + i, "sha256": f"h{fam}{i}"}
               for fam in ("aa", "bb", "cc") for i in range(5)]

    def test_split_is_deterministic_and_40_20_40(self) -> None:
        first, second = make_splits(self.entries), make_splits(list(reversed(self.entries)))
        self.assertEqual(first, second)
        self.assertEqual([len(first[k]) for k in ("train", "dev", "heldout")], [6, 3, 6])
        self.assertEqual(sorted(sum(first.values(), [])), sorted(e["name"] for e in self.entries))

    def test_heldout_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            splits = make_splits(self.entries)
            self.assertTrue(freeze(splits, self.entries, path))
            self.assertFalse(freeze(splits, self.entries, path))
            changed = {**splits, "heldout": splits["heldout"][1:] + splits["train"][:1]}
            with self.assertRaises(FrozenSplitError):
                freeze(changed, self.entries, path)
            self.assertEqual(json.loads(path.read_text())["heldout"], splits["heldout"])


class RunTeamTest(TinyInstance):
    def test_best_of_attempts_and_feedback_in_second_prompt(self) -> None:
        llm = ScriptedLLM({"solver": [WEAK, BRUTE]})
        r = run_team(self.version(), None, self.task, llm)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.passed)
        self.assertEqual(r.iterations, 2)
        self.assertEqual(r.calls, 2)
        self.assertEqual(len(r.failures), 1)
        self.assertTrue(r.failures[0].startswith("solver: tiny4 gap "), r.failures[0])
        self.assertIn("with identity", r.failures[0])
        second = llm.calls_for("solver")[1]["messages"]
        self.assertIn("Attempt 1 result", second[-1]["content"])
        self.assertIn(f"cost {self.identity_cost} vs BKS {self.bks}", second[-1]["content"])
        self.assertEqual(len(llm.calls_for("solver")[0]["messages"]), 2)
        proposal = json.loads(r.notes)["proposals"][0]
        self.assertIn("brute force", proposal["text"])
        self.assertEqual(proposal["evidence"][0]["run_id"], "qaplib-tiny4-7")
        self.assertEqual(proposal["evidence"][0]["measurement"], "gap=0.00%")
        self.assertGreater(r.prompt_tokens, 0)

    def test_crash_is_reported_and_weak_answer_is_kept(self) -> None:
        llm = ScriptedLLM({"solver": [CRASH, WEAK]})
        r = run_team(self.version(), None, self.task, llm)
        self.assertEqual(r.iterations, 3)
        self.assertEqual(r.failures[0], "solver: tiny4 runtime_error NameError")
        self.assertAlmostEqual(r.score, self.bks / self.identity_cost)
        self.assertFalse(r.passed)
        self.assertIn("NameError", llm.calls_for("solver")[1]["messages"][-1]["content"])

    def test_timeout_kills_the_program(self) -> None:
        llm = ScriptedLLM({"solver": [SLEEP]})
        r = run_team(self.version(attempts=1, time_limit_s=0.2), None, self.task, llm)
        self.assertEqual(r.score, 0.0)
        self.assertIn("timeout", r.failures[0])
        self.assertEqual(r.notes, "")
        self.assertLess(r.wall_s, 15)

    def test_program_does_not_see_api_keys(self) -> None:
        os.environ["CRUSOE_API_KEY"] = "test-not-a-key"
        try:
            r = run_team(self.version(attempts=1), None, self.task, ScriptedLLM({"solver": [ENV_CHECK]}))
        finally:
            del os.environ["CRUSOE_API_KEY"]
        self.assertGreater(r.score, 0.0)

    def test_brain_context_and_playbook_reach_the_system_prompt(self) -> None:
        brain, llm = FakeBrain(), ScriptedLLM({"solver": [BRUTE]})
        r = run_team(self.version(), brain, self.task, llm)
        system = llm.calls_for("solver")[0]["messages"][0]["content"]
        self.assertIn("1. Print a valid permutation early.", system)
        self.assertIn("robust tabu search beats 2-opt", system)
        self.assertEqual(brain.reads, [("solver", self.task.text)])
        self.assertEqual(r.brain_ids_read, ["abc123"])
        self.assertEqual(r.iterations, 1)

    def test_thinking_and_model_per_role_reach_the_llm(self) -> None:
        llm = ScriptedLLM({"solver": [BRUTE]})
        run_team(self.version(thinking={"solver": False}, models={"solver": "m-solver"}), None, self.task, llm)
        call = llm.calls_for("solver")[0]
        self.assertIs(call["thinking"], False)
        self.assertEqual(call["model"], "m-solver")
        unset = ScriptedLLM({"solver": [BRUTE]})
        run_team(self.version(), None, self.task, unset)
        self.assertIsNone(unset.calls_for("solver")[0]["thinking"])

    def test_unchecked_solver_knobs_are_clamped_or_defaulted(self) -> None:
        many = run_team(self.version(attempts=50), None, self.task, ScriptedLLM({"solver": [WEAK]}))
        self.assertEqual(many.calls, 6)
        garbage = run_team(self.version(attempts="lots"), None, self.task, ScriptedLLM({"solver": [WEAK]}))
        self.assertEqual(garbage.calls, 3)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@unittest.skipUnless(HAVE_SPLITS, "instances or reports/qaplib_baseline.json missing")
class EvolveCaseStudyTest(unittest.TestCase):
    """The QAPLIB case study plugs into the unchanged Evolver through run_team."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_task_selection_is_seeded_disjoint_and_small(self) -> None:
        first, again = select_tasks((6, 4, 8), seed=0), select_tasks((6, 4, 8), seed=0)
        self.assertEqual([[t.id for t in s] for s in first], [[t.id for t in s] for s in again])
        self.assertEqual([len(s) for s in first], [6, 4, 8])
        ids = [{t.id for t in s} for s in first]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        frozen = load_splits()
        for split, tasks in zip(("train", "dev", "heldout"), first):
            self.assertTrue(all(t.payload["n"] <= 50 and t.payload["name"] in frozen[split] for t in tasks))

    def test_offline_evolve_accepts_local_search_over_random(self) -> None:
        code = quiet(main, ["--offline", "--out", str(self.out), "--split", "2", "2", "2", "--generations", "1",
                            "--workers", "2"])
        self.assertEqual(code, 0)
        history = json.loads((self.out / "reports" / "before_after.json").read_text())["history"]
        self.assertEqual(history[0]["accepted"], "playbook_add")
        rows = json.loads((self.out / "reports" / "qaplib_rows.json").read_text())
        self.assertEqual([r["config"] for r in rows[:3]], ["v0 + empty brain (cold)", "v0 + final brain", "v1 + final brain"])
        self.assertGreater(rows[2]["mean_score"], rows[0]["mean_score"] + 0.1)
        self.assertLess(rows[2]["mean_gap_pct"], rows[0]["mean_gap_pct"])
        self.assertTrue(rows[3]["config"].startswith("2-swap baseline"))
        self.assertEqual(rows[3]["n"], 2)
        self.assertIn("2-swap baseline", (self.out / "reports" / "vs_2swap.md").read_text())

    def test_heldout_is_untouched_until_final_report(self) -> None:
        train, dev, heldout = select_tasks((2, 2, 2))
        llm, recorder = offline_llm(), Recorder()
        args = build_parser().parse_args(["--offline", "--workers", "2"])
        evolver = make_evolver(llm, args, self.out, qaplib_v0(), recorder)
        quiet(evolver.run, 1, train, dev, heldout)
        held_ids, held_names = {t.id for t in heldout}, [t.payload["name"] for t in heldout]
        self.assertTrue(recorder.order)
        self.assertFalse(held_ids & set(recorder.order))
        reviewer_text = json.dumps([c["messages"] for c in llm.calls_for("reviewer")])
        self.assertTrue(reviewer_text)
        for name in held_names:
            self.assertNotIn(name, reviewer_text)
        rows = final_rows(evolver, recorder, heldout)
        self.assertTrue(held_ids <= set(recorder.order))
        self.assertTrue(all(r["mean_gap_pct"] is not None for r in rows))

    def test_min_gain_flag_reaches_the_gate(self) -> None:
        default = make_evolver(ScriptedLLM({}), build_parser().parse_args(["--offline"]), self.out / "a", qaplib_v0())
        self.assertEqual(default.gate.min_gain, 0.003)
        args = build_parser().parse_args(["--offline", "--min-gain", "0.02"])
        evolver = make_evolver(ScriptedLLM({}), args, self.out / "b", qaplib_v0())
        self.assertEqual(evolver.gate.min_gain, 0.02)
        parent = [RunResult(task_id=f"t{i}", version=0, score=0.98, passed=True) for i in range(2)]
        small = [RunResult(task_id=f"t{i}", version=1, score=0.99, passed=True) for i in range(2)]
        self.assertFalse(evolver.gate.decide(parent, small)["accepted"])
        self.assertTrue(default.gate.decide(parent, small)["accepted"])


class DcInstancesTest(unittest.TestCase):
    """GPU rack placement: data-center QAP instances in QAPLIB's format, scored by the unchanged verifier."""

    @classmethod
    def setUpClass(cls) -> None:
        from benchmarks.qaplib import dc_instances as dc

        cls.dc = dc
        cls.index, cls.splits = dc.load_index(), dc.load_splits()
        cls.names = [n for s in ("train", "dev", "heldout") for n in cls.splits[s]]

    def test_generation_is_deterministic_and_matches_the_stored_files(self) -> None:
        import hashlib

        e = self.index[self.splits["heldout"][0]]
        first, again = self.dc.build(e["seed"], e["tier"]), self.dc.build(e["seed"], e["tier"])
        self.assertEqual(first, again)
        text = self.dc.dat_text(first["n"], first["A"], first["B"])
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), e["sha256"])
        self.assertEqual(self.dc.choose_seeds(), {k: [tuple(x) for x in v] for k, v in self.splits["seeds"].items()})

    def test_matrices_are_symmetric_non_negative_with_zero_diagonal(self) -> None:
        lo, hi = self.dc.N_RANGE
        for name in self.names:
            n, a, b = parse_dat(self.dc.DC_DIR / f"{name}.dat")
            self.assertTrue(lo <= n <= hi, name)
            for m in (a, b):
                self.assertTrue(all(m[i][j] == m[j][i] and m[i][j] >= 0 for i in range(n) for j in range(n)), name)
                self.assertTrue(all(m[i][i] == 0 for i in range(n)), name)
            self.assertTrue(any(v > 0 for row in a for v in row) and any(v > 0 for row in b for v in row))

    def test_verify_scores_a_known_permutation_exactly(self) -> None:
        name = self.splits["heldout"][0]
        e, dat = self.index[name], self.dc.DC_DIR / f"{name}.dat"
        n, a, b = parse_dat(dat)
        perm = list(range(n))[::-1]
        by_hand = sum(a[i][j] * b[perm[i]][perm[j]] for i in range(n) for j in range(n))
        v = verify(name, perm, bks=e["bks"], dat_path=dat)
        self.assertEqual(v["cost"], by_hand)
        self.assertAlmostEqual(v["score"], e["bks"] / by_hand)
        for stored in (e["two_swap"], e["reference"]):  # the stored answers reproduce their stored costs
            self.assertEqual(verify(name, stored["perm"], bks=e["bks"], dat_path=dat)["cost"], stored["cost"])
        self.assertEqual(e["bks"], min(e["two_swap"]["cost"], e["reference"]["tabu_cost"]))

    def test_delta_matrix_matches_a_recomputed_swap(self) -> None:
        import numpy as np

        _, a, b = parse_dat(self.dc.DC_DIR / f"{self.names[0]}.dat")
        a, b = np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.permutation(len(a))
            r, s = rng.choice(len(a), 2, replace=False)
            q = p.copy()
            q[[r, s]] = q[[s, r]]
            change = int((a * b[q][:, q]).sum() - (a * b[p][:, p]).sum())
            self.assertEqual(int(self.dc.delta_matrix(a, b[p][:, p])[r, s]), change)

    def test_splits_are_disjoint_by_name_and_seed(self) -> None:
        from donetwin import halls  # dc_instances put donetwin/ on sys.path

        self.assertEqual([len(self.splits[s]) for s in ("train", "dev", "heldout")], list(self.dc.SPLIT_SIZES))
        names = [set(self.splits[s]) for s in ("train", "dev", "heldout")]
        seeds = [{x[0] for x in self.splits["seeds"][s]} for s in ("train", "dev", "heldout")]
        for group in (names, seeds):
            self.assertFalse(group[0] & group[1] or group[0] & group[2] or group[1] & group[2])
        self.assertTrue(seeds[2] <= set(halls.heldout_seeds()))
        self.assertTrue(all(s < 10000 for s in seeds[0] | seeds[1]))
        tasks = self.dc.load_tasks("heldout")
        self.assertEqual({k for t in tasks for k in t.payload}, {"name", "path", "n", "bks"})
        self.assertTrue(all(t.text.startswith("GPU rack placement in hall ") for t in tasks))

    def test_evolve_script_wires_the_dc_splits_and_baseline(self) -> None:
        from scripts.run_qaplib_evolve import baseline_row, select_dc_tasks

        self.assertEqual(build_parser().parse_args(["--offline"]).instances, "qaplib")
        self.assertEqual(build_parser().parse_args(["--offline", "--instances", "dc"]).split, None)
        train, dev, heldout = select_dc_tasks()
        self.assertEqual([len(train), len(dev), len(heldout)], list(self.dc.SPLIT_SIZES))
        self.assertTrue(all(self.dc.is_dc(t.payload["name"]) for t in train + dev + heldout))
        row = baseline_row(heldout)
        expected = sum(self.index[t.payload["name"]]["bks"] / self.index[t.payload["name"]]["two_swap"]["cost"]
                       for t in heldout) / len(heldout)
        self.assertEqual(row["n"], len(heldout))
        self.assertAlmostEqual(row["mean_score"], round(expected, 4))

    def test_verifier_dc_route_scores_against_the_reference(self) -> None:
        from benchmarks.qaplib.verifier_api import Rejected, evaluate_dc

        held = self.splits["heldout"]
        two = [{"instance": n, "permutation": self.index[n]["two_swap"]["perm"]} for n in held]
        res = evaluate_dc({"submission": {"solutions": two}, "split": "heldout"})
        self.assertTrue(res["baseline"]["is_baseline"])
        self.assertAlmostEqual(res["score"], res["baseline"]["score"])
        self.assertTrue(res["feedback"].startswith("This is the baseline"))
        ref = [{"instance": n, "permutation": self.index[n]["reference"]["perm"]} for n in held]
        self.assertAlmostEqual(evaluate_dc({"submission": {"solutions": ref}, "split": "heldout"})["score"], 1.0)
        partial = evaluate_dc({"submission": {"solutions": ref[1:]}, "split": "heldout"})
        self.assertAlmostEqual(partial["score"], (len(held) - 1) / len(held))
        with self.assertRaises(Rejected):
            evaluate_dc({"submission": {"solutions": ref}, "split": "test"})
        with self.assertRaises(Rejected):
            evaluate_dc({"submission": {"instance": "nug20", "permutation": [0]}})


if __name__ == "__main__":
    unittest.main()
