"""QAPLIB benchmark: loader BKS validation, exact verifier, frozen splits, and run_team's best-of-attempts loop."""
from __future__ import annotations

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
from benchmarks.qaplib.splits import FrozenSplitError, freeze, make_splits  # noqa: E402
from benchmarks.qaplib.team_runner import run_team  # noqa: E402
from benchmarks.qaplib.verify import INSTANCES, cost, parse_dat, verify  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.types import Task  # noqa: E402

HAVE_DATA = (INSTANCES / "nug20.dat").exists() and (INSTANCES / "kra30a.dat").exists()
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


if __name__ == "__main__":
    unittest.main()
