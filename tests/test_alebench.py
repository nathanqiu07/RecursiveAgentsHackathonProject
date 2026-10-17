"""Offline tests for the ALE-Bench AHC039 case study: real official scorer, scripted LLM, no network."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.alebench import scorer, splits  # noqa: E402
from benchmarks.alebench.team_runner import ROLE, alebench_v0, run_team  # noqa: E402
from scripts.run_alebench_evolve import offline_llm  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.evolve import Evolver, Gate  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402

HAVE_DATA = (scorer.PROBLEM_DIR / "data.json").is_file() and scorer.REFERENCE_JSON.is_file()
SKIP = "AHC039 data not fetched; run .venv/bin/python -m benchmarks.alebench.fetch"


def program(code: str) -> str:
    return f"TECHNIQUE: test\n```python\n{code}\n```"


def version(**config):
    v = alebench_v0()
    v.config.update({"attempts": 1, **config})
    return v


@unittest.skipUnless(HAVE_DATA, SKIP)
class ScorerTest(unittest.TestCase):
    def test_reference_greedy_reproduces_known_case_score(self):
        rows = scorer.evaluate(scorer.GREEDY.read_text(), [0], 2.0)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["score"], 1521)  # official visualizer score of the greedy on seed 0
        self.assertEqual(scorer.reference_scores([0]), {0: 1521})
        self.assertEqual(scorer.evaluate(scorer.NAIVE.read_text(), [0], 2.0)[0]["score"], 1)

    def test_rank_and_performance_from_contest_tables(self):
        top = scorer.rank_performance(0)["top_human_total"]
        self.assertEqual(scorer.rank_performance(top)["rank"], 1)
        self.assertGreater(scorer.rank_performance(top)["performance"],
                           scorer.rank_performance(sum(scorer.reference_scores(splits.private_seeds()).values()))[
                               "performance"])

    def test_child_environment_carries_no_keys(self):
        code = "import os\nprint(' '.join(sorted(os.environ)))\n"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"CRUSOE_API_KEY": "sekrit", "GITHUB_TOKEN": "t", "OTHER_KEY": "k"}):
            work = Path(tmp)
            (work / "p.py").write_text(code)
            r = scorer.run_case(work / "p.py", 0, 2.0, work)
            names = (work / "out_0.txt").read_text().split()
        self.assertEqual(r["status"], "ran")
        self.assertFalse([n for n in names if n.endswith(("_KEY", "_TOKEN")) or n.startswith("CRUSOE")], names)


@unittest.skipUnless(HAVE_DATA, SKIP)
class SplitsTest(unittest.TestCase):
    def test_splits_are_disjoint_and_sized(self):
        train, dev, heldout = splits.load_tasks()
        self.assertEqual([len(train), len(dev), len(heldout)], [4, 3, 5])
        self.assertEqual(train[0].payload["seeds"], splits.public_seeds())
        seeds = [set(s for t in part for s in t.payload["seeds"]) for part in (train, dev, heldout)]
        self.assertEqual([len(s) for s in seeds], [20, 30, 150])
        self.assertFalse(seeds[0] & seeds[1] or seeds[0] & seeds[2] or seeds[1] & seeds[2])
        self.assertEqual(seeds[2], set(splits.private_seeds()))
        for t in heldout:
            self.assertFalse(set(t.payload["feedback_seeds"]) & seeds[2])
            self.assertFalse(any(str(s) in t.text for s in t.payload["seeds"]))

    def test_heldout_seeds_and_scores_never_reach_prompts(self):
        _, _, heldout = splits.load_tasks()
        task = heldout[0]
        llm = ScriptedLLM({ROLE: [program(scorer.NAIVE.read_text()), program(scorer.GREEDY.read_text())]})
        with tempfile.TemporaryDirectory() as tmp:
            r = run_team(version(attempts=2), SharedBrain(tmp), task, llm)
        self.assertAlmostEqual(r.score, 1.0)  # greedy chosen on public seeds, then scored on the 30 private ones
        prompts = "\n".join(m["content"] for c in llm.calls for m in c["messages"])
        self.assertTrue(prompts)
        for s in splits.private_seeds():
            self.assertNotIn(str(s), prompts)
        shown = {int(x) for x in re.findall(r"case (\d+):", prompts)}
        self.assertEqual(shown, set(splits.public_seeds()))
        self.assertNotIn("proposals", json.loads(r.notes))  # nothing about held-out runs goes to the brain


@unittest.skipUnless(HAVE_DATA, SKIP)
class RunTeamTest(unittest.TestCase):
    def run_reply(self, reply: str, **config):
        train, _, _ = splits.load_tasks()
        with tempfile.TemporaryDirectory() as tmp:
            return run_team(version(**config), SharedBrain(tmp), train[0], ScriptedLLM({ROLE: [reply]}))

    def test_garbage_program_scores_zero_without_crash(self):
        r = self.run_reply(program("print('not a polygon')"))
        self.assertEqual(r.score, 0.0)
        self.assertIn("solver: wrong_answer", r.failures)
        self.assertIn("solver: no_improvement", r.failures)

    def test_error_categories(self):
        self.assertIn("solver: runtime_error", self.run_reply(program("raise ValueError('x')")).failures)
        self.assertIn("solver: no_code", self.run_reply("I would use simulated annealing.").failures)
        slow = self.run_reply(program("import time\ntime.sleep(5)"), time_limit=0.3)
        self.assertIn("solver: timeout", slow.failures)
        self.assertEqual(slow.score, 0.0)

    def test_time_limit_knob_cannot_exceed_contest_limit(self):
        from benchmarks.alebench.team_runner import _settings
        self.assertEqual(_settings(version(time_limit=60))["time_limit"], scorer.contest_time_limit())


@unittest.skipUnless(HAVE_DATA, SKIP)
class OfflineEvolveTest(unittest.TestCase):
    def test_offline_evolve_accepts_the_better_program(self):
        train, dev, heldout = splits.load_tasks()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ev = Evolver(run_team, offline_llm(), "reviewer-model", SharedBrain(out / "brain"), out / "versions",
                         max_workers=2, initial=alebench_v0(), gate=Gate(min_gain=0.01), reports_dir=out / "reports")
            final = ev.run(1, train[:1], dev[:1], heldout[:1])
        self.assertEqual(final.version, 1)
        self.assertEqual(ev.history[0]["accepted"], "playbook_add")
        self.assertLess(ev.history[0]["dev_mean"], 0.01)  # v0 sends the 1x1 square


if __name__ == "__main__":
    unittest.main()
