"""Inference serving cost (layer 4): deterministic workload, exact checker, disjoint splits, run_team's knobs,
and the offline Evolver accepting a cheaper version at equal accuracy."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.inference_opt.team_runner import run_team  # noqa: E402
from benchmarks.inference_opt.workload import (  # noqa: E402
    KW, QTYPES, grade, is_correct, load_tasks, make_questions)
from scripts.run_inference_opt import inference_v0, main, offline_llm  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.crusoe import NANO, SUPER  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402


class WorkloadTest(unittest.TestCase):
    def test_generator_is_deterministic(self):
        self.assertEqual(make_questions(100, "small"), make_questions(100, "small"))
        self.assertNotEqual(make_questions(100, "small"), make_questions(101, "small"))

    def test_every_task_has_three_questions_of_each_type(self):
        for split in load_tasks():
            for t in split:
                qs = t.payload["questions"]
                self.assertEqual(len(qs), 12)
                self.assertEqual({q["qtype"] for q in qs}, set(QTYPES))
                self.assertEqual(len({q["id"] for q in qs}), 12)

    def test_splits_are_disjoint_with_sizes_4_3_4(self):
        train, dev, heldout = load_tasks()
        self.assertEqual((len(train), len(dev), len(heldout)), (4, 3, 4))
        ids = [{t.id for t in s} for s in (train, dev, heldout)]
        seeds = [{t.payload["seed"] for t in s} for s in (train, dev, heldout)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            self.assertFalse(ids[a] & ids[b])
            self.assertFalse(seeds[a] & seeds[b])

    def test_answers_match_an_independent_recomputation(self):
        for q in make_questions(100, "small"):
            if q["qtype"] == "busway_left":
                text = q["text"]
                cap = float(text.split("capacity_kw=")[1].split(",")[0])
                racks = text.split("already in row")[1].split(": ", 1)[1].split(". How")[0].split(", ")
                load = sum(KW[r.split(" ")[1]] for r in racks)
                self.assertAlmostEqual(q["answer"], cap * 0.8 - load)

    def test_checker_grades_known_answers_right_and_wrong(self):
        qs = make_questions(100, "small")
        right = {q["id"]: q["answer"] for q in qs}
        self.assertEqual(grade(qs, right)["accuracy"], 1.0)
        self.assertEqual(grade(qs, {k: str(v) for k, v in right.items()})["accuracy"], 1.0)  # numeric strings ok
        wrong = {q["id"]: q["answer"] + q["tol"] + 1.0 for q in qs}
        self.assertEqual(grade(qs, wrong)["accuracy"], 0.0)
        self.assertEqual(grade(qs, {})["accuracy"], 0.0)
        q = next(q for q in qs if q["qtype"] == "hose_slack")
        self.assertTrue(is_correct(q, q["answer"] + 0.04))
        self.assertFalse(is_correct(q, q["answer"] + 0.06))
        self.assertFalse(is_correct(q, True))
        fit = next(q for q in qs if q["qtype"] == "rack_fit")
        self.assertFalse(is_correct(fit, fit["answer"] + 0.5))


class RunTeamTest(unittest.TestCase):
    def setUp(self):
        self.tasks = load_tasks()
        self.task = self.tasks[0][0]
        self.llm = offline_llm([t for s in self.tasks for t in s])

    def run_with(self, **config):
        v = inference_v0()
        v.config.update(config)
        with tempfile.TemporaryDirectory() as tmp:
            return run_team(v, SharedBrain(tmp), self.task, self.llm)

    def test_v0_is_super_reasoning_on_one_call_per_question(self):
        r = self.run_with()
        self.assertEqual((r.score, r.passed, r.calls), (1.0, True, 12))
        self.assertTrue(all(c["model"] == SUPER and c["thinking"] is True and c["max_tokens"] == 4000
                            for c in self.llm.calls))
        self.assertEqual({c["role"] for c in self.llm.calls}, set(QTYPES))
        self.assertEqual(r.solution["solution"]["seed"], self.task.payload["seed"])
        self.assertEqual(len(r.solution["solution"]["answers"]), 12)

    def test_batching_routing_and_thinking_knobs(self):
        r = self.run_with(batch_size=3, models={"cdu_headroom": NANO}, thinking={"cdu_headroom": False})
        self.assertEqual((r.score, r.calls), (1.0, 4))
        cdu = self.llm.calls_for("cdu_headroom")
        self.assertEqual((cdu[0]["model"], cdu[0]["thinking"]), (NANO, False))
        self.assertEqual(self.llm.calls_for("rack_fit")[0]["model"], SUPER)

    def test_bad_json_and_wrong_answers_are_failures_by_type(self):
        self.llm = ScriptedLLM({"rack_fit": ["no json here"], "hose_slack": ['{"answers": {"q1": -999}}']},
                               default='{"answers": {}}')
        r = self.run_with()
        self.assertIn("bad_json:rack_fit", r.failures)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.passed)


    def test_a_failed_call_is_an_infra_failure_not_a_wrong_answer(self):
        class Flaky(ScriptedLLM):
            def chat(self, model, messages, **kw):
                if kw.get("role") == "hose_slack":
                    raise RuntimeError("429 max_requests_limit")
                return super().chat(model, messages, **kw)

        truth = self.llm
        self.llm = Flaky(lambda role, messages: truth.chat("m", messages, role=role).text)
        r = self.run_with()
        self.assertTrue(any(f.startswith("infra:hose_slack") for f in r.failures))
        self.assertFalse(any(f.startswith("wrong:") for f in r.failures))
        self.assertAlmostEqual(r.score, 9 / 12)


class VerifierApiTest(unittest.TestCase):
    def test_split_submission_is_regraded_from_seeds_and_missing_tasks_score_zero(self):
        from benchmarks.inference_opt.verifier_api import Rejected, evaluate

        held = load_tasks()[2]
        full = [{"seed": t.payload["seed"], "tier": t.payload["tier"],
                 "answers": {q["id"]: q["answer"] for q in t.payload["questions"]}} for t in held]
        self.assertEqual(evaluate({"submission": {"tasks": full}, "split": "heldout"})["score"], 1.0)
        self.assertEqual(evaluate({"submission": {"tasks": full[:1]}, "split": "heldout"})["score"], 0.25)
        with self.assertRaises(Rejected):
            evaluate({"submission": {"tasks": [{"seed": 100, "tier": "small", "answers": {}}]}, "split": "heldout"})


class ReviewerBudgetTest(unittest.TestCase):
    def test_only_the_reviewer_call_gets_the_larger_budget_and_format_hint(self):
        from scripts.run_inference_opt import REVIEW_FORMAT, ReviewerBudget

        inner = ScriptedLLM({}, default="{}")
        llm = ReviewerBudget(inner)
        llm.chat("m", [{"role": "user", "content": "review"}], role="reviewer")
        llm.chat("m", [{"role": "user", "content": "work"}], role="rack_fit", max_tokens=300, thinking=True)
        rev, work = inner.calls
        self.assertEqual((rev["max_tokens"], rev["thinking"]), (8000, False))
        self.assertTrue(rev["messages"][-1]["content"].endswith(REVIEW_FORMAT))
        self.assertEqual((work["max_tokens"], work["thinking"], work["messages"][-1]["content"]), (300, True, "work"))
        self.assertIs(llm.calls, inner.calls)


class OfflineEvolveTest(unittest.TestCase):
    def test_offline_evolve_accepts_a_cheaper_version_at_equal_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--offline", "--generations", "1", "--out", tmp]), 0)
            report = json.loads((Path(tmp) / "reports" / "heldout_cost_latency.json").read_text())
            self.assertEqual(report["final_version"], 1)
            self.assertEqual(report["final_config"]["batch_size"], 3)
            v0, final = report["rows"]
            self.assertEqual(v0["accuracy"], final["accuracy"])
            self.assertLess(final["tokens_per_task"], 0.85 * v0["tokens_per_task"])
            ledger = [json.loads(x) for x in (Path(tmp) / "versions" / "ledger.jsonl").read_text().splitlines()]
            self.assertTrue(any(r.get("accepted_by") == "cost" for r in ledger))
            self.assertTrue(any(str(r.get("reason", "")).startswith("invalid") for r in ledger))
            sols = list((Path(tmp) / "solutions").rglob("*.json"))
            self.assertTrue(sols)
            self.assertEqual(json.loads(sols[0].read_text())["case_study"], "inference_opt")


if __name__ == "__main__":
    unittest.main()
