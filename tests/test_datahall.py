"""Data-hall domain on Rikin's DoneTwin scorer: splits, greedy baseline, failure categories, tools, brain reads, bad JSON."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team.brain import SharedBrain  # noqa: E402
from team.datahall import (  # noqa: E402
    TOOL_CATALOG, datahall_run_team, datahall_v0, halls, load_tasks, make_task, score_payload,
)
from team.fake_llm import ScriptedLLM  # noqa: E402

TRAIN, DEV, HELDOUT = load_tasks(seed=0)


def layout_llm(layout: dict | str) -> ScriptedLLM:
    text = layout if isinstance(layout, str) else json.dumps(layout)
    return ScriptedLLM({"Layout": [text], "Planner": ['{"total_racks": 1}'], "Checker": ['{"fix": "drop R_DUP"}']})


class DataHallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.brain = SharedBrain(Path(self.tmp.name) / "brain")
        self.task = TRAIN[0]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_splits_are_sized_disjoint_seeded_and_mix_tiers(self) -> None:
        self.assertEqual([len(s) for s in (TRAIN, DEV, HELDOUT)], [12, 8, 15])
        ids = [{t.id for t in s} for s in (TRAIN, DEV, HELDOUT)]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertFalse({t.payload["seed"] for t in TRAIN} & {t.payload["seed"] for t in DEV})
        again = load_tasks(seed=0)
        self.assertEqual([t.id for s in again for t in s], [t.id for s in (TRAIN, DEV, HELDOUT) for t in s])
        self.assertNotEqual([t.id for t in load_tasks(seed=1)[0]], [t.id for t in TRAIN])
        self.assertEqual({t.payload["tier"] for t in TRAIN}, {"small", "medium", "large"})

    def test_no_heldout_seed_in_train_or_dev(self) -> None:
        held = set(halls.heldout_seeds())
        self.assertTrue(all(t.payload["seed"] in held for t in HELDOUT))
        self.assertFalse(held & {t.payload["seed"] for t in TRAIN + DEV})
        self.assertTrue(all(0 <= t.payload["seed"] <= 9999 for t in TRAIN + DEV))

    def test_greedy_layout_scores_feasible_and_matches_baseline(self) -> None:
        greedy = self.task.payload["greedy_layout"]
        res = score_payload(self.task.payload["hall"], greedy)
        self.assertTrue(res["feasible"])
        self.assertGreater(res["score"], 0)
        r = datahall_run_team(datahall_v0(), self.brain, self.task, layout_llm(greedy))
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.score, self.task.payload["greedy_score"] / 100, places=4)
        self.assertEqual(r.iterations, 1)
        self.assertEqual(r.calls, 2)  # Planner + one feasible Layout attempt

    def test_violation_scores_zero_with_check_name_failures_and_fix_hints_in_notes(self) -> None:
        racks = self.task.payload["greedy_layout"]["racks"]
        bad = {"racks": racks + [{**racks[0], "id": "R_DUP"}]}
        r = datahall_run_team(datahall_v0(), self.brain, self.task, layout_llm(bad))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.passed)
        self.assertIn("no_collision", r.failures)
        self.assertEqual(r.iterations, 3)
        notes = json.loads(r.notes)
        self.assertTrue(any(v["check"] == "no_collision" and v["fix"] for v in notes["violations"]))
        self.assertTrue(any("0.6 m apart" in p["text"] for p in notes["proposals"]))

    def test_feedback_reaches_layout_and_recovery_keeps_the_best_attempt(self) -> None:
        racks = self.task.payload["greedy_layout"]["racks"]
        llm = ScriptedLLM({"Layout": [json.dumps({"racks": racks + [{**racks[0], "id": "R_DUP"}]}),
                                      json.dumps({"racks": racks})]})
        r = datahall_run_team(datahall_v0(), self.brain, self.task, llm)
        self.assertTrue(r.passed)
        self.assertEqual(r.iterations, 2)
        second = llm.calls_for("Layout")[1]["messages"][-1]["content"]
        self.assertIn("Verifier feedback", second)
        self.assertIn("Checker:", second)

    def test_malformed_reply_yields_bad_json_without_raising(self) -> None:
        r = datahall_run_team(datahall_v0(), self.brain, self.task, layout_llm("racks go here, trust me"))
        self.assertEqual(r.score, 0.0)
        self.assertIn("bad_json", r.failures)
        self.assertEqual(r.iterations, 3)

    def test_llm_error_is_recorded_not_raised(self) -> None:
        def boom(role: str, messages: list) -> str:
            raise ConnectionError("network down")
        r = datahall_run_team(datahall_v0(), self.brain, self.task, ScriptedLLM(boom))
        self.assertEqual(r.score, 0.0)
        self.assertIn("llm_error:Layout", r.failures)

    def test_brain_ids_read_are_recorded(self) -> None:
        ev = [{"run_id": "seed", "measurement": "score=0"}]
        self.brain.propose([{"op": "add", "kind": "lesson", "roles": ["Layout"], "evidence": ev,
                             "text": "Give each rack the nearest CDU with hose reach and spare capacity."}], "seed")
        (entry_id,) = [e.id for e in self.brain.top(5)]
        llm = layout_llm(self.task.payload["greedy_layout"])
        r = datahall_run_team(datahall_v0(), self.brain, self.task, llm)
        self.assertIn(entry_id, r.brain_ids_read)
        self.assertIn("nearest CDU", llm.calls_for("Layout")[0]["messages"][0]["content"])

    def test_tools_are_off_in_v0_except_feedback_and_switch_on_from_config(self) -> None:
        v0 = datahall_v0()
        self.assertEqual(v0.config["tool_catalog"], TOOL_CATALOG)
        llm = layout_llm(self.task.payload["greedy_layout"])
        datahall_run_team(v0, self.brain, self.task, llm)
        brief = llm.calls_for("Layout")[0]["messages"][1]["content"]
        self.assertNotIn("Capacity budget", brief)
        self.assertNotIn("Starting layout", brief)
        for tool in ("capacity_calc", "greedy_seed"):
            v0.config["tools"][tool] = {"enabled": True}
        llm = layout_llm(self.task.payload["greedy_layout"])
        datahall_run_team(v0, self.brain, self.task, llm)
        brief = llm.calls_for("Layout")[0]["messages"][1]["content"]
        self.assertIn("Capacity budget", brief)
        self.assertIn(json.dumps(self.task.payload["greedy_layout"], separators=(",", ":")), brief)

    def test_make_task_is_deterministic(self) -> None:
        a, b = make_task(7, "small", "train"), make_task(7, "small", "train")
        self.assertEqual(a.payload["hall"], b.payload["hall"])
        self.assertEqual(a.payload["greedy_score"], b.payload["greedy_score"])


if __name__ == "__main__":
    unittest.main()
