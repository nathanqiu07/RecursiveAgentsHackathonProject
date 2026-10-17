"""Data-hall domain on Rikin's DoneTwin scorer: splits, greedy incumbent hill climb, failure categories, tools,
thinking switch, brain reads, bad JSON."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team.brain import SharedBrain  # noqa: E402
from team.datahall import (  # noqa: E402
    TOOL_CATALOG, compact, datahall_run_team, datahall_v0, halls, library_add, load_tasks, make_task, score_payload,
)
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.types import Reply  # noqa: E402

TRAIN, DEV, HELDOUT = load_tasks(seed=0)


def layout_llm(*layouts: dict | str) -> ScriptedLLM:
    texts = [x if isinstance(x, str) else json.dumps(x) for x in layouts]
    return ScriptedLLM({"Layout": texts, "Planner": ['{"total_racks": 1}'], "Checker": ['{"fix": "drop R_DUP"}']})


def improved(task) -> dict:
    """Greedy with one GB200 upgraded to GB300 on CDU_1: scores above greedy on TRAIN[0]."""
    racks = [dict(r) for r in task.payload["greedy_layout"]["racks"]]
    racks[11].update(model="GB300-NVL72", cdu="CDU_1")
    return {"racks": racks}


class DataHallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.brain = SharedBrain(Path(self.tmp.name) / "brain")
        self.task = TRAIN[0]
        self.greedy = round(self.task.payload["greedy_score"] / 100, 4)  # RunResult.score is rounded to 4 places

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_splits_are_sized_disjoint_seeded_and_mix_tiers_with_at_most_two_large(self) -> None:
        self.assertEqual([len(s) for s in (TRAIN, DEV, HELDOUT)], [8, 6, 10])
        ids = [{t.id for t in s} for s in (TRAIN, DEV, HELDOUT)]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertFalse({t.payload["seed"] for t in TRAIN} & {t.payload["seed"] for t in DEV})
        again = load_tasks(seed=0)
        self.assertEqual([t.id for s in again for t in s], [t.id for s in (TRAIN, DEV, HELDOUT) for t in s])
        self.assertNotEqual([t.id for t in load_tasks(seed=1)[0]], [t.id for t in TRAIN])
        for split in (TRAIN, DEV, HELDOUT):
            self.assertEqual({t.payload["tier"] for t in split}, {"small", "medium", "large"})
            self.assertLessEqual(sum(t.payload["tier"] == "large" for t in split), 2)

    def test_no_heldout_seed_in_train_or_dev(self) -> None:
        held = set(halls.heldout_seeds())
        self.assertTrue(all(t.payload["seed"] in held for t in HELDOUT))
        self.assertFalse(held & {t.payload["seed"] for t in TRAIN + DEV})
        self.assertTrue(all(0 <= t.payload["seed"] <= 9999 for t in TRAIN + DEV))

    def test_layout_sees_greedy_incumbent_and_an_equal_reply_keeps_greedy(self) -> None:
        greedy = self.task.payload["greedy_layout"]
        self.assertTrue(score_payload(self.task.payload["hall"], greedy)["feasible"])
        llm = layout_llm(greedy)
        r = datahall_run_team(datahall_v0(), self.brain, self.task, llm)
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.score, self.greedy, places=4)
        self.assertEqual(r.iterations, 2)
        self.assertEqual(r.calls, 4)  # Planner + 2 Layout + 1 Checker after the non-improving first attempt
        self.assertIn("no_improvement", r.failures)
        brief = llm.calls_for("Layout")[0]["messages"][1]["content"]
        self.assertIn("INCUMBENT (greedy solver)", brief)
        self.assertIn(compact(greedy), brief)
        self.assertIn("Incumbent score breakdown", brief)
        notes = json.loads(r.notes)
        self.assertEqual(notes["greedy_score"], self.task.payload["greedy_score"])
        self.assertEqual(notes["gain_over_greedy"], 0.0)

    def test_improvement_becomes_the_incumbent_and_sets_the_score(self) -> None:
        better = improved(self.task)
        best = score_payload(self.task.payload["hall"], better)["score"]
        self.assertGreater(best, self.task.payload["greedy_score"])
        llm = layout_llm(better, "garbage")
        r = datahall_run_team(datahall_v0(), self.brain, self.task, llm)
        self.assertAlmostEqual(r.score, best / 100, places=4)
        self.assertNotIn("no_improvement", r.failures)
        self.assertIn("INCUMBENT (your attempt 1)", llm.calls_for("Layout")[1]["messages"][1]["content"])
        self.assertEqual(llm.calls_for("Checker"), [])  # accepted attempts need no fix
        notes = json.loads(r.notes)
        self.assertEqual(notes["best_attempt"], 0)
        self.assertGreater(notes["gain_over_greedy"], 0)

    def test_violation_is_rejected_with_check_name_failures_and_score_stays_greedy(self) -> None:
        racks = self.task.payload["greedy_layout"]["racks"]
        bad = {"racks": racks + [{**racks[0], "id": "R_DUP"}]}
        r = datahall_run_team(datahall_v0(), self.brain, self.task, layout_llm(bad))
        self.assertAlmostEqual(r.score, self.greedy, places=4)
        self.assertTrue(r.passed)
        self.assertIn("no_collision", r.failures)
        self.assertEqual(r.iterations, 2)
        notes = json.loads(r.notes)
        self.assertIn("no_collision", notes["attempts"][0]["failures"])
        self.assertFalse(notes["attempts"][0]["accepted"])
        self.assertTrue(any("0.6 m apart" in p["text"] for p in notes["proposals"]))

    def test_feedback_after_a_rejected_attempt_reaches_layout(self) -> None:
        racks = self.task.payload["greedy_layout"]["racks"]
        llm = layout_llm({"racks": racks + [{**racks[0], "id": "R_DUP"}]}, improved(self.task))
        r = datahall_run_team(datahall_v0(), self.brain, self.task, llm)
        self.assertGreater(r.score, self.greedy)
        second = llm.calls_for("Layout")[1]["messages"][-1]["content"]
        self.assertIn("rejected", second)
        self.assertIn("Verifier feedback", second)
        self.assertIn("Checker:", second)

    def test_garbage_reply_is_bad_json_and_still_scores_greedy(self) -> None:
        r = datahall_run_team(datahall_v0(), self.brain, self.task, layout_llm("racks go here, trust me"))
        self.assertAlmostEqual(r.score, self.greedy, places=4)
        self.assertGreaterEqual(r.score, self.greedy)
        self.assertIn("bad_json", r.failures)
        self.assertEqual(r.iterations, 2)

    def test_racks_as_arrays_in_format_key_order_are_accepted(self) -> None:
        r = datahall_run_team(datahall_v0(), self.brain, self.task, layout_llm(compact(improved(self.task))))
        self.assertNotIn("bad_json", r.failures)
        self.assertGreater(r.score, self.greedy)

    def test_reply_at_max_tokens_counts_as_truncated_bad_json(self) -> None:
        class Truncating(ScriptedLLM):
            def chat(self, model, messages, temperature=0.2, max_tokens=2048, json_mode=False, role="", thinking=None):
                reply = super().chat(model, messages, temperature, max_tokens, json_mode, role, thinking)
                return Reply(text=reply.text, model=model, completion_tokens=max_tokens if role == "Layout" else 5)
        llm = Truncating({"Layout": [json.dumps(improved(self.task))]})
        r = datahall_run_team(datahall_v0(), self.brain, self.task, llm)
        self.assertIn("bad_json", r.failures)
        self.assertIn("truncated", r.failures)
        self.assertAlmostEqual(r.score, self.greedy, places=4)

    def test_llm_error_is_recorded_not_raised(self) -> None:
        def boom(role: str, messages: list) -> str:
            raise ConnectionError("network down")
        r = datahall_run_team(datahall_v0(), self.brain, self.task, ScriptedLLM(boom))
        self.assertAlmostEqual(r.score, self.greedy, places=4)
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
        self.assertNotIn("greedy_seed", TOOL_CATALOG)
        other = TRAIN[1]
        library_add(other.payload["hall"], other.payload["greedy_layout"], other.payload["greedy_score"])
        llm = layout_llm(self.task.payload["greedy_layout"])
        datahall_run_team(v0, self.brain, self.task, llm)
        brief = llm.calls_for("Layout")[0]["messages"][1]["content"]
        self.assertNotIn("Capacity budget", brief)
        self.assertNotIn("VERIFIED EXAMPLE", brief)
        for tool in ("capacity_calc", "nearest_examples"):
            v0.config["tools"][tool] = {"enabled": True}
        llm = layout_llm(self.task.payload["greedy_layout"])
        datahall_run_team(v0, self.brain, self.task, llm)
        brief = llm.calls_for("Layout")[0]["messages"][1]["content"]
        self.assertIn("Capacity budget", brief)
        self.assertIn(f"VERIFIED EXAMPLE 1 (other hall {other.id}", brief)

    def test_thinking_is_off_in_v0_and_switches_on_per_role_from_config(self) -> None:
        v0 = datahall_v0()
        self.assertEqual(v0.config["max_tokens"], 6000)
        self.assertEqual(v0.config["max_iters"], 2)
        llm = layout_llm(self.task.payload["greedy_layout"])
        datahall_run_team(v0, self.brain, self.task, llm)
        self.assertTrue(llm.calls and all(c["thinking"] is False for c in llm.calls))
        v0.config["thinking"] = {"Layout": True}
        llm = layout_llm(self.task.payload["greedy_layout"])
        datahall_run_team(v0, self.brain, self.task, llm)
        self.assertTrue(all(c["thinking"] is True for c in llm.calls_for("Layout")))
        self.assertTrue(all(c["thinking"] is False for c in llm.calls_for("Planner")))

    def test_make_task_is_deterministic(self) -> None:
        a, b = make_task(7, "small", "train"), make_task(7, "small", "train")
        self.assertEqual(a.payload["hall"], b.payload["hall"])
        self.assertEqual(a.payload["greedy_score"], b.payload["greedy_score"])

    def test_final_report_writes_each_configs_best_layout_as_a_solution_file(self) -> None:
        from team.evolve import Evolver
        from team.sa_library import NullLibrary

        task = HELDOUT[0]
        root = Path(self.tmp.name) / "run"
        ev = Evolver(datahall_run_team, layout_llm("garbage"), "reviewer", self.brain, root / "versions",
                     initial=datahall_v0(), reports_dir=root / "reports", library=NullLibrary())
        ev.final_report([task])
        files = sorted((root / "solutions").glob("*/*.json"))
        # v0 and vN share the "v0 + final brain" label (and file) when the Gate accepted no version
        self.assertEqual([f.parent.name for f in files], ["v0-empty-brain-cold", "v0-final-brain"])
        d = json.loads(files[0].read_text())
        self.assertEqual((d["case_study"], d["task_id"], d["split"], d["seed"], d["tier"]),
                         ("datahall", task.id, "heldout", task.payload["seed"], task.payload["tier"]))
        self.assertEqual(d["config"], "v0 + empty brain (cold)")
        self.assertEqual(d["solution"], task.payload["greedy_layout"])  # nothing beat greedy
        self.assertAlmostEqual(d["score"], round(task.payload["greedy_score"] / 100, 4), places=4)


if __name__ == "__main__":
    unittest.main()
