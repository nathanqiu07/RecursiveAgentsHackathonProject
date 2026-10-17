"""Version N+1: gate decisions, reviewer validation, promote_brain, held-out isolation, end to end."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team.bench import REVIEWER_SCRIPT, make_tasks, toy_run_team, toy_v0  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.evolve import Evolver, Gate, Reviewer, TeamVersion  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.types import RunResult, Task  # noqa: E402


def results(scores: list[float]) -> list[RunResult]:
    return [RunResult(task_id=f"d{i}", version=0, score=s, passed=s >= 1.0) for i, s in enumerate(scores)]


def tasks(split: str, n: int) -> list[Task]:
    return [Task(id=f"{split}-{i}", split=split, payload={}, text="ground the racks") for i in range(n)]


class EvolveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_gate_accepts_real_gain_and_rejects_regression_or_lost_passes(self) -> None:
        gate = Gate(min_gain=0.05)
        self.assertTrue(gate.decide(results([0.5, 0.5, 1.0]), results([0.7, 0.6, 1.0]))["accepted"])
        self.assertFalse(gate.decide(results([0.5, 0.5, 1.0]), results([0.4, 0.5, 1.0]))["accepted"])
        self.assertFalse(gate.decide(results([0.5, 0.5, 1.0]), results([0.52, 0.5, 1.0]))["accepted"])
        lost_pass = gate.decide(results([0.0, 0.0, 1.0]), results([0.9, 0.9, 0.9]))
        self.assertFalse(lost_pass["accepted"])
        self.assertIn("passed dropped", lost_pass["reason"])

    def test_reviewer_drops_invalid_mutations_and_caps_at_three(self) -> None:
        reply = json.dumps({"mutations": [
            {"type": "playbook_add", "role": "Layout", "step": "a", "rationale": "r"},
            {"type": "playbook_add", "role": "Ghost", "step": "b"},
            {"type": "config_set", "key": "roles_enabled", "value": []},
            {"type": "config_set", "key": "max_iters", "value": 3},
            {"type": "prompt_edit", "role": "Layout", "text": "c"},
            {"type": "remove_role", "role": "Layout"},
            {"type": "playbook_add", "role": "Layout", "step": "d"},
        ]})
        llm = ScriptedLLM({"reviewer": [reply]})
        v0 = TeamVersion(version=0, parent=None, prompts={"Layout": "p"})
        reviewer = Reviewer(llm, "m")
        valid = reviewer.propose(v0, results([0.5]), SharedBrain(self.dir / "brain"))
        self.assertEqual([m["type"] for m in valid], ["playbook_add", "config_set", "prompt_edit"])
        reasons = [d["reason"] for d in reviewer.dropped]
        self.assertEqual(len(reasons), 4)
        for expected in ("unknown role", "other than roles_enabled", "last role", "over the limit"):
            self.assertTrue(any(expected in r for r in reasons), expected)
        self.assertTrue(llm.calls_for("reviewer")[0]["json_mode"])

    def run_promote(self, outcome: str) -> tuple[SharedBrain, str, Evolver, list]:
        """outcome: "helps", "neutral" or "hurts", the effect of moving the entry into the playbook."""
        brain = SharedBrain(self.dir / f"brain-{outcome}")
        eid = brain.propose([{"op": "add", "kind": "fact", "roles": ["Layout"], "text": "[X] always ground the racks",
                              "evidence": [{"run_id": "r0", "task_id": "t0", "measurement": "score=1.0"}]}], "r0")[0]["id"]
        seen = []

        def run_team(version: TeamVersion, b: SharedBrain, task: Task, llm: ScriptedLLM) -> RunResult:
            _, ids = b.read("Layout", task.text)
            seen.append((version.version, ids))
            promoted = "[X]" in " ".join(version.playbooks.get("Layout", []))
            score = {"helps": 1.0, "neutral": 0.5, "hurts": 0.3}[outcome] if promoted else 0.5
            return RunResult(task_id=task.id, version=version.version, score=score, passed=score >= 1.0, brain_ids_read=ids)

        mutation = {"type": "promote_brain", "entry_ids": [eid], "role": "Layout", "rationale": "proven rule"}
        llm = ScriptedLLM({"reviewer": [json.dumps({"mutations": [mutation]})]})
        evolver = Evolver(run_team, llm, "m", brain, self.dir / f"versions-{outcome}",
                          initial=TeamVersion(version=0, parent=None, prompts={"Layout": "p"}))
        evolver.run(1, tasks("train", 2), tasks("dev", 3), tasks("heldout", 2))
        return brain, eid, evolver, seen

    def test_promote_brain_retires_consolidated_entries_only_when_accepted(self) -> None:
        brain, eid, evolver, seen = self.run_promote("helps")
        self.assertEqual(evolver.current.version, 1)
        self.assertIn("[X] always ground the racks", evolver.current.playbooks["Layout"])
        self.assertEqual(brain.get([eid])[0].status, "retired")
        self.assertTrue(all(eid not in ids for v, ids in seen if v == 1), "candidate must run without the entry")
        ledger = [json.loads(l) for l in (brain.dir / "ledger.jsonl").read_text().splitlines()]
        self.assertTrue(any(r.get("reason") == "consolidated into v1" for r in ledger))

        brain, eid, evolver, _ = self.run_promote("hurts")
        self.assertEqual(evolver.current.version, 0)
        self.assertEqual(brain.get([eid])[0].status, "active")

    def test_promote_brain_accepted_at_equal_score_because_brain_gets_leaner(self) -> None:
        brain, eid, evolver, _ = self.run_promote("neutral")
        self.assertEqual(evolver.current.version, 1)
        self.assertEqual(brain.get([eid])[0].status, "retired")

    def toy_evolver(self, run_team=toy_run_team) -> tuple[Evolver, ScriptedLLM]:
        llm = ScriptedLLM({"reviewer": REVIEWER_SCRIPT})
        return Evolver(run_team, llm, "m", SharedBrain(self.dir / "brain"), self.dir / "versions",
                       initial=toy_v0(), reports_dir=self.dir / "reports"), llm

    def test_heldout_tasks_untouched_until_final_report(self) -> None:
        touched: list[str] = []

        def recording(version, brain, task, llm) -> RunResult:
            touched.append(task.id)
            return toy_run_team(version, brain, task, llm)

        train, dev, heldout = make_tasks(0)
        evolver, _ = self.toy_evolver(recording)
        evolver.run(2, train, dev, heldout)
        heldout_ids = {t.id for t in heldout}
        self.assertFalse(heldout_ids & set(touched))
        with self.assertRaises(RuntimeError):
            evolver._batch(evolver.current, evolver.brain, heldout[:1])
        evolver.final_report(heldout)
        self.assertTrue(heldout_ids <= set(touched))
        with self.assertRaises(ValueError):
            evolver.run(1, train, dev, train[:1])

    def test_end_to_end_toy_final_version_beats_cold_v0_on_heldout(self) -> None:
        train, dev, heldout = make_tasks(0)
        evolver, llm = self.toy_evolver()
        final = evolver.run(3, train, dev, heldout)
        cold, brain_only, best = evolver.final_report(heldout)
        self.assertGreater(final.version, 0)
        self.assertGreater(brain_only["mean_score"], cold["mean_score"])
        self.assertGreater(best["mean_score"], brain_only["mean_score"])
        self.assertEqual(len(llm.calls_for("reviewer")), 3)
        self.assertTrue((self.dir / "reports" / "before_after.md").read_text().startswith("| Config"))
        self.assertTrue((self.dir / "versions" / f"v{final.version}.json").exists())
        decisions = [json.loads(l) for l in (self.dir / "versions" / "ledger.jsonl").read_text().splitlines()]
        self.assertTrue(any(d.get("accepted") is False and d.get("gain", 0) < 0 for d in decisions))


if __name__ == "__main__":
    unittest.main()
