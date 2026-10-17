"""Shared Brain: evidence gate, dedup, contradictions, credit/prune, stable render, concurrency."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team.brain import Curator, SharedBrain, llm_judge  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.types import RunResult  # noqa: E402


def ev(run_id: str = "r1", measurement: str = "score=0.5") -> list[dict]:
    return [{"run_id": run_id, "task_id": "t1", "measurement": measurement}]


def add(text: str, roles: list[str] | None = None, evidence: list[dict] | None = None, kind: str = "lesson") -> dict:
    return {"op": "add", "kind": kind, "roles": roles or ["Layout"], "text": text,
            "evidence": ev() if evidence is None else evidence}


class BrainTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def live(self, brain: SharedBrain) -> dict:
        data = json.loads((brain.dir / "brain.json").read_text())["entries"]
        return {k: v for k, v in data.items() if v["status"] != "retired"}

    def test_rejects_proposals_without_run_and_measurement_evidence(self) -> None:
        brain = SharedBrain(self.dir)
        recs = brain.propose([add("aisles need 1.2 m", evidence=[]),
                              add("racks face the cold aisle", evidence=[{"run_id": "r1"}]),
                              add("tiles are 600 mm", evidence=[{"measurement": "score=1"}])], "r1")
        self.assertEqual([r["decision"] for r in recs], ["rejected"] * 3)
        self.assertEqual(brain.stats()["active"], 0)
        ledger = [json.loads(l) for l in (self.dir / "ledger.jsonl").read_text().splitlines()]
        self.assertTrue(all("evidence" in r["reason"] for r in ledger))

    def test_dedup_merges_evidence_but_different_numbers_stay_separate(self) -> None:
        brain = SharedBrain(self.dir)
        brain.propose([add("Always check rack clearance before layout", evidence=ev("r1"))], "r1")
        recs = brain.propose([add("always check the rack clearance before layout.", evidence=ev("r2"))], "r2")
        self.assertEqual(recs[0]["decision"], "merged")
        (entry,) = self.live(brain).values()
        self.assertEqual({e["run_id"] for e in entry["evidence"]}, {"r1", "r2"})
        brain.propose([add("check F6 failed"), add("check F8 failed")], "r3")
        self.assertEqual(len(self.live(brain)), 3)

    def test_contradiction_keeps_better_supported_entry_and_links_supersedes(self) -> None:
        brain = SharedBrain(self.dir, Curator(judge=lambda a, b: "contradict"))
        brain.propose([add("hot aisle is 1.2 m wide", evidence=ev("r1") + ev("r2"))], "r1")
        (old_id,) = self.live(brain)
        recs = brain.propose([add("cold aisle containment is optional", evidence=ev("r3"))], "r3")
        self.assertEqual(recs[0]["decision"], "contradiction")
        live = self.live(brain)
        self.assertEqual(list(live), [old_id])
        self.assertIn(recs[0]["id"], live[old_id]["supersedes"])

    def test_contradiction_with_equal_support_marks_both_disputed(self) -> None:
        brain = SharedBrain(self.dir, Curator(judge=lambda a, b: "contradict"))
        brain.propose([add("hot aisle is 1.2 m wide")], "r1")
        brain.propose([add("hot aisle may be 0.9 m wide", evidence=ev("r2"))], "r2")
        self.assertEqual(sorted(e["status"] for e in self.live(brain).values()), ["disputed", "disputed"])
        context, ids = brain.read("Layout", "aisle width")
        self.assertEqual(len(ids), 2)
        self.assertEqual(context.count("[DISPUTED]"), 2)
        self.assertEqual(brain.stats()["disputed"], 2)

    def test_resolve_dispute_activates_winner_and_retires_loser_citing_the_measurement(self) -> None:
        brain = SharedBrain(self.dir, Curator(judge=lambda a, b: "contradict"))
        a = brain.propose([add("hot aisle is 1.2 m wide")], "r1")[0]["id"]
        b = brain.propose([add("hot aisle may be 0.9 m wide", evidence=ev("r2"))], "r2")[0]["id"]
        self.assertEqual(brain.disputes(), [tuple(sorted((a, b)))])
        self.assertEqual(brain.resolve_dispute(a, "ghost", "dev mean 0.6 vs 0.4 on 3 tasks", "g0")[0]["decision"], "rejected")
        measurement = "dev mean 0.62 vs 0.48 on 6 tasks"
        brain.resolve_dispute(a, b, measurement, "g0-dispute")
        win, lose = brain.get([a, b])
        self.assertEqual((win.status, lose.status), ("active", "retired"))
        self.assertIn(b, win.supersedes)
        self.assertIn({"run_id": "g0-dispute", "measurement": measurement}, win.evidence)
        self.assertEqual((brain.disputes(), brain.stats()["disputed"]), ([], 0))
        ledger = [json.loads(l) for l in (self.dir / "ledger.jsonl").read_text().splitlines()]
        (retired,) = [r for r in ledger if r["op"] == "resolve_dispute" and r["decision"] == "retired"]
        self.assertEqual(retired["id"], b)
        self.assertIn(measurement, retired["reason"])
        self.assertEqual(brain.resolve_dispute(a, b, measurement, "g1")[0]["decision"], "rejected")

    def test_llm_judge_parses_one_word_verdict(self) -> None:
        judge = llm_judge(ScriptedLLM({"judge": ['{"verdict": "contradict"}', "nonsense"]}), "m")
        self.assertEqual(judge("a", "b"), "contradict")
        self.assertEqual(judge("a", "b"), "compatible")

    def test_credit_counts_helps_hurts_and_prune_retires_harmful_then_overflow(self) -> None:
        brain = SharedBrain(self.dir)
        bad = brain.propose([add("aisles may shrink to 0.8 m", kind="fact")], "r1")[0]["id"]
        good = brain.propose([add("tiles are 600 mm on a grid", kind="fact")], "r1")[0]["id"]
        other = brain.propose([add("cooling needs liquid loops above 30 kW", kind="fact")], "r1")[0]["id"]
        for _ in range(3):
            brain.credit(RunResult(task_id="t", version=0, score=0.2, passed=False, brain_ids_read=[bad]), 0.6)
            brain.credit(RunResult(task_id="t", version=0, score=0.9, passed=True, brain_ids_read=[good]), 0.6)
        brain.credit(RunResult(task_id="t", version=0, score=0.6, passed=False, brain_ids_read=[other]), 0.6)
        entries = self.live(brain)
        self.assertEqual((entries[bad]["uses"], entries[bad]["hurts"], entries[bad]["helps"]), (3, 3, 0))
        self.assertEqual((entries[good]["helps"], entries[other]["uses"], entries[other]["helps"]), (3, 1, 0))
        retired = brain.prune(max_active=1, min_uses=3)
        self.assertEqual([r["id"] for r in retired], [bad, other])
        self.assertEqual(list(self.live(brain)), [good])

    def test_rendering_is_byte_identical_across_reads_and_instances(self) -> None:
        brain = SharedBrain(self.dir)
        brain.propose([add(f"rule {i} about aisle number {i}", roles=["Layout" if i % 2 else "all"]) for i in range(6)], "r1")
        first = brain.read("Layout", "aisle layout task")
        self.assertEqual(first, brain.read("Layout", "aisle layout task"))
        self.assertEqual(first, SharedBrain(self.dir).read("Layout", "aisle layout task"))
        self.assertEqual(len(first[1]), 6)
        self.assertEqual(brain.stable_prefix("Layout"), SharedBrain(self.dir).stable_prefix("Layout"))
        self.assertEqual(brain.read("Layout", "x", budget_chars=150)[1], first[1][:1])

    def test_eight_threads_proposing_concurrently_lose_nothing(self) -> None:
        def worker(t: int) -> None:
            brain = SharedBrain(self.dir)  # separate instance per thread: exercises flock, not just a mutex
            for i in range(5):
                brain.propose([add(f"thread {t} item {i} rule", evidence=ev(f"r{t}-{i}"))], f"r{t}-{i}")
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        brain = SharedBrain(self.dir)
        self.assertEqual(brain.stats()["active"], 40)
        self.assertEqual(brain.version(), 40)
        self.assertEqual(len((self.dir / "ledger.jsonl").read_text().splitlines()), 40)


if __name__ == "__main__":
    unittest.main()
