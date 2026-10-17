"""Version N+1: gate decisions, reviewer validation, promote_brain, held-out isolation, end to end."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team.bench import REVIEWER_SCRIPT, make_tasks, toy_run_team, toy_v0  # noqa: E402
from team.brain import Curator, SharedBrain  # noqa: E402
from team.crusoe import LIGHTNING, UsageLedger  # noqa: E402
from team.evolve import (Evolver, Gate, Reviewer, TeamVersion, apply_mutation, markdown_table, paper,  # noqa: E402
                         per_task_usage, validate_mutation)
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.types import RunResult, Task  # noqa: E402


def results(scores: list[float]) -> list[RunResult]:
    return [RunResult(task_id=f"d{i}", version=0, score=s, passed=s >= 1.0) for i, s in enumerate(scores)]


def costed(scores: list[float], cost_usd: float = 0.0, tokens: int = 0, passed: list[bool] | None = None) -> list[RunResult]:
    return [RunResult(task_id=f"d{i}", version=0, score=s, passed=s >= 1.0 if passed is None else passed[i],
                      cost_usd=cost_usd, prompt_tokens=tokens) for i, s in enumerate(scores)]


def claimed(m: dict, dev_gain: float = 0.1, cost_change_pct: float = 0.0) -> dict:
    return {**m, "claim": {"dev_gain": dev_gain, "cost_change_pct": cost_change_pct, "why": "test"}}


class LedgerLLM(ScriptedLLM):
    """ScriptedLLM plus the per-role usage ledger CrusoeLLM keeps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ledger = UsageLedger()

    def chat(self, *args, **kwargs):
        reply = super().chat(*args, **kwargs)
        self.ledger.record(reply, kwargs.get("role", ""))
        return reply


def tasks(split: str, n: int) -> list[Task]:
    return [Task(id=f"{split}-{i}", split=split, payload={}, text="ground the racks") for i in range(n)]


def same_tag(a: str, b: str) -> str:
    """Rules contradict only within one "[Dk]" pair, so each pair disputes itself and nothing else."""
    return "contradict" if a.split()[0] == b.split()[0] else "compatible"


def disputed_pair(brain: SharedBrain, k: int, topic: str) -> tuple[str, str]:
    """Two equally supported contradictory Layout rules; A says 1.2 m, B says 0.9 m."""
    ids = []
    for run_id, width in (("ra", "1.2"), ("rb", "0.9")):
        ids.append(brain.propose([{"op": "add", "kind": "lesson", "roles": ["Layout"], "text": f"[D{k}] {topic} is {width} m",
                                   "evidence": [{"run_id": run_id, "task_id": "t0", "measurement": "score=0.5"}]}], run_id)[0]["id"])
    return ids[0], ids[1]


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
        reply = json.dumps({"mutations": [claimed(m) for m in (
            {"type": "playbook_add", "role": "Layout", "step": "a", "rationale": "r"},
            {"type": "playbook_add", "role": "Ghost", "step": "b"},
            {"type": "config_set", "key": "roles_enabled", "value": []},
            {"type": "config_set", "key": "max_iters", "value": 3},
            {"type": "prompt_edit", "role": "Layout", "text": "c"},
            {"type": "remove_role", "role": "Layout"},
            {"type": "playbook_add", "role": "Layout", "step": "d"},
        )]})
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

    def test_tool_set_applies_from_the_catalog_and_unknown_tools_are_dropped(self) -> None:
        reply = json.dumps({"mutations": [claimed(m) for m in (
            {"type": "tool_set", "tool": "solver", "enabled": True, "params": {"time_limit": 30}, "rationale": "r"},
            {"type": "tool_set", "tool": "web_search", "enabled": True},
            {"type": "config_set", "key": "tool_catalog", "value": ["web_search"]},
        )]})
        llm = ScriptedLLM({"reviewer": [reply]})
        v0 = TeamVersion(version=0, parent=None, prompts={"Layout": "p"}, config={"tool_catalog": ["solver"]})
        brain = SharedBrain(self.dir / "brain")
        reviewer = Reviewer(llm, "m")
        valid = reviewer.propose(v0, results([0.5]), brain)
        self.assertEqual([m.get("tool") for m in valid], ["solver"])
        reasons = [d["reason"] for d in reviewer.dropped]
        self.assertTrue(any("unknown tool 'web_search'" in r for r in reasons), reasons)
        self.assertTrue(any("tool_catalog" in r for r in reasons), reasons)
        v1 = apply_mutation(v0, valid[0], brain, 1)
        self.assertEqual(v1.config["tools"], {"solver": {"enabled": True, "time_limit": 30}})
        self.assertIn("time_limit", v1.changelog[-1])
        system, user = (m["content"] for m in llm.calls_for("reviewer")[0]["messages"])
        self.assertIn("tool_set", system)
        self.assertEqual(json.loads(user)["tool_catalog"], ["solver"])
        self.assertIsNone(validate_mutation({"type": "tool_set", "tool": "web_search", "enabled": False}, v0, brain,
                                            tools=["web_search"]))

    def dispute_evolver(self, pairs: int, generations: int, score_of) -> tuple[SharedBrain, list, list[dict], list[str]]:
        """Evolver over `pairs` open disputes; score_of(context) is the dev score given what Layout read."""
        brain = SharedBrain(self.dir / "brain", Curator(judge=same_tag))
        ids = [disputed_pair(brain, k, topic) for k, topic in enumerate(("hot aisle", "cold aisle", "rack gap")[:pairs])]
        touched: list[str] = []

        def run_team(version: TeamVersion, b: SharedBrain, task: Task, llm: ScriptedLLM) -> RunResult:
            context, read = b.read("Layout", task.text)
            touched.append(task.id)
            return RunResult(task_id=task.id, version=version.version, score=score_of(context), passed=False,
                             brain_ids_read=read)

        llm = ScriptedLLM({"reviewer": [json.dumps({"mutations": []})]})
        evolver = Evolver(run_team, llm, "m", brain, self.dir / "versions",
                          initial=TeamVersion(version=0, parent=None, prompts={"Layout": "p"}))
        evolver.run(generations, tasks("train", 2), tasks("dev", 3), tasks("heldout", 2))
        log = [json.loads(l) for l in (self.dir / "versions" / "ledger.jsonl").read_text().splitlines()]
        return brain, ids, [r for r in log if "dispute" in r], touched

    def test_dispute_settled_by_dev_run_in_favour_of_the_side_whose_absence_hurts(self) -> None:
        brain, [(a, b)], tests, touched = self.dispute_evolver(1, 1, lambda ctx: 0.8 if "is 1.2 m" in ctx else 0.4)
        self.assertEqual(len(tests), 1)
        self.assertEqual((sorted(tests[0]["dispute"]), tests[0]["means"], tests[0]["winner"]),
                         (sorted([a, b]), {a: 0.8, b: 0.4}, a))
        win, lose = brain.get([a, b])
        self.assertEqual((win.status, lose.status), ("active", "retired"))
        self.assertIn(b, win.supersedes)
        ledger = [json.loads(l) for l in (brain.dir / "ledger.jsonl").read_text().splitlines()]
        self.assertTrue(any(r.get("id") == b and r["decision"] == "retired" and "dev mean 0.80 vs 0.40 on 3 tasks" in r["reason"]
                            for r in ledger))
        self.assertFalse(any("dev-" in f"{r.get('run_id')}{r.get('task_id')}" for r in ledger), "dev runs wrote to the brain")
        self.assertFalse(any(t.startswith("heldout") for t in touched))

    def test_tied_dispute_stays_open(self) -> None:
        brain, [(a, b)], tests, _ = self.dispute_evolver(1, 1, lambda ctx: 0.5)
        self.assertEqual([t["winner"] for t in tests], [None])
        self.assertEqual([e.status for e in brain.get([a, b])], ["disputed", "disputed"])
        self.assertEqual(brain.disputes(), [tuple(sorted((a, b)))])

    def test_at_most_two_disputes_tested_per_generation_and_ties_do_not_starve_the_rest(self) -> None:
        brain, ids, tests, _ = self.dispute_evolver(3, 2, lambda ctx: 0.5)
        self.assertEqual([sum(t["generation"] == g for t in tests) for g in (0, 1)], [2, 2])
        self.assertEqual({tuple(t["dispute"]) for t in tests}, {tuple(sorted(p)) for p in ids})
        self.assertEqual(len(brain.disputes()), 3)

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

        mutation = claimed({"type": "promote_brain", "entry_ids": [eid], "role": "Layout", "rationale": "proven rule"}, 0.0)
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

    def test_gate_accepts_20_percent_cheaper_at_equal_score_with_the_cost_reason(self) -> None:
        d = Gate().decide(costed([0.5, 0.5, 1.0], cost_usd=0.010), costed([0.5, 0.5, 1.0], cost_usd=0.008))
        self.assertTrue(d["accepted"])
        self.assertEqual((d["accepted_by"], d["reason"]), ("cost", "cheaper at equal score: cost -20%"))
        self.assertEqual((d["cost_unit"], d["parent_cost"], d["cand_cost"], d["cost_cut"]), ("usd", 0.01, 0.008, 0.2))

    def test_gate_rejects_10_percent_cheaper_at_equal_score(self) -> None:
        d = Gate().decide(costed([0.5, 0.5, 1.0], cost_usd=0.010), costed([0.5, 0.5, 1.0], cost_usd=0.009))
        self.assertFalse(d["accepted"])
        self.assertIn("cost -10% misses the -15% cut", d["reason"])

    def test_gate_rejects_cheaper_when_score_drops_or_a_pass_is_lost(self) -> None:
        lower = Gate().decide(costed([0.5, 0.5, 1.0], cost_usd=0.010), costed([0.4, 0.5, 1.0], cost_usd=0.002))
        self.assertFalse(lower["accepted"])
        self.assertIsNone(lower["accepted_by"])
        lost = Gate().decide(costed([1.0, 0.5], cost_usd=0.010, passed=[True, False]),
                             costed([1.0, 0.5], cost_usd=0.002, passed=[False, False]))
        self.assertEqual((lost["accepted"], lost["gain"], lost["cost_cut"]), (False, 0.0, 0.8))
        self.assertNotIn("cheaper", lost["reason"])

    def test_gate_falls_back_to_tokens_when_neither_arm_has_a_cost(self) -> None:
        d = Gate().decide(costed([0.5, 1.0], tokens=1000), costed([0.5, 1.0], tokens=800))
        self.assertEqual((d["accepted"], d["cost_unit"], d["parent_cost"], d["cand_cost"]), (True, "tokens", 1000, 800))
        self.assertEqual(Gate().decide(costed([0.5], tokens=1000), costed([0.5], tokens=1000), min_gain=0.0)["accepted_by"],
                         "score")

    def test_reviewer_sees_cost_and_validates_inference_knobs(self) -> None:
        reply = json.dumps({"mutations": [
            claimed({"type": "config_set", "key": "models", "value": {"Layout": "nvidia/Imaginary-9000"}}),
            claimed({"type": "config_set", "key": "default_model", "value": "gpt-5"}),
            claimed({"type": "config_set", "key": "max_tokens", "value": 0}),
            claimed({"type": "config_set", "key": "thinking", "value": {"Layout": "yes"}}),
            {"type": "config_set", "key": "max_iters", "value": 1},
            claimed({"type": "config_set", "key": "models", "value": {"Layout": LIGHTNING}}, 0.0, -30.0),
            claimed({"type": "config_set", "key": "thinking", "value": {"Layout": False}}, 0.0, -90.0),
        ]})
        llm = ScriptedLLM({"reviewer": [reply]})
        v0 = TeamVersion(version=0, parent=None, prompts={"Layout": "p"})
        reviewer = Reviewer(llm, "m")
        runs = [RunResult(task_id="t", version=0, score=0.5, passed=False, calls=4, prompt_tokens=900,
                          completion_tokens=100, cost_usd=0.002)]
        valid = reviewer.propose(v0, runs, SharedBrain(self.dir / "brain"))
        self.assertEqual([m["value"] for m in valid], [{"Layout": LIGHTNING}, {"Layout": False}])
        reasons = [d["reason"] for d in reviewer.dropped]
        for expected in ("unknown model id 'nvidia/Imaginary-9000'", "unknown model id 'gpt-5'", "max_tokens must be",
                         "thinking values", "claim must be"):
            self.assertTrue(any(expected in r for r in reasons), (expected, reasons))
        system, user = (m["content"] for m in llm.calls_for("reviewer")[0]["messages"])
        self.assertIn("at least 15% without lowering the mean score", system)
        for knob in ("thinking", "25x", "max_tokens", "max_iters", "temperature", LIGHTNING):
            self.assertIn(knob, system)
        summary = json.loads(user)
        self.assertEqual({k: summary["runs"][k] for k in ("cost_per_task_usd", "tokens_per_task", "calls_per_task")},
                         {"cost_per_task_usd": 0.002, "tokens_per_task": 1000, "calls_per_task": 4})
        self.assertIn("not available", summary["cost_per_task_by_role"])

    def test_reviewer_gets_cost_per_task_by_role_from_the_llm_ledger(self) -> None:
        self.assertEqual(per_task_usage({"A": {"calls": 2, "cost_usd": 0.1}}, {"A": {"calls": 6, "cost_usd": 0.3}}, 2)["A"]
                         ["calls"], 2.0)
        self.assertIsNone(per_task_usage(None, {}, 2))
        train, dev, heldout = make_tasks(0)
        llm = LedgerLLM({"reviewer": [json.dumps({"mutations": []})]})
        evolver = Evolver(toy_run_team, llm, "m", SharedBrain(self.dir / "brain"), self.dir / "versions",
                          initial=toy_v0(), reports_dir=self.dir / "reports")
        evolver.run(1, train, dev, heldout)
        by_role = json.loads(llm.calls_for("reviewer")[0]["messages"][1]["content"])["cost_per_task_by_role"]
        self.assertEqual(sorted(by_role), ["Layout", "Spec"])
        self.assertEqual(by_role["Layout"]["calls"], 1.0)
        self.assertGreater(by_role["Layout"]["prompt_tokens"], 0)

    def test_claims_are_published_as_reproduced_or_not_and_claimless_mutations_are_dropped(self) -> None:
        def run_team(version: TeamVersion, b: SharedBrain, task: Task, llm: ScriptedLLM) -> RunResult:
            good = "[good]" in " ".join(version.playbooks.get("Layout", []))
            return RunResult(task_id=task.id, version=version.version, score=0.8 if good else 0.5, passed=False)

        reply = json.dumps({"mutations": [
            claimed({"type": "playbook_add", "role": "Layout", "step": "[good] works"}, 0.3),
            claimed({"type": "playbook_add", "role": "Layout", "step": "[noop] does nothing"}, 0.2),
            {"type": "playbook_add", "role": "Layout", "step": "[bare] no claim"},
        ]})
        evolver = Evolver(run_team, ScriptedLLM({"reviewer": [reply]}), "m", SharedBrain(self.dir / "brain"),
                          self.dir / "versions", initial=TeamVersion(version=0, parent=None, prompts={"Layout": "p"}),
                          reports_dir=self.dir / "reports")
        evolver.run(1, tasks("train", 2), tasks("dev", 3), tasks("heldout", 2))
        self.assertEqual([(p["type"], p["verdict"]) for p in evolver.papers],
                         [("playbook_add", "reproduced"), ("playbook_add", "not reproduced")])
        self.assertEqual(evolver.papers[1]["claim"]["dev_gain"], 0.2)
        self.assertEqual(evolver.papers[1]["measured"], {"dev_gain": 0.0, "cost_change_pct": 0.0})
        log = [json.loads(l) for l in (self.dir / "versions" / "ledger.jsonl").read_text().splitlines()]
        self.assertTrue(any(r.get("reason", "").startswith("invalid: claim must be") for r in log))
        self.assertEqual([r["paper"]["verdict"] for r in log if "paper" in r], ["reproduced", "not reproduced"])
        papers_md = (self.dir / "reports" / "papers.md").read_text()
        self.assertTrue(papers_md.startswith("| Generation | Mutation type | Claim | Measured | Verdict |"))
        self.assertIn("| 0 | playbook_add | gain +0.200, cost +0.0% (test) | gain +0.000, cost +0.0% | not reproduced |",
                      papers_md)
        cost_only = Gate().decide(costed([0.5], cost_usd=0.01), costed([0.5], cost_usd=0.005))
        self.assertEqual(paper(claimed({"type": "x"}, 0.2, -50.0), cost_only)["verdict"],
                         "accepted, effect smaller than claimed")

    def test_report_has_calls_tokens_cost_per_task_and_cost_per_verified_point(self) -> None:
        train, dev, heldout = make_tasks(0)
        evolver, _ = self.toy_evolver()
        evolver.run(1, train, dev, heldout)
        rows = evolver.final_report(heldout)
        header = (self.dir / "reports" / "before_after.md").read_text().splitlines()[0]
        for column in ("Calls/task", "Tokens/task", "Cost/task (USD)", "Cost per verified point (USD)"):
            self.assertIn(column, header)
        r = rows[0]
        self.assertGreater(r["calls_per_task"], 0)
        self.assertAlmostEqual(r["cost_per_point_usd"], r["cost_usd"] / (r["mean_score"] * r["n"]), places=5)
        self.assertAlmostEqual(r["tokens_per_task"], r["tokens"] / r["n"], places=1)
        table = markdown_table([{**r, "mean_score": 0.0, "cost_per_point_usd": None},
                                {k: r[k] for k in ("config", "mean_score", "pass_rate", "mean_iterations", "tokens",
                                                   "cost_usd")}])
        zero, bare = table.splitlines()[2:]
        self.assertEqual(zero.split(" | ")[7], "n/a")
        self.assertEqual(bare.split(" | ")[4:8], ["-", "-", "-", "-"])

    def test_failed_llm_call_costs_one_task_and_the_gate_excludes_it_from_both_sides(self) -> None:
        def run_team(version: TeamVersion, b: SharedBrain, task: Task, llm: ScriptedLLM) -> RunResult:
            steps = " ".join(version.playbooks.get("Layout", []))
            if task.id in ("train-0", "heldout-0") or ("[good]" in steps and task.id == "dev-0") or (
                    "[down]" in steps and task.split == "dev"):
                raise RuntimeError("Crusoe 429 max_requests_limit (orchestrator)")
            return RunResult(task_id=task.id, version=version.version, score=0.8 if "[good]" in steps else 0.5,
                             passed=False)

        reply = json.dumps({"mutations": [claimed({"type": "playbook_add", "role": "Layout", "step": "[good] works"}),
                                          claimed({"type": "playbook_add", "role": "Layout", "step": "[down] dies"})]})
        brain = SharedBrain(self.dir / "brain")
        evolver = Evolver(run_team, ScriptedLLM({"reviewer": [reply]}), "m", brain, self.dir / "versions",
                          initial=TeamVersion(version=0, parent=None, prompts={"Layout": "p"}),
                          reports_dir=self.dir / "reports")
        with self.assertLogs("evolve", "WARNING"):
            evolver.run(1, tasks("train", 2), tasks("dev", 3), tasks("heldout", 2))
        self.assertEqual(evolver.current.version, 1)
        gated = [json.loads(l) for l in (self.dir / "versions" / "ledger.jsonl").read_text().splitlines()]
        gated = [r for r in gated if "paper" in r]
        self.assertEqual([(r["n"], r["excluded_infra"], r["accepted"]) for r in gated],
                         [(2, ["dev-0"], True), (0, ["dev-0", "dev-1", "dev-2"], False)])
        self.assertEqual(gated[1]["reason"], "infra: no comparable tasks")
        self.assertFalse(any("infra" in e.text for e in brain._snapshot()[1].values()))
        rows = evolver.final_report(tasks("heldout", 2))
        self.assertEqual([(r["n"], r["infra_failures"]) for r in rows], [(2, 1)] * 3)
        self.assertIn("3 infra failures", (self.dir / "reports" / "before_after.md").read_text())

    def test_failed_reviewer_call_logs_the_error_and_the_next_generation_still_evolves(self) -> None:
        class FlakyReviewer(ScriptedLLM):
            failed = False

            def chat(self, *args, **kwargs):
                if kwargs.get("role") == "reviewer" and not self.failed:
                    self.failed = True
                    raise RuntimeError("Crusoe 429 max_requests_limit")
                return super().chat(*args, **kwargs)

        def run_team(version: TeamVersion, b: SharedBrain, task: Task, llm: ScriptedLLM) -> RunResult:
            good = "[good]" in " ".join(version.playbooks.get("Layout", []))
            return RunResult(task_id=task.id, version=version.version, score=0.8 if good else 0.5, passed=False)

        reply = json.dumps({"mutations": [claimed({"type": "playbook_add", "role": "Layout", "step": "[good] works"})]})
        evolver = Evolver(run_team, FlakyReviewer({"reviewer": [reply]}), "m", SharedBrain(self.dir / "brain"),
                          self.dir / "versions", initial=TeamVersion(version=0, parent=None, prompts={"Layout": "p"}),
                          reports_dir=self.dir / "reports")
        evolver.run(2, tasks("train", 2), tasks("dev", 3), tasks("heldout", 2))
        log = [json.loads(l) for l in (self.dir / "versions" / "ledger.jsonl").read_text().splitlines()]
        self.assertIn({"generation": 0, "reviewer_error": "RuntimeError: Crusoe 429 max_requests_limit"}, log)
        self.assertEqual([h["mutations"] for h in evolver.history], [0, 1])
        self.assertEqual(evolver.current.version, 1)


if __name__ == "__main__":
    unittest.main()
