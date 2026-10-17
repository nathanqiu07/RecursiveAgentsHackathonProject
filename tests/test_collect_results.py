"""collect_results builds results from fixtures in a temp repo: a done live run, an in-progress run with a
half-written ledger line, a missing run, a crashed run, and offline runs that must never show up as results."""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_results as cr  # noqa: E402

OFFLINE_SCORE = 0.987654  # a number only the offline fixtures carry


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def row(config: str, version, score: float, tokens: int, cost: float) -> dict:
    return {"config": config, "version": version, "n": 15, "mean_score": score, "pass_rate": 1.0,
            "mean_iterations": 2.0, "tokens": tokens, "cost_usd": cost}


def build_fixture(root: Path) -> None:
    # Done live run: final report with the duplicated final arm, greedy row in the md, paid usage, a wall line.
    d = root / "reports" / "datahall_live_1"
    rows = [row("v0 + empty brain (cold)", 0, 0.70, 1000, 0.01), row("v0 + final brain", 0, 0.75, 1200, 0.02),
            row("v0 + final brain", 0, 0.75, 1200, 0.02)]
    write(d / "before_after.json", json.dumps({"rows": rows, "history": [
        {"generation": 0, "parent": 0, "version": 0, "train_mean": 0.6, "dev_mean": 0.65, "accepted": None}]}))
    write(d / "vs_greedy.md", "| Config | Mean score | Pass rate | Mean iters | Tokens | Cost (USD) |\n"
                              "|---|---|---|---|---|---|\n"
                              "| v0 + empty brain (cold) | 0.700 | 100% | 2.00 | 1000 | 0.0100 |\n"
                              "| greedy baseline (no LLM) | 0.710 | 100% | 0.00 | 0 | 0.0000 |\n")
    write(d / "usage.json", json.dumps({"total": {"calls": 40, "errors": 0, "prompt_tokens": 3000,
                                                  "completion_tokens": 500, "cached_tokens": 0, "cost_usd": 0.05}}))
    write(d / "version_ledger.jsonl", "\n".join(json.dumps(r) for r in [
        {"generation": 0, "parent": 0, "mutation": {"type": "tool_set", "tool": "capacity_calc", "enabled": True},
         "n": 6, "gain": -0.0006, "accepted": False, "reason": "gain -0.001 < 0.05"},
        {"generation": 0, "parent": 0, "mutation": {"type": "playbook_add", "role": "Layout", "step": "x" * 300},
         "accepted": False, "reason": "invalid: step longer than 280 chars"}]) + "\n")
    write(d / "run.log", "greedy baseline: train 68.74, dev 70.28, held-out 71.00 (of 100)\nwall 12.5s; artifacts: x\n")

    # In-progress QAPLIB run: one complete ledger record, one half-written line, no final report, no usage.
    q = root / "runs" / "qaplib-evolve-1"
    write(q / "versions" / "v0.json", "{}")
    write(q / "versions" / "ledger.jsonl", json.dumps(
        {"generation": 0, "parent": 0, "mutation": {"type": "prompt_edit", "role": "solver", "text": "be robust"},
         "n": 3, "parent_mean": 0.93, "gain": -0.13, "accepted": False, "reason": "gain -0.130 < 0.003",
         "paper": {"claim": {"dev_gain": 0.07, "cost_change_pct": 1.0, "why": "multi-start"},
                   "verdict": "not reproduced"}}) + "\n" + '{"generation": 0, "parent": 0, "mutat')
    write(root / "runs" / "qaplib-evolve-1.log", "Estimated LLM calls: ~50 team runs; at 20 RPM per model\n"
                                                 "train: els19(n=19)\ndev: had20(n=20)\nheld-out: tai15a(n=15), rou20(n=20)\n")
    write(root / "reports" / "qaplib_baseline.json", json.dumps({"time_limit_s": 10.0, "instances": {
        "tai15a": {"two_swap": {"score": 1.0, "gap_pct": 0.0}}, "rou20": {"two_swap": {"score": 0.98, "gap_pct": 2.0}}},
        "summary": {"heldout": {"two_swap": {"mean_score": 0.95}}}}))

    # Crashed ALE-Bench run: the log ends in an uncaught exception.
    write(root / "runs" / "alebench-live-1" / "versions" / "v0.json", "{}")
    write(root / "runs" / "alebench-live-1.log",
          "Traceback (most recent call last):\n  File \"x.py\", line 1\n"
          "openai.RateLimitError: Error code: 429 - {'error': {'message': 'Too many requests. Reason: "
          "max_requests_limit (orchestrator)'}}\n")

    # Offline runs: a scripted KernelBench run (final report, no usage.json) and a scripted data hall run.
    for name in ("kernelbench-20261017-150000", "datahall-offline"):
        write(root / "runs" / name / "reports" / "before_after.json",
              json.dumps({"rows": [row("v1 + final brain", 1, OFFLINE_SCORE, 10, 0.0)], "history": []}))


class CollectResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        build_fixture(cls.root)
        cls.data = cr.collect(cls.root, cls.root / "no-headroom", now=time.time())
        cls.cases = {c["id"]: c for c in cls.data["case_studies"]}
        cls.json_path, cls.md_path = cr.write_outputs(cls.data, cls.root / "out")
        cls.md = cls.md_path.read_text()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_done_run_reports_heldout_totals_and_generations(self):
        c = self.cases["datahall-live-1"]
        self.assertEqual(c["status"], "done")
        self.assertEqual(c["baseline"]["score"], 0.71)
        configs = [r["config"] for r in c["heldout"]]
        self.assertEqual(configs, ["v0 + empty brain (cold)", "v0 + final brain", "greedy baseline (no LLM)"])
        self.assertTrue(c["notes"], "the duplicated final arm is explained")
        self.assertEqual(c["best"]["config"], "v0 + final brain")
        self.assertAlmostEqual(c["best"]["gain"], 0.04)
        self.assertEqual(c["totals"]["calls"], 40)
        self.assertEqual(c["totals"]["tokens"], 3500)
        self.assertEqual(c["totals"]["cost_usd"], 0.05)
        self.assertEqual(c["totals"]["wall_s"], 12.5)
        self.assertEqual([g["accepted"] for g in c["generations"]], [False, False])
        self.assertIsNone(c["generations"][1]["measured_gain"])
        self.assertIn("reports/datahall_live_1/usage.json", c["sources"])

    def test_in_progress_run_keeps_complete_ledger_lines_and_shows_no_heldout(self):
        c = self.cases["qaplib-evolve-1"]
        self.assertEqual(c["status"], "running")
        self.assertEqual(len(c["generations"]), 1, "the half-written line is skipped")
        self.assertEqual(c["generations"][0]["verdict"], "not reproduced")
        self.assertEqual(c["heldout"], [])
        self.assertIsNone(c["best"])
        self.assertIsNone(c["totals"])
        self.assertAlmostEqual(c["baseline"]["score"], 0.99)  # 2-swap on this run's two held-out instances

    def test_missing_run_is_not_started(self):
        c = self.cases["datahall-live-2"]
        self.assertEqual(c["status"], "not started")
        self.assertEqual(c["generations"], [])
        self.assertEqual(self.data["experiments"][0]["status"], "not started")

    def test_crashed_run_is_blocked_with_a_short_reason(self):
        c = self.cases["alebench-live-1"]
        self.assertEqual(c["status"], "blocked")
        self.assertIn("RateLimitError, HTTP 429, max_requests_limit", c["status_detail"])

    def test_offline_runs_never_appear_as_results(self):
        kb = self.cases["kernelbench-mini"]
        self.assertEqual(kb["status"], "blocked")
        self.assertIn("needs a Crusoe GPU VM", kb["status_detail"])
        self.assertEqual(kb["heldout"], [])
        self.assertNotIn(str(OFFLINE_SCORE), self.md)
        self.assertNotIn(str(OFFLINE_SCORE), self.json_path.read_text())
        not_counted = [r["run"] for r in self.data["not_counted"]]
        self.assertIn("runs/datahall-offline", not_counted)
        self.assertIn("runs/kernelbench-20261017-150000", not_counted)

    def test_markdown_layout(self):
        self.assertNotIn("—", self.md)
        lines = self.md.splitlines()
        self.assertIn("| Case study | Baseline | Best team version | Gain | Status | Cost |", lines)
        summary = lines.index("| Case study | Baseline | Best team version | Gain | Status | Cost |")
        self.assertLess(summary, lines.index("## Data hall layout, live run 1"))
        self.assertLess(lines.index("## Data hall layout, live run 1"), lines.index("## Live smoke tests"))
        self.assertTrue(lines[-1].startswith("Generated ") and cr.COMMAND in lines[-1])
        self.assertIn("| Data hall layout, live run 1 | greedy baseline (no LLM) 0.710 | v0 + final brain 0.750 | "
                      "+0.040 (+5.6%) | done | $0.0500 (40 calls) |", lines)

    def test_json_schema_is_stable(self):
        data = json.loads(self.json_path.read_text())
        self.assertEqual(data["schema_version"], cr.SCHEMA_VERSION)
        for c in data["case_studies"]:
            for key in ("name", "verifier", "baseline", "status", "heldout", "generations", "totals", "sources"):
                self.assertIn(key, c)
            self.assertIn(c["status"], ("done", "running", "not started", "blocked"))


if __name__ == "__main__":
    unittest.main()
