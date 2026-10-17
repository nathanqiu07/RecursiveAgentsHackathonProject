"""Evolver -> SA's Library: a mutation becomes a paper, the Gate outcome its verdict, and a down Library breaks nothing.

The fake Library serves SA's real HTTP shapes from SA's own PaperQueue on a temp SQLite file, so the report
is written exactly the way SA's worker writes it and read back through GET /papers/{id}.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "library"))

from paper_library.queue import PaperQueue  # noqa: E402  SA's code, unmodified
from scripts.publish_run_to_library import main as backfill  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.evolve import Evolver, TeamVersion  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402
from team.sa_library import LibraryPublisher, NullLibrary, claim_submission, from_env, gate_report  # noqa: E402
from team.types import RunResult, Task  # noqa: E402


def fake_library(queue: PaperQueue) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                return self.reply(200, {"status": "ok"})
            if self.path.startswith("/papers?"):
                return self.reply(200, queue.list(100, 0))
            paper = queue.get(self.path.rsplit("/", 1)[-1])
            self.reply(200 if paper else 404, paper or {"detail": "Paper not found"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.reply(202, queue.submit(body))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def dead_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def tasks(split: str, n: int) -> list[Task]:
    return [Task(id=f"{split}-{i}", split=split, payload={}, text="t") for i in range(n)]


def run_team(version: TeamVersion, b: SharedBrain, task: Task, llm: ScriptedLLM) -> RunResult:
    good = "[good]" in " ".join(version.playbooks.get("Layout", []))
    return RunResult(task_id=task.id, version=version.version, score=0.8 if good else 0.5, passed=False)


MUTATIONS = [
    {"type": "playbook_add", "role": "Layout", "step": "[good] works", "claim": {"dev_gain": 0.3, "cost_change_pct": 0.0, "why": "t"}},
    {"type": "playbook_add", "role": "Layout", "step": "[noop] nothing", "claim": {"dev_gain": 0.2, "cost_change_pct": 0.0, "why": "t"}},
    {"type": "playbook_add", "role": "Layout", "step": "[bare] no claim"},
]


class SALibraryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = self.dir / "library.sqlite3"
        self.queue = PaperQueue(self.db)
        self.server = fake_library(self.queue)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def publisher(self, url: str | None = None, state: str = "library.jsonl") -> LibraryPublisher:
        return LibraryPublisher(url or self.url, "toy", state_path=self.dir / state, db_path=self.db)

    def evolver(self, library) -> Evolver:
        llm = ScriptedLLM({"reviewer": [json.dumps({"mutations": MUTATIONS})]})
        return Evolver(run_team, llm, "m", SharedBrain(self.dir / "brain"), self.dir / "versions",
                       initial=TeamVersion(version=0, parent=None, prompts={"Layout": "p"}),
                       reports_dir=self.dir / "reports", library=library)

    def test_mutation_becomes_a_paper_with_its_claim_as_a_verification_contract(self) -> None:
        [pid] = self.publisher().publish_claims(2, 1, MUTATIONS[:1], {"n_dev": 4, "min_gain": 0.05, "min_cost_cut": 0.2})
        paper = self.queue.get(pid)
        self.assertEqual(paper["status"], "pending_verification")
        sub = paper["submission"]
        self.assertEqual(sub["title"], "[toy] g2 v1: playbook_add Layout: [good] works")
        self.assertEqual(sub["submitted_by"], "evolve-reviewer")
        for part in ("Claim. Applying this change raises the mean dev score by +0.3000", "Acceptance criteria.",
                     "paired comparison on 4 dev tasks", "Reproduction."):
            self.assertIn(part, sub["paper_text"])
        contract = json.loads(sub["notes"])
        self.assertEqual((contract["claim"]["dev_gain"], contract["acceptance"]["min_gain"]), (0.3, 0.05))

    def test_gate_outcome_becomes_the_papers_verdict_read_back_through_the_api(self) -> None:
        pub = self.publisher()
        [pid] = pub.publish_claims(0, 0, MUTATIONS[1:2])
        decision = {"n": 3, "parent_mean": 0.5, "cand_mean": 0.5, "gain": 0.0, "parent_passed": 0, "cand_passed": 0,
                    "cost_change_pct": 0.0, "accepted": False, "reason": "gain +0.000 < 0.05"}
        self.assertEqual(pub.post_verdict(pid, MUTATIONS[1], decision), "not_verified")
        paper = pub.client.get(pid)
        self.assertEqual((paper["status"], paper["report"]["verdict"]), ("completed", "not_verified"))
        self.assertIn("Claimed dev gain +0.2000", paper["report"]["summary"])
        self.assertIn("measured dev gain +0.0000", paper["report"]["summary"])
        self.assertEqual(gate_report(MUTATIONS[0], {**decision, "accepted": True, "gain": 0.3})["verdict"], "verified")

    def test_evolver_publishes_every_mutation_and_verdict_live(self) -> None:
        evolver = self.evolver(self.publisher())
        evolver.run(1, tasks("train", 2), tasks("dev", 3), tasks("heldout", 2))
        papers = self.queue.list(100, 0)["papers"]
        self.assertEqual(len(papers), 3)  # two gated claims, one dropped as invalid
        verdicts = sorted(p["report"]["verdict"] for p in papers)
        self.assertEqual(verdicts, ["inconclusive", "not_verified", "verified"])
        self.assertEqual([p["verdict"] for p in evolver.papers], ["reproduced", "not reproduced"])

    def test_library_down_does_not_break_the_loop(self) -> None:
        evolver = self.evolver(self.publisher(dead_url()))
        with self.assertLogs("sa_library", "WARNING") as logs:
            evolver.run(1, tasks("train", 2), tasks("dev", 3), tasks("heldout", 2))
        self.assertEqual([p["verdict"] for p in evolver.papers], ["reproduced", "not reproduced"])
        self.assertEqual(evolver.current.version, 1)
        self.assertTrue(any("the loop continues" in line for line in logs.output))

    def test_report_falls_back_to_a_linked_paper_when_the_database_is_elsewhere(self) -> None:
        pub = LibraryPublisher(self.url, "toy", db_path=self.dir / "missing.sqlite3")
        [pid] = pub.publish_claims(0, 0, MUTATIONS[:1])
        with self.assertLogs("sa_library", "WARNING"):
            self.assertEqual(pub.post_verdict(pid, MUTATIONS[0], {"accepted": False, "reason": "gain -0.1 < 0.05"}),
                             "not_verified")
        linked = [p for p in self.queue.list(100, 0)["papers"] if p["id"] != pid]
        self.assertEqual(json.loads(linked[0]["submission"]["notes"])["verifies"], pid)

    def test_env_var_turns_publishing_on_and_its_absence_is_a_no_op(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SA_LIBRARY_URL", None)
            self.assertIsInstance(from_env(self.dir / "v"), NullLibrary)
            self.assertNotIsInstance(from_env(self.dir / "v"), LibraryPublisher)
            os.environ.update(SA_LIBRARY_URL=self.url, SA_LIBRARY_BENCHMARK="qaplib", SA_LIBRARY_DB=str(self.db))
            pub = from_env(self.dir / "v")
            self.assertEqual((type(pub), pub.benchmark), (LibraryPublisher, "qaplib"))

    def test_backfill_publishes_a_ledger_once_and_rerunning_skips_it(self) -> None:
        run = self.dir / "run"
        (run / "versions").mkdir(parents=True)
        decision = {"n": 6, "parent_mean": 0.7034, "cand_mean": 0.7027, "gain": -0.0006, "parent_passed": 6,
                    "cand_passed": 6, "accepted": False, "reason": "gain -0.001 < 0.05"}
        rows = [{"generation": 0, "parent": 0, "mutation": {"type": "tool_set", "tool": "x", "enabled": True}, **decision},
                {"generation": 0, "parent": 0, "mutation": {"type": "playbook_add", "role": "L", "step": "s" * 300},
                 "accepted": False, "reason": "invalid: step longer than 280 chars"},
                {"generation": 1, "dispute": ["a1", "b2"], "means": {"a1": 0.6, "b2": 0.5}, "winner": "a1"}]
        (run / "versions" / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        args = [str(run), "--url", self.url, "--benchmark", "datahall", "--db", str(self.db)]
        self.assertEqual(backfill(args), 0)
        self.assertEqual(backfill(args), 0)
        papers = self.queue.list(100, 0)["papers"]
        self.assertEqual(len(papers), 3)
        self.assertEqual(sorted(p["report"]["verdict"] for p in papers), ["inconclusive", "not_verified", "verified"])


if __name__ == "__main__":
    unittest.main()
