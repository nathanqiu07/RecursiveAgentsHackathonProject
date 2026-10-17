"""Each verifier HTTP API must return exactly the score its library computes directly.

Run: .venv/bin/python -m unittest tests/test_verifier_apis.py
Servers start in-process on ephemeral loopback ports, so the running dashboard services are not touched.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "donetwin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmarks.alebench import scorer  # noqa: E402
from benchmarks.alebench import verifier_api as ale_api  # noqa: E402
from benchmarks.qaplib import verifier_api as qap_api  # noqa: E402
from benchmarks.qaplib.verify import INSTANCES, parse_sln, verify  # noqa: E402


def summary(feedback):
    """The plain-English part of a verifier's feedback: the paragraph before 'Details:'."""
    return feedback.split("\n\n")[0]


def sentence_count(text):
    import re
    return len([p for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if p])


def serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/evaluations"


def post(url, body, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


class QaplibVerifierApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, cls.url = serve(qap_api.Handler)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_single_instance_score_matches_verify(self):
        for name in ("nug22", "chr15c"):
            n, _, perm = parse_sln(INSTANCES / f"{name}.sln")
            for p in (perm, list(range(n))):
                code, res = post(self.url, {"submission": {"instance": name, "permutation": p}})
                self.assertEqual(code, 200)
                direct = verify(name, p)
                self.assertEqual(res["cost"], direct["cost"])
                self.assertAlmostEqual(res["score"], direct["score"], places=12)

    def test_split_mean_counts_missing_instances_as_zero(self):
        heldout = qap_api.load_split("heldout")
        _, _, perm = parse_sln(INSTANCES / "nug22.sln")
        code, res = post(self.url, {"submission": {"solutions": [{"instance": "nug22", "permutation": perm}]},
                                    "split": "heldout"})
        self.assertEqual(code, 200)
        self.assertAlmostEqual(res["score"], verify("nug22", perm)["score"] / len(heldout), places=12)
        self.assertEqual(res["n"], len(heldout))

    def test_unknown_instance_and_duplicates_are_rejected_not_scored(self):
        code, res = post(self.url, {"submission": {"instance": "nope", "permutation": [0]}})
        self.assertEqual((code, res["status"]), (422, "rejected"))
        sol = {"instance": "nug22", "permutation": list(range(22))}
        code, res = post(self.url, {"submission": {"solutions": [sol, sol]}})
        self.assertEqual((code, res["status"]), (422, "rejected"))

    def test_feedback_explains_result_against_two_swap_in_two_sentences(self):
        code, res = post(self.url, {"submission": {"instance": "nug22", "permutation": list(range(22))}})
        self.assertEqual(code, 200)
        text = summary(res["feedback"])
        self.assertTrue(text.startswith("Costs "), text)
        self.assertIn("2-swap baseline", text)
        self.assertLessEqual(sentence_count(text), 2, text)
        self.assertIn("Details:", res["feedback"])
        self.assertEqual(res["baseline"]["name"], "2-swap")

    def test_feedback_names_the_baseline_when_submission_is_the_baseline(self):
        heldout = qap_api.load_split("heldout")
        base = qap_api.two_swap_baseline()
        if not all(n in base for n in heldout) or not qap_api.TWO_SWAP_CACHE.is_file():
            self.skipTest("no 2-swap permutations cached for the held-out split")
        perms = json.loads(qap_api.TWO_SWAP_CACHE.read_text())
        code, res = post(self.url, {"submission": {"solutions": [{"instance": n, "permutation": perms[n]}
                                                                 for n in heldout]}, "split": "heldout"})
        self.assertEqual(code, 200)
        self.assertTrue(summary(res["feedback"]).startswith("This is the baseline:"), res["feedback"][:200])
        self.assertTrue(res["baseline"]["is_baseline"])
        self.assertAlmostEqual(res["baseline"]["score"], res["score"], places=12)

    def test_invalid_permutation_scores_zero(self):
        code, res = post(self.url, {"submission": {"instance": "nug22", "permutation": [0] * 22}})
        self.assertEqual(code, 200)
        self.assertEqual(res["score"], 0.0)


class AleBenchVerifierApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            scorer.require_data()
        except scorer.MissingDataError as exc:
            raise unittest.SkipTest(str(exc))
        cls.srv, cls.url = serve(ale_api.Handler)
        cls.greedy = scorer.GREEDY.read_text()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_program_total_matches_scorer_evaluate(self):
        seeds = ale_api.public_seeds()
        code, res = post(self.url, {"submission": {"program": self.greedy}})
        self.assertEqual(code, 200)
        direct = scorer.evaluate(self.greedy, seeds, scorer.contest_time_limit(), 5)
        self.assertEqual(res["score"], float(sum(r["score"] for r in direct)))
        self.assertEqual([c["score"] for c in res["cases"]], [r["score"] for r in direct])

    def test_feedback_compares_to_greedy_and_names_the_baseline(self):
        code, res = post(self.url, {"submission": {"program": self.greedy}})
        self.assertEqual(code, 200)
        self.assertTrue(summary(res["feedback"]).startswith("This is the baseline:"))
        self.assertEqual(res["baseline"]["score"], res["score"])
        code, res = post(self.url, {"submission": {"program": scorer.NAIVE.read_text()}})
        self.assertEqual(code, 200)
        text = summary(res["feedback"])
        self.assertIn("the greedy reference program", text)
        self.assertLessEqual(sentence_count(text), 2, text)
        self.assertFalse(res["baseline"]["is_baseline"])

    def test_output_score_matches_official_scorer(self):
        seed = 1
        with open(scorer.input_path(seed)) as fin:
            out = subprocess.run([sys.executable, "-I", str(scorer.GREEDY)], stdin=fin, capture_output=True,
                                 text=True, check=True, timeout=30).stdout
        code, res = post(self.url, {"submission": {"seed": seed, "output": out}})
        self.assertEqual(code, 200)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.txt"
            path.write_text(out)
            direct = scorer.score_files([(scorer.input_path(seed), path)])[0]
        self.assertEqual(direct["err"], "")
        self.assertEqual(res["score"], float(direct["score"]))

    def test_malformed_submission_is_rejected(self):
        code, res = post(self.url, {"submission": {"seed": -1, "output": "x"}})
        self.assertEqual((code, res["status"]), (422, "rejected"))


class DataHallVerifierApi(unittest.TestCase):
    """Rikin's donetwin verifier, unmodified, against donetwin's own scorer on Hall B."""

    def test_hall_b_greedy_score_matches_score_py(self):
        import verifier_server
        from donetwin import greedy
        from donetwin import score as scoring

        hall = json.loads((ROOT / "donetwin" / "scenarios" / "data_hall_2n.json").read_text())
        verifier_server.HALL.clear()
        verifier_server.HALL.update(hall)
        srv, url = serve(verifier_server.Handler)
        try:
            layout = greedy.solve(hall)
            code, res = post(url, {"submission": layout})
            self.assertEqual(code, 200)
            self.assertAlmostEqual(res["score"], float(scoring.score(hall, layout)["score"]), places=9)
            self.assertTrue(res["feasible"])
        finally:
            srv.shutdown()
            srv.server_close()


class DataHallHeldoutVerifierApi(unittest.TestCase):
    """benchmarks/datahall/verifier_api.py (port 9003) against donetwin's scorer on regenerated halls."""

    @classmethod
    def setUpClass(cls):
        from benchmarks.datahall import verifier_api as dh_api

        cls.srv, cls.url = serve(dh_api.Handler)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_greedy_score_matches_score_py_for_single_and_multi_hall(self):
        from donetwin import greedy, halls
        from donetwin import score as scoring

        keys = [(10003, "small"), (10011, "medium")]
        direct, items = [], []
        for seed, tier in keys:
            hall = halls.generate(seed, tier)
            layout = greedy.solve(hall)
            direct.append(float(scoring.score(hall, layout)["score"]))
            items.append({"seed": seed, "tier": tier, "layout": layout})
        code, res = post(self.url, {"submission": items[0]})
        self.assertEqual(code, 200)
        self.assertAlmostEqual(res["score"], direct[0], places=9)
        code, res = post(self.url, {"submission": {"halls": items}})
        self.assertEqual(code, 200)
        self.assertAlmostEqual(res["score"], sum(direct) / 2, places=9)

    def test_violating_layout_scores_zero_with_violations_in_feedback(self):
        from donetwin import greedy, halls

        layout = greedy.solve(halls.generate(10003, "small"))
        layout["racks"].append(dict(layout["racks"][0], id="R_DUP"))
        code, res = post(self.url, {"submission": {"seed": 10003, "tier": "small", "layout": layout}})
        self.assertEqual(code, 200)
        self.assertEqual(res["score"], 0.0)
        self.assertFalse(res["feasible"])
        self.assertTrue(res["violations"])
        self.assertIn("INFEASIBLE", res["feedback"])

    def test_feedback_compares_to_greedy_on_the_same_halls(self):
        from donetwin import greedy, halls

        items = [{"seed": s, "tier": t, "layout": greedy.solve(halls.generate(s, t))}
                 for s, t in ((10003, "small"), (10011, "medium"))]
        code, res = post(self.url, {"submission": {"halls": items}})
        self.assertEqual(code, 200)
        self.assertTrue(summary(res["feedback"]).startswith("This is the baseline:"))
        self.assertTrue(res["baseline"]["is_baseline"])
        self.assertAlmostEqual(res["baseline"]["score"], res["score"], places=9)
        items[0]["layout"]["racks"].pop()
        code, res = post(self.url, {"submission": {"halls": items}})
        self.assertEqual(code, 200)
        text = summary(res["feedback"])
        self.assertTrue(text.startswith("Scores "), text)
        self.assertIn("the greedy layout (", text)
        self.assertLessEqual(sentence_count(text), 2, text)
        self.assertFalse(res["baseline"]["is_baseline"])

    def test_bad_tier_and_duplicate_halls_are_rejected_not_scored(self):
        code, res = post(self.url, {"submission": {"seed": 1, "tier": "huge", "layout": {"racks": []}}})
        self.assertEqual((code, res["status"]), (422, "rejected"))
        item = {"seed": 1, "tier": "small", "layout": {"racks": []}}
        code, res = post(self.url, {"submission": {"halls": [item, item]}})
        self.assertEqual((code, res["status"]), (422, "rejected"))


class KernelBenchVerifierApi(unittest.TestCase):
    """benchmarks/kernelbench_mini/verifier_api.py (port 9005): baseline by definition, an error without a GPU."""

    @classmethod
    def setUpClass(cls):
        from benchmarks.kernelbench_mini import verifier_api as kb_api

        cls.kb = kb_api
        cls.srv, cls.url = serve(kb_api.Handler)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_pytorch_eager_is_the_baseline_at_speedup_one(self):
        sols = [{"task": t, "model_new": self.kb.eager_candidate(t)} for t in self.kb.heldout()]
        code, res = post(self.url, {"submission": {"solutions": sols}})
        self.assertEqual(code, 200)
        self.assertEqual((res["status"], res["score"]), ("completed", 1.0))
        expected = self.kb.NOT_RUN if self.kb.remote.gpu_mode() is None else None
        if expected:
            self.assertEqual(summary(res["feedback"]), expected)
        else:
            self.assertTrue(summary(res["feedback"]).startswith("This is the baseline:"))
        self.assertIn("Not timed", res["feedback"])

    def test_without_a_gpu_a_candidate_gets_an_error_not_a_score(self):
        if self.kb.remote.gpu_mode() is not None:
            self.skipTest("a GPU is attached")
        code, res = post(self.url, {"submission": {"solutions": [{"task": "kb_relu", "model_new": "class ModelNew: pass"}]}})
        self.assertEqual(code, 200)
        self.assertEqual(res["status"], "failed")
        self.assertIsNone(res["score"])
        self.assertIn("Needs a GPU VM, not yet attached", res["feedback"])

    def test_unknown_or_duplicate_task_is_rejected(self):
        code, res = post(self.url, {"submission": {"solutions": [{"task": "kb_nope", "model_new": "x"}]}})
        self.assertEqual((code, res["status"]), (422, "rejected"))
        sol = {"task": "kb_relu", "model_new": "x"}
        code, res = post(self.url, {"submission": {"solutions": [sol, sol]}})
        self.assertEqual((code, res["status"]), (422, "rejected"))


if __name__ == "__main__":
    unittest.main()
