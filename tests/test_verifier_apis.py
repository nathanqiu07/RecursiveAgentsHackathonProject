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


if __name__ == "__main__":
    unittest.main()
