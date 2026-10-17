"""Offline tests for donetwin.hill_climb (scripted FakeLLM, no network)."""
import copy
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

from donetwin import hill_climb

MISSING = [m for m in ("donetwin.score", "donetwin.greedy", "donetwin.halls") if importlib.util.find_spec(m) is None]
SKIP = f"waiting on parallel modules not yet written: {', '.join(MISSING)}" if MISSING else ""


class FakeLLM:
    """Same chat signature as donetwin.llm.LLM; replays scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, model, messages, max_tokens=4096, temperature=0.4, retries=5):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        return self.replies.pop(0) if self.replies else "no more scripted replies"


def quiet(*a, **k):
    pass


@unittest.skipIf(MISSING, SKIP)
class HillClimbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from donetwin import greedy, halls, score
        cls.score = score.score
        cls.hall = halls.generate(1, "small")
        cls.full = greedy.solve(cls.hall)
        cls.full_res = cls.score(cls.hall, cls.full)
        assert cls.full_res["feasible"] and cls.full["racks"], "greedy baseline must be feasible and non-empty"
        # find a rack whose removal keeps feasibility but lowers the score: full = reduced + one valid rack
        cls.reduced = None
        for i in reversed(range(len(cls.full["racks"]))):
            red = {"racks": [r for j, r in enumerate(cls.full["racks"]) if j != i]}
            rr = cls.score(cls.hall, red)
            if rr["feasible"] and rr["score"] < cls.full_res["score"]:
                cls.reduced, cls.reduced_res = red, rr
                break
        assert cls.reduced is not None, "could not derive an improvable incumbent from greedy"
        # infeasible (collision): duplicate the first rack at the same spot
        dup = dict(cls.full["racks"][0], id="DUP_1")
        cls.worse = {"racks": cls.full["racks"] + [dup]}
        assert cls.score(cls.hall, cls.worse)["score"] < cls.reduced_res["score"]

    def setUp(self):
        self.lib = tempfile.mkdtemp(prefix="hc_lib_")

    def tearDown(self):
        shutil.rmtree(self.lib, ignore_errors=True)

    def test_parse_fail_reject_accept(self):
        llm = FakeLLM(["I think the layout is fine, no changes needed!",
                       "```json\n" + json.dumps(self.worse) + "\n```",
                       "Here you go: " + json.dumps(self.full)])
        best, hist = hill_climb.solve(self.hall, llm=llm, iters=3, model="fake-model",
                                      library_dir=self.lib, start=self.reduced, log=quiet)
        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(llm.calls[0]["model"], "fake-model")
        self.assertEqual([h["iter"] for h in hist], [0, 1, 2, 3])
        self.assertAlmostEqual(hist[0]["score"], self.reduced_res["score"])
        self.assertEqual(hist[1].get("error"), "parse")
        self.assertFalse(hist[1]["accepted"])
        self.assertFalse(hist[2]["accepted"])
        self.assertTrue(hist[3]["accepted"])
        self.assertAlmostEqual(hist[3]["score"], self.full_res["score"])
        self.assertEqual(best, {"racks": self.full["racks"]})
        for h in hist:
            self.assertTrue({"iter", "score", "feasible", "accepted", "feedback_head"} <= set(h))
        # prompt carries hall summary, rules, incumbent and feedback; rejected attempt is fed back
        p0 = llm.calls[0]["messages"][-1]["content"]
        for needle in ("ROWS", "CDUs:", "HARD RULES", "INCUMBENT", "VERIFIER FEEDBACK"):
            self.assertIn(needle, p0)
        self.assertIn("REJECTED", llm.calls[2]["messages"][-1]["content"])
        # feasible best saved to the library
        lib = hill_climb.load_library(self.lib)
        self.assertEqual(len(lib), 1)
        self.assertAlmostEqual(lib[0]["score"], self.full_res["score"])

    def test_greedy_start_and_early_stop(self):
        llm = FakeLLM(["garbage", json.dumps(self.worse), "{not json"] + ["garbage"] * 5)
        best, hist = hill_climb.solve(self.hall, llm=llm, iters=6, library_dir=self.lib, start="greedy", log=quiet)
        self.assertEqual(len(llm.calls), 3, "must stop after 3 consecutive non-improving iterations")
        self.assertAlmostEqual(hist[0]["score"], self.full_res["score"])
        self.assertEqual(best, self.full)
        self.assertFalse(any(h["accepted"] for h in hist[1:]))

    def test_empty_start(self):
        llm = FakeLLM([json.dumps(self.full)])
        best, hist = hill_climb.solve(self.hall, llm=llm, iters=1, library_dir=self.lib, start="empty", log=quiet)
        self.assertEqual(hist[0]["start"], "empty")
        self.assertTrue(hist[1]["accepted"])
        self.assertEqual(best["racks"], self.full["racks"])

    def test_library_round_trip_and_only_if_better(self):
        s = self.full_res["score"]
        self.assertTrue(hill_climb.save_to_library(self.lib, self.hall, self.full, s))
        lib = hill_climb.load_library(self.lib)
        self.assertEqual(len(lib), 1)
        self.assertEqual(lib[0]["payload"], self.full)
        self.assertEqual(lib[0]["hall"]["name"], self.hall["name"])
        self.assertEqual(lib[0]["features"], hill_climb.features(self.hall))
        # worse or equal does not overwrite
        self.assertFalse(hill_climb.save_to_library(self.lib, self.hall, self.reduced, s - 1))
        self.assertFalse(hill_climb.save_to_library(self.lib, self.hall, self.reduced, s))
        self.assertEqual(hill_climb.load_library(self.lib)[0]["payload"], self.full)
        # better overwrites
        self.assertTrue(hill_climb.save_to_library(self.lib, self.hall, self.reduced, s + 1))
        self.assertEqual(hill_climb.load_library(self.lib)[0]["payload"], self.reduced)
        # solve() does not overwrite a better existing entry
        hill_climb.solve(self.hall, llm=FakeLLM(["x"] * 3), iters=3, library_dir=self.lib, start="greedy", log=quiet)
        self.assertEqual(hill_climb.load_library(self.lib)[0]["payload"], self.reduced)

    def test_nearest_examples_in_prompt(self):
        other = copy.deepcopy(self.hall)
        other["name"] = "other_hall"
        hill_climb.save_to_library(self.lib, other, self.full, 50.0)
        llm = FakeLLM(["garbage"])
        hill_climb.solve(self.hall, llm=llm, iters=1, library_dir=self.lib, start="empty", log=quiet)
        self.assertIn("VERIFIED EXAMPLE 1", llm.calls[0]["messages"][-1]["content"])
        self.assertTrue(os.path.exists(os.path.join(self.lib, "other_hall.json")))


class OfflineUnitTest(unittest.TestCase):
    """Pieces that need no parallel modules."""

    def test_parse_variants(self):
        self.assertEqual(hill_climb._parse('{"racks": [], "extra": 1}'), {"racks": []})
        self.assertEqual(hill_climb._parse('[{"id": "R1"}]'), {"racks": [{"id": "R1"}]})
        self.assertEqual(hill_climb._parse('{"payload": {"racks": []}}'), {"racks": []})
        with self.assertRaises(ValueError):
            hill_climb._parse("no json at all")

    def test_library_tolerates_missing_dir(self):
        self.assertEqual(hill_climb.load_library(os.path.join(tempfile.gettempdir(), "no_such_lib_dir_xyz")), [])


if __name__ == "__main__":
    unittest.main()
