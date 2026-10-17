import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

from donetwin.score import COOLING, rebuild, score, upper_bound_kw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _bw(bid, y, feed, row):
    return {"id": bid, "kind": "busway", "x": 6.0, "y": y, "w": 10.0, "d": 0.25, "h": 0.15, "rot": 0, "fixed": True,
            "capacity_kw": 400, "derate": 0.8, "tap_m": 0.8, "feed": feed, "row": row}


def _cdu(cid, x):
    return {"id": cid, "kind": "cdu", "x": x, "y": 7.5, "w": 1.2, "d": 0.8, "h": 2.0, "rot": 0, "fixed": True,
            "capacity_kw": 300, "hose_m": 10.0, "points": {"port": [0.0, -0.4]}}


HALL = {
    "name": "toy", "title": "toy hall", "goal": "test", "floor": {"w": 12, "d": 8},
    "air_cooling_kw": 40, "aisle_min_m": 1.2, "domain_notes": "",
    "objects": [
        _bw("BW_R1", 3.75, "A", "R"), _bw("BW_R2", 4.25, "B", "R"),
        _bw("BW_S1", 1.35, "A", "S"), _bw("BW_S2", 1.85, "B", "S"),
        _cdu("CDU_1", 3.0), _cdu("CDU_2", 9.0),
        {"id": "P1", "kind": "pillar", "x": 6.0, "y": 6.0, "w": 0.5, "d": 0.5, "h": 4, "rot": 0, "fixed": True},
        {"id": "KO_1", "kind": "keepout", "x": 11.3, "y": 4.0, "w": 1.4, "d": 2.0, "h": 0, "rot": 0, "fixed": True},
        {"id": "W1", "kind": "walkway", "x": 0.5, "y": 4.0, "w": 1.0, "d": 8.0, "h": 0, "rot": 0, "fixed": True},
    ],
}


def rack(rid, model, x, y, rot, cdu, fa, fb):
    return {"id": rid, "model": model, "x": x, "y": y, "rot": rot, "cdu": cdu, "feed_a": fa, "feed_b": fb}


def valid():
    # row R faces -y, row S faces +y: front-to-front cold aisle between y=2.2 and 3.4 (1.2 m)
    return {"racks": [
        rack("R01", "GB200-NVL72", 3.3, 4.0, 180, "CDU_1", "BW_R1", "BW_R2"),
        rack("R02", "GB200-NVL72", 8.7, 4.0, 180, "CDU_2", "BW_R1", "BW_R2"),
        rack("H01", "HGX-B200", 3.3, 1.6, 0, None, "BW_S1", "BW_S2"),
        rack("H02", "HGX-B200", 3.9, 1.6, 0, None, "BW_S1", "BW_S2"),
    ]}


def check(res, name):
    return next(c for c in res["checks"] if c["name"] == name)


class ScoreTests(unittest.TestCase):
    def test_upper_bound(self):
        # busways 4 x 400 x 0.8 / 2 = 640; cooling 300 + 300 + 40 = 640
        self.assertAlmostEqual(upper_bound_kw(HALL), 640.0)
        hall = copy.deepcopy(HALL)
        hall["air_cooling_kw"] = 0
        self.assertAlmostEqual(upper_bound_kw(hall), 600.0)  # cooling-bound

    def test_upper_bound_dominates_greedy(self):
        from donetwin import greedy, halls
        for tier in ("small", "medium", "large"):
            for seed in range(10):
                hall = halls.generate(seed, tier)
                res = score(hall, greedy.solve(hall))
                with self.subTest(tier=tier, seed=seed):
                    self.assertLessEqual(res["it_kw"], res["upper_bound_kw"] + 1e-6)
                    self.assertLessEqual(res["components"]["density"], 1.0)

    def test_cooling_table(self):
        self.assertEqual(COOLING["HGX-B200"], "air")
        self.assertEqual(COOLING["GB300-NVL72"], "liquid")

    def test_valid_layout(self):
        res = score(HALL, valid())
        self.assertTrue(res["feasible"], res["feedback"])
        self.assertEqual(res["it_kw"], 294)
        self.assertEqual(res["components"]["resilience"], 1.0)
        self.assertAlmostEqual(res["components"]["hose"], 1 - 6.8 / 20, places=3)
        self.assertAlmostEqual(res["score"], round(70 * 294 / 640 + 20 + 10 * (1 - 6.8 / 20), 3), places=2)
        self.assertTrue(res["feedback"].startswith("feasible:"))

    def test_rebuild_uses_catalog(self):
        data, errs = rebuild(HALL, valid())
        self.assertEqual(errs, [])
        r = next(o for o in data["objects"] if o["id"] == "R01")
        self.assertEqual((r["kw"], r["w"], r["d"], r["cdu"], r["feed_b"]), (132, 0.6, 1.2, "CDU_1", "BW_R2"))

    def test_forged_fields(self):
        p = valid()
        p["racks"][0]["kw"] = 5000
        res = score(HALL, p)
        self.assertEqual(res["score"], 0)
        self.assertFalse(res["feasible"])
        self.assertFalse(check(res, "payload")["passed"])
        p = valid()
        p["hall"] = {"air_cooling_kw": 1e9}
        self.assertFalse(check(score(HALL, p), "payload")["passed"])
        for bad in ({"id": "CDU_1"}, {"rot": 45}, {"x": float("nan")}, {"model": "H100"}, {"x": True}):
            p = valid()
            p["racks"][0].update(bad)
            self.assertTrue(rebuild(HALL, p)[1], bad)
        p = valid()
        p["racks"][1]["id"] = "R01"
        self.assertTrue(rebuild(HALL, p)[1])

    def test_cooling_type(self):
        p = valid()
        p["racks"][2]["cdu"] = "CDU_1"
        res = score(HALL, p)
        self.assertFalse(check(res, "cooling_type")["passed"])
        self.assertEqual(res["score"], 0)
        p = valid()
        p["racks"][0]["cdu"] = None
        v = check(score(HALL, p), "cooling_type")
        self.assertFalse(v["passed"])
        self.assertIn("assign R01 to CDU_1", v["violations"][0]["fix"])

    def test_dual_feed(self):
        p = valid()
        p["racks"][0]["feed_b"] = "BW_R1"
        self.assertFalse(check(score(HALL, p), "dual_feed")["passed"])
        p = valid()
        p["racks"][0]["feed_b"] = "BW_S2"  # 1.7 m away
        v = check(score(HALL, p), "dual_feed")
        self.assertFalse(v["passed"])
        self.assertIn("feed_a=BW_R1, feed_b=BW_R2", v["violations"][0]["fix"])

    def test_power_2n(self):
        p = valid()
        p["racks"].append(rack("R03", "GB200-NVL72", 3.9, 4.0, 180, "CDU_1", "BW_R1", "BW_R2"))
        res = score(HALL, p)  # 396 kW on each of BW_R1/BW_R2 > 320
        self.assertFalse(check(res, "power_2n")["passed"])
        self.assertTrue(check(res, "cdu_capacity")["passed"])  # 264 <= 300
        self.assertEqual(res["score"], 0)

    def test_hot_cold_aisle(self):
        p = valid()
        p["racks"][0]["rot"] = 0  # R01 now faces +y; H01 front blows into R01's back
        v = check(score(HALL, p), "hot_cold_aisle")
        self.assertFalse(v["passed"])
        self.assertIn("H01", v["violations"][0]["detail"])
        p = valid()
        p["racks"][3]["rot"] = 180  # H02 abuts H01 in-row but faces the other way
        self.assertFalse(check(score(HALL, p), "hot_cold_aisle")["passed"])
        self.assertTrue(check(score(HALL, valid()), "hot_cold_aisle")["passed"])

    def test_geometry_rules(self):
        p = valid()
        p["racks"][1]["x"] = 10.9  # into keepout KO_1
        self.assertFalse(check(score(HALL, p), "zones_clear")["passed"])
        p = valid()
        p["racks"].append(rack("R03", "GB200-NVL72", 6.0, 5.9, 180, "CDU_2", "BW_R1", "BW_R2"))
        self.assertFalse(check(score(HALL, p), "no_collision")["passed"])
        p = valid()
        p["racks"][2]["y"] = p["racks"][3]["y"] = 1.9  # aisle 0.9 m
        self.assertFalse(check(score(HALL, p), "aisle_width")["passed"])

    def test_air_capacity(self):
        p = valid()
        p["racks"] += [rack(f"H{i}", "HGX-B200", 3.3 + 0.6 * i, 1.6, 0, None, "BW_S1", "BW_S2") for i in (2, 3)]
        self.assertFalse(check(score(HALL, p), "air_capacity")["passed"])

    def test_resilience_unreachable(self):
        hall = copy.deepcopy(HALL)
        for o in hall["objects"]:
            if o["kind"] == "cdu":
                o["hose_m"] = 6.0  # own runs 3.4 m; cross-hall re-home 8.8 m
        res = score(hall, valid())
        self.assertTrue(res["feasible"], res["feedback"])
        self.assertLess(res["components"]["resilience"], 1.0)
        self.assertFalse(check(res, "resilience")["passed"])

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            hp, pp = os.path.join(d, "h.json"), os.path.join(d, "p.json")
            for path, obj in ((hp, HALL), (pp, valid())):
                with open(path, "w") as f:
                    json.dump(obj, f)
            out = subprocess.run([sys.executable, "-m", "donetwin.score", hp, pp], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout
            self.assertTrue(json.loads(out)["feasible"])


if __name__ == "__main__":
    unittest.main()
