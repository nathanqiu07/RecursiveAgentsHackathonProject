import json
import os
import unittest

from donetwin import halls
from donetwin.checks import footprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestHalls(unittest.TestCase):
    def test_deterministic(self):
        for tier in halls.TIERS:
            a = json.dumps(halls.generate(42, tier), sort_keys=True)
            b = json.dumps(halls.generate(42, tier), sort_keys=True)
            self.assertEqual(a, b)

    def test_seeds_differ(self):
        a = halls.generate(1, "medium")
        b = halls.generate(2, "medium")
        self.assertNotEqual(json.dumps(a["objects"]), json.dumps(b["objects"]))
        self.assertNotEqual(json.dumps(halls.generate(1, "small")), json.dumps(halls.generate(1, "large")))

    def test_tiers_validate(self):
        for tier, (_, _, n_rows, n_cdu) in halls.TIERS.items():
            for seed in range(20):
                h = halls.generate(seed, tier)
                with self.subTest(tier=tier, seed=seed):
                    self.assertEqual(halls.validate(h), [])
                    self.assertEqual(h["name"], f"{tier}-{seed}")
                    for k in ("title", "goal", "floor", "objects", "domain_notes", "air_cooling_kw", "aisle_min_m"):
                        self.assertIn(k, h)
                    self.assertEqual(h["aisle_min_m"], 1.2)
                    self.assertTrue(40 <= h["air_cooling_kw"] <= 240)
                    W, D = h["floor"]["w"], h["floor"]["d"]
                    objs = h["objects"]
                    for o in objs:  # in bounds
                        x0, y0, x1, y1 = footprint(o).bounds
                        self.assertTrue(x0 >= -1e-6 and y0 >= -1e-6 and x1 <= W + 1e-6 and y1 <= D + 1e-6, o["id"])
                        self.assertTrue(o["fixed"])
                    phys = [o for o in objs if o["kind"] in ("pillar", "cdu", "panel")]
                    for i, a in enumerate(phys):  # no fixed physical overlaps
                        for b in phys[i + 1:]:
                            self.assertLess(footprint(a).intersection(footprint(b)).area, 1e-6, (a["id"], b["id"]))
                    bws = [o for o in objs if o["kind"] == "busway"]
                    self.assertEqual(len(bws), 2 * n_rows)
                    for rid in halls.ROW_IDS[:n_rows]:
                        a = next(o for o in bws if o["id"] == f"BW_{rid}1")
                        b = next(o for o in bws if o["id"] == f"BW_{rid}2")
                        self.assertEqual((a["feed"], b["feed"], a["row"], b["row"]), ("A", "B", rid, rid))
                        self.assertAlmostEqual(b["y"] - a["y"], 0.5, places=6)
                        self.assertEqual((a["x"], a["w"], a["capacity_kw"]), (b["x"], b["w"], b["capacity_kw"]))
                        self.assertTrue(600 <= a["capacity_kw"] <= 900)
                        self.assertGreater(a["y"] - 0.35, 1.4)  # rack strip above the south walkway
                        self.assertEqual((a["d"], a["h"], a["z"], a["derate"], a["tap_m"]), (0.25, 0.15, 3.0, 0.8, 0.8))
                    cdus = [o for o in objs if o["kind"] == "cdu"]
                    self.assertEqual(len(cdus), n_cdu)
                    for c in cdus:
                        self.assertEqual(c["points"]["port"], [0.0, -0.4])
                        self.assertAlmostEqual(c["y"], D - 0.8, places=2)
                        self.assertTrue(600 <= c["capacity_kw"] <= 900 and 10 <= c["hose_m"] <= 16)
                    self.assertEqual(sum(o["kind"] == "panel" for o in objs), 1)
                    self.assertGreaterEqual(sum(o["kind"] == "keepout" for o in objs), 1)

    def test_heldout(self):
        s = halls.heldout_seeds()
        self.assertEqual(len(s), 50)
        self.assertEqual(s[0], 10000)
        self.assertTrue(set(s).isdisjoint(halls.TRAIN_SEEDS))

    def test_data_hall_2n(self):
        with open(os.path.join(ROOT, "scenarios", "data_hall_2n.json"), encoding="utf-8") as f:
            h = json.load(f)
        self.assertEqual(h["name"], "data_hall_2n")
        self.assertEqual(h["air_cooling_kw"], 150)
        self.assertEqual(h["aisle_min_m"], 1.2)
        bws = [o for o in h["objects"] if o["kind"] == "busway"]
        self.assertEqual(len(bws), 8)
        self.assertEqual(sorted(o["feed"] for o in bws), ["A"] * 4 + ["B"] * 4)
        for rid, y in zip("ABCD", (3.4, 6.2, 9.8, 12.6)):
            a = next(o for o in bws if o["id"] == f"BW_{rid}1")
            b = next(o for o in bws if o["id"] == f"BW_{rid}2")
            self.assertAlmostEqual(a["y"], y - 0.25)
            self.assertAlmostEqual(b["y"], y + 0.25)
            self.assertEqual((a["feed"], b["feed"], a["row"], b["row"]), ("A", "B", rid, rid))
        self.assertEqual(halls.validate(h), [])
        with open(os.path.join(ROOT, "scenarios", "data_hall.json"), encoding="utf-8") as f:
            orig = json.load(f)
        keep = [o for o in orig["objects"] if o["kind"] != "busway"]
        self.assertEqual(keep, [o for o in h["objects"] if o["kind"] != "busway"])


if __name__ == "__main__":
    unittest.main()
