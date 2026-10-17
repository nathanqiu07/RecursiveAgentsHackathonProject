"""Mini KernelBench, offline: static rules reject known hacks, the fake evaluator catches the ReLU identity on
signed inputs, splits are disjoint and frozen, no_gpu never crashes, run_team keeps the best, and --offline evolves."""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.kernelbench_mini import evaluate, remote, splits  # noqa: E402
from benchmarks.kernelbench_mini.fake_eval import FakeEvaluator  # noqa: E402
from benchmarks.kernelbench_mini.static_checks import check  # noqa: E402
from benchmarks.kernelbench_mini.team_runner import kb_v0, make_run_team, run_team  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_kernelbench_evolve as kb_script  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "benchmarks" / "kernelbench_mini"
GOOD = (HERE / "examples" / "relu_ok.py").read_text()
FORWARD = "        return relu_ext.relu_cuda(x.contiguous())"


def with_forward(body: str, extra_top: str = "") -> str:
    """The known-good kernel with its forward body (and optionally module-level code) swapped for a hack."""
    src = GOOD.replace(FORWARD, body)
    return src.replace("class ModelNew", extra_top + "\n\nclass ModelNew", 1) if extra_top else src


def rules(src: str) -> list[str]:
    return [h["rule"] for h in check(src)]


HACKS = {
    "reference_op": with_forward("        return torch.relu(x)"),
    "functional_call": with_forward("        return torch.nn.functional.relu(x)"),
    "try_except": with_forward("        try:\n            return relu_ext.relu_cuda(x)\n        except RuntimeError:\n"
                               "            return torch.relu(x)"),
    "cuda_stream": with_forward("        s = torch.cuda.Stream()\n        with torch.cuda.stream(s):\n"
                                "            y = relu_ext.relu_cuda(x)\n        return y"),
    "timer_patch": with_forward(FORWARD, extra_top="torch.cuda.Event.elapsed_time = lambda self, end: 1e-3"),
    "global_cache": with_forward("        key = x.data_ptr()\n        return _CACHE.setdefault(key, relu_ext.relu_cuda(x))",
                                 extra_top="_CACHE = {}"),
    "instance_cache": with_forward("        self.last = relu_ext.relu_cuda(x)\n        return self.last"),
    "low_precision": with_forward("        return relu_ext.relu_cuda(x.half()).float()"),
    "introspection": with_forward("        import inspect\n        return inspect.currentframe().f_back.f_locals['ref_out']"),
}


class StaticRules(unittest.TestCase):
    def test_known_good_kernel_passes(self):
        self.assertEqual(check(GOOD), [])

    def test_each_hack_is_rejected_with_its_rule_first(self):
        for rule, src in HACKS.items():
            with self.subTest(rule=rule):
                self.assertEqual(rules(src)[0], rule, check(src))

    def test_try_except_fallback_is_also_caught_by_kernelbench_checker(self):
        self.assertIn("kb_code_bypass", rules(HACKS["try_except"]))

    def test_side_stream_in_cuda_source_and_aten_op_in_cpp(self):
        self.assertIn("cuda_stream", rules(GOOD.replace("auto y = torch::empty_like(x);",
                                                        "cudaStream_t s; cudaStreamCreate(&s);\n    auto y = torch::empty_like(x);")))
        self.assertIn("aten_op_in_cpp", rules(GOOD.replace("auto y = torch::empty_like(x);",
                                                           "return at::relu(x);\n    auto y = torch::empty_like(x);")))

    def test_tensor_methods_matmul_operator_and_layer_calls_are_reference_ops(self):
        self.assertEqual(rules(with_forward("        return x.relu()"))[0], "reference_op")
        self.assertEqual(rules(with_forward("        return x @ x"))[0], "reference_op")
        conv = GOOD.replace("        super().__init__()", "        super().__init__()\n        self.conv = nn.Conv2d(3, 3, 3)")
        self.assertEqual(rules(conv.replace(FORWARD, "        return self.conv(x)"))[0], "reference_op")

    def test_layer_weights_and_extension_methods_named_like_ops_are_allowed(self):
        conv = GOOD.replace("        super().__init__()", "        super().__init__()\n        self.conv = nn.Conv2d(3, 3, 3)")
        self.assertEqual(check(conv.replace(FORWARD, "        return relu_ext.relu_cuda(self.conv.weight)")), [])
        renamed = GOOD.replace('functions=["relu_cuda"]', 'functions=["relu"]').replace("relu_ext.relu_cuda(", "relu_ext.relu(")
        self.assertEqual(check(renamed), [])

    def test_build_config_at_module_level_is_not_a_cache(self):
        cfg = 'import os\nos.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"\nFLAGS = ["-O3", "--use_fast_math"]'
        self.assertEqual(check(with_forward(FORWARD, extra_top=cfg)), [])
        writes = with_forward("        y = relu_ext.relu_cuda(x)\n        SEEN[0] = y\n        return y",
                              extra_top="SEEN = [None]")
        self.assertEqual(rules(writes)[0], "global_cache")

    def test_missing_modelnew_and_syntax_error(self):
        self.assertEqual(rules("import torch\nclass Foo: ..."), ["no_modelnew"])
        self.assertEqual(rules("def (:"), ["syntax_error"])

    def test_evaluate_rejects_statically_without_torch_or_gpu(self):
        res = evaluate.evaluate(HERE / "problems" / "19_ReLU.py", HACKS["try_except"])
        self.assertEqual(res["category"], "static_reject:try_except")
        self.assertFalse(res["compiled"])
        self.assertEqual(res["score"], 0.0)
        json.dumps(res)


class FakeEvaluatorModelsSignedInputs(unittest.TestCase):
    def identity_relu(self) -> str:
        return GOOD.replace("y[i] = v > 0.0f ? v : 0.0f;", "y[i] = v;  // fake: behaviour=identity speedup=1.3")

    def test_relu_identity_passes_unsigned_inputs_but_fails_signed(self):
        src = self.identity_relu()
        self.assertEqual(check(src), [])  # nothing static gives it away
        self.assertTrue(FakeEvaluator(signed=False)("19_ReLU", src)["correct"])
        res = FakeEvaluator(signed=True)("19_ReLU", src)
        self.assertFalse(res["correct"])
        self.assertEqual(res["category"], "incorrect")
        self.assertIn("signed", res["reasons"][0])

    def test_low_precision_and_output_cache_fail_tolerance(self):
        for behaviour in ("lowprec", "cache"):
            src = GOOD.replace("y[i] = v > 0.0f ? v : 0.0f;", f"y[i] = v > 0.0f ? v : 0.0f;  // fake: behaviour={behaviour}")
            self.assertEqual(FakeEvaluator()("26_GELU_", src)["category"], "incorrect", behaviour)

    def test_suspicious_speedup_is_flagged_and_capped(self):
        src = GOOD.replace("y[i] = v > 0.0f ? v : 0.0f;", "y[i] = v > 0.0f ? v : 0.0f;  // fake: speedup=40")
        res = FakeEvaluator()("19_ReLU", src)
        self.assertTrue(res["suspicious"])
        self.assertEqual(res["score"], 10.0)
        self.assertEqual(res["category"], "suspicious")


class Splits(unittest.TestCase):
    def test_disjoint_sized_seeded_and_frozen(self):
        s = splits.make_splits()
        self.assertEqual([len(s[k]) for k in ("train", "dev", "heldout")], [4, 3, 5])
        all_ids = s["train"] + s["dev"] + s["heldout"]
        self.assertEqual(len(set(all_ids)), 12)
        self.assertEqual(s, splits.make_splits())
        self.assertNotEqual(s, splits.make_splits(seed=1))
        frozen = json.loads(splits.SPLITS_JSON.read_text())
        self.assertEqual({k: frozen[k] for k in s}, s)
        self.assertEqual(frozen["heldout_sha256"], splits.heldout_hash(s["heldout"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            self.assertTrue(splits.freeze(s, path))
            with self.assertRaises(splits.FrozenSplitError):
                splits.freeze({**s, "heldout": s["dev"]}, path)

    def test_problem_files_have_license_and_reference_api(self):
        for pid, (file, _cat, text) in splits.PROBLEMS.items():
            src = (splits.PROBLEMS_DIR / file).read_text()
            names = {n.name for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            self.assertTrue({"Model", "get_inputs", "get_init_inputs"} <= names, pid)
            self.assertIn("MIT License", src)
            self.assertTrue(0 < len(text) <= 140, pid)
        self.assertTrue((splits.PROBLEMS_DIR / "LICENSE").exists())


class NoGpu(unittest.TestCase):
    def test_evaluate_candidate_returns_no_gpu(self):
        with mock.patch.dict(os.environ, {"KB_REMOTE": ""}), mock.patch.object(remote, "local_cuda", return_value=False):
            self.assertEqual(remote.evaluate_candidate("19_ReLU", GOOD)["error"], "no_gpu")

    def test_run_team_stops_at_no_gpu_without_crashing(self):
        llm = ScriptedLLM({"kernel": [f"TECHNIQUE: plain\n```python\n{GOOD}```"]})
        with mock.patch.dict(os.environ, {"KB_REMOTE": ""}), mock.patch.object(remote, "local_cuda", return_value=False):
            r = run_team(kb_v0(), None, splits.make_task("kb_relu", "train"), llm)
        self.assertEqual((r.score, r.passed, r.failures, r.calls), (0.0, False, ["no_gpu"], 1))

    def test_live_without_gpu_exits_3_before_any_llm(self):
        with mock.patch.object(remote, "gpu_mode", return_value=None), \
                mock.patch.object(kb_script, "CrusoeLLM", side_effect=AssertionError("no LLM")):
            self.assertEqual(kb_script.main(["--live", "--one"]), 3)

    def test_remote_mode_ships_evaluator_once_and_parses_json(self):
        import subprocess
        cmds = []

        def fake_run(cmd, timeout):
            cmds.append(cmd)
            out = json.dumps({"problem": "19_ReLU", "compiled": True, "correct": True, "speedup": 1.4}) if "evaluate.py" in cmd[-1] else ""
            return subprocess.CompletedProcess(cmd, 0, stdout="building...\n" + out + "\n", stderr="")

        remote._SYNCED.clear()
        with mock.patch.dict(os.environ, {"KB_REMOTE": "ubuntu@10.0.0.1", "CRUSOE_API_KEY": "sk-test"}), \
                mock.patch.object(remote, "_run", side_effect=fake_run):
            first = remote.evaluate_candidate("19_ReLU", GOOD)
            remote.evaluate_candidate("19_ReLU", GOOD)
        self.assertEqual((first["speedup"], first["host"]), (1.4, "ubuntu@10.0.0.1"))
        flat = [" ".join(c) for c in cmds]
        self.assertEqual(sum("problems/" in f and f.startswith("scp") for f in flat), 1)  # synced once
        self.assertEqual(sum("evaluate.py --problem problems/19_ReLU.py" in f for f in flat), 2)
        self.assertFalse(any("sk-test" in f for f in flat))

    def test_ssh_children_never_see_api_keys(self):
        with mock.patch.dict(os.environ, {"CRUSOE_API_KEY": "sk-test", "OPENAI_API_KEY": "x", "HOME": "/h"}):
            env = remote.ssh_env()
        self.assertFalse([k for k in env if "KEY" in k or "TOKEN" in k])
        self.assertEqual(env["HOME"], "/h")


class FakeBrain:
    def __init__(self) -> None:
        self.reads: list[tuple[str, str]] = []

    def read(self, role, text, budget_chars=3000):
        self.reads.append((role, text))
        return "- softmax: one warp per row was correct at 1.8x", ["e1", "e2"]


class RunTeam(unittest.TestCase):
    def test_keeps_best_records_brain_ids_and_passes_thinking_and_model(self):
        task = splits.make_task("kb_relu", "train")
        tagged = lambda s: GOOD.replace("y[i] = v > 0.0f ? v : 0.0f;", f"y[i] = v > 0.0f ? v : 0.0f;  // fake: speedup={s}")
        llm = ScriptedLLM({"kernel": [f"TECHNIQUE: fast\n```python\n{tagged(1.8)}```",
                                      f"TECHNIQUE: try\n```python\n{HACKS['try_except']}```",
                                      f"TECHNIQUE: slow\n```python\n{tagged(0.7)}```"]})
        v = kb_v0()
        v.config.update(thinking={"kernel": True}, models={"kernel": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"})
        brain = FakeBrain()
        r = make_run_team(FakeEvaluator())(v, brain, task, llm)
        self.assertEqual((r.score, r.passed, r.calls, r.iterations), (1.8, True, 3, 3))
        self.assertEqual(r.failures, ["static_reject:try_except", "slower"])
        self.assertEqual(r.brain_ids_read, ["e1", "e2"])
        self.assertEqual(brain.reads, [("kernel", task.text)])
        self.assertTrue(all(c["thinking"] is True for c in llm.calls))
        self.assertEqual({c["model"] for c in llm.calls}, {"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"})
        second_prompt = llm.calls[1]["messages"][-1]["content"]
        self.assertIn("Incumbent (best so far): fast, correct, 1.800x", second_prompt)
        self.assertIn("one warp per row", llm.calls[0]["messages"][0]["content"])
        self.assertIn("class Model", llm.calls[0]["messages"][1]["content"])

    def test_incorrect_only_scores_zero_and_slower_is_not_passed(self):
        task = splits.make_task("kb_relu", "train")
        ident = GOOD.replace("y[i] = v > 0.0f ? v : 0.0f;", "y[i] = v;  // fake: behaviour=identity")
        r = make_run_team(FakeEvaluator())(kb_v0(), None, task, ScriptedLLM({"kernel": [f"```python\n{ident}```"]}))
        self.assertEqual((r.score, r.passed, r.failures), (0.0, False, ["incorrect"]))
        slow = GOOD.replace("y[i] = v > 0.0f ? v : 0.0f;", "y[i] = v > 0.0f ? v : 0.0f;  // fake: speedup=0.5")
        r = make_run_team(FakeEvaluator())(kb_v0(), None, task, ScriptedLLM({"kernel": [f"```python\n{slow}```"]}))
        self.assertEqual((r.score, r.passed, r.failures), (0.5, False, ["slower"]))


class OfflineEvolve(unittest.TestCase):
    def test_offline_evolve_runs_and_reports_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(kb_script.main(["--offline", "--generations", "2", "--out", tmp, "--workers", "2"]), 0)
            report = json.loads((Path(tmp) / "reports" / "before_after.json").read_text())
        rows = report["rows"]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["n"], 5)
        for key in ("fast_1", "correct_rate", "mean_speedup_correct", "calls_per_task", "tokens_per_task", "cost_per_task_usd"):
            self.assertIn(key, rows[0])
        self.assertIn("static_reject:try_except", rows[0]["failures"])
        self.assertGreaterEqual(rows[2]["mean_score"], rows[0]["mean_score"])

    def test_offline_one(self):
        self.assertEqual(kb_script.main(["--offline", "--one"]), 0)


if __name__ == "__main__":
    unittest.main()
