"""Mini KernelBench, KB_DEVICE=mps (Apple GPU via PyTorch MPS, not CUDA): device-aware static rules, prompts and the
local runner. The end-to-end cases score candidates on this Mac's GPU through the .venv-kb interpreter and skip when
that venv, torch or MPS is missing. Run: .venv/bin/python -m unittest tests/test_kernelbench_mps.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.kernelbench_mini import remote, team_runner  # noqa: E402
from benchmarks.kernelbench_mini.splits import load_tasks  # noqa: E402
from benchmarks.kernelbench_mini.static_checks import check, rules_summary  # noqa: E402

HEAD = "import torch\nimport torch.nn as nn\n"


def relu_src(forward: str, top: str = "", init: str = "") -> str:
    return (f"{HEAD}{top}\n\nclass ModelNew(nn.Module):\n    def __init__(self):\n        super().__init__()\n"
            f"{init}\n    def forward(self, x):\n{forward}\n")


def mps_rules(src: str) -> list[str]:
    return [h["rule"] for h in check(src, device="mps")]


TORCH_RELU = relu_src("        return torch.relu(x)")
MPS_HACKS = {
    "try_except": relu_src("        try:\n            return x.clamp(min=0)\n        except RuntimeError:\n"
                           "            return torch.relu(x)"),
    "timer_patch": relu_src("        return x.clamp(min=0)", top="torch.mps.synchronize = lambda: None"),
    "global_cache": relu_src("        return _C.setdefault(x.data_ptr(), x.clamp(min=0))", top="_C = {}"),
    "instance_cache": relu_src("        self.last = x.clamp(min=0)\n        return self.last"),
    "low_precision": relu_src("        return x.half().clamp(min=0).float()"),
    "introspection": relu_src("        import inspect\n        return inspect.currentframe().f_back.f_locals['ref_out']"),
    "forbidden_import": relu_src("        return x.clamp(min=0)", top="import subprocess"),
    "reference_op": relu_src("        return torch.compile(torch.relu)(x)"),
}
STATE_HACKS = [  # content-keyed caches need state on self; all blocked on MPS
    relu_src("        self.buf.copy_(x.clamp(min=0))\n        return self.buf",
             init="        self.buf = torch.empty(64, 16384)"),
    relu_src("        torch.clamp(x, min=0, out=self.buf)\n        return self.buf.clone()",
             init="        self.buf = torch.empty(64, 16384)"),
    relu_src("        y = torch.empty_like(x)\n        lib.relu(x, self.buf, y)\n        return y",
             top='lib = torch.mps.compile_shader("kernel void relu() {}")',
             init="        self.buf = torch.empty(64, 16384)"),
]
METAL_HALF = relu_src("        y = torch.empty_like(x)\n        lib.relu(x, y)\n        return y",
                      top='lib = torch.mps.compile_shader("kernel void relu(device const half* x, device half* y) {}")')
LEGIT_MSE = ("import torch\nimport torch.nn as nn\n\n\nclass ModelNew(nn.Module):\n    def __init__(self):\n"
             "        super().__init__()\n\n    def forward(self, predictions, targets):\n"
             "        return torch.nn.functional.mse_loss(predictions, targets)\n")


class MpsStaticRules(unittest.TestCase):
    def test_torch_ops_are_allowed_on_mps_but_still_rejected_on_cuda(self):
        self.assertEqual(mps_rules(TORCH_RELU), [])
        self.assertEqual(mps_rules(LEGIT_MSE), [])
        self.assertIn("reference_op", [h["rule"] for h in check(TORCH_RELU)])
        self.assertIn("functional_call", [h["rule"] for h in check(LEGIT_MSE)])

    def test_each_hack_is_still_rejected_on_mps(self):
        for rule, src in MPS_HACKS.items():
            self.assertIn(rule, mps_rules(src), rule)

    def test_state_on_self_is_rejected_on_mps(self):
        for src in STATE_HACKS:
            self.assertIn("instance_cache", mps_rules(src), src)

    def test_scalar_state_passed_to_a_kernel_is_allowed_but_a_registered_buffer_is_not(self):
        top = 'lib = torch.mps.compile_shader("kernel void relu() {}")'
        scalar = relu_src("        y = torch.empty_like(x)\n        lib.relu(x, y, self.eps)\n        return y", top=top,
                          init="        self.eps = 1e-5")
        self.assertEqual(mps_rules(scalar), [])
        buffer = relu_src("        y = torch.empty_like(x)\n        lib.relu(x, self.buf, y)\n        return y", top=top,
                          init='        self.register_buffer("buf", torch.empty(1))')
        self.assertIn("instance_cache", mps_rules(buffer))

    def test_metal_half_precision_is_rejected(self):
        self.assertIn("low_precision", mps_rules(METAL_HALF))

    def test_rules_summary_is_device_aware_and_cuda_text_unchanged(self):
        mps = rules_summary("mps")
        self.assertIn("reference_copy", mps)
        self.assertNotIn("load_inline", mps)
        cuda = rules_summary()
        self.assertIn("aten_op_in_cpp", cuda)
        self.assertNotIn("mps", cuda.lower())


class MpsPromptsAndRunner(unittest.TestCase):
    def test_task_text_names_the_apple_gpu_on_mps_only(self):
        task = load_tasks()[0][0]
        with mock.patch.dict(os.environ, {"KB_DEVICE": "mps"}):
            text = team_runner.task_text(task)
            v0 = team_runner.kb_v0()
        self.assertIn("Apple GPU via PyTorch MPS", text)
        self.assertIn("torch.mps.compile_shader", text)
        self.assertIn("CUDA is unavailable", text)
        self.assertIn("Apple GPU (MPS), not CUDA", v0.config["device"])
        self.assertIn("MPS", v0.prompts["kernel"])
        with mock.patch.dict(os.environ, {"KB_DEVICE": ""}):
            self.assertIn("__global__", team_runner.task_text(task))
            self.assertNotIn("device", team_runner.kb_v0().config)

    def test_gpu_mode_needs_the_kb_venv_and_env_is_scrubbed(self):
        with mock.patch.dict(os.environ, {"KB_DEVICE": "mps", "KB_PYTHON": "/nonexistent/python", "KB_REMOTE": ""}):
            self.assertIsNone(remote.gpu_mode())
        with mock.patch.dict(os.environ, {"KB_DEVICE": "mps", "KB_PYTHON": sys.executable, "KB_REMOTE": "",
                                          "CRUSOE_API_KEY": "secret-do-not-leak"}):
            self.assertEqual(remote.gpu_mode(), "mps")
            env = remote.mps_env()
        self.assertEqual(env["KB_DEVICE"], "mps")
        self.assertNotIn("CRUSOE_API_KEY", env)
        self.assertNotIn("secret-do-not-leak", "".join(env.values()))


def _mps_ready() -> bool:
    py = remote.DEFAULT_KB_PYTHON
    if not py.exists():
        return False
    run = subprocess.run([str(py), "-c", "import torch,sys; sys.exit(0 if torch.backends.mps.is_available() else 1)"],
                         capture_output=True, timeout=120)
    return run.returncode == 0


@unittest.skipUnless(_mps_ready(), "needs .venv-kb with torch and an MPS device")
class MpsEndToEnd(unittest.TestCase):
    def ev(self, problem: str, src: str) -> dict:
        with mock.patch.dict(os.environ, {"KB_DEVICE": "mps", "KB_REMOTE": "", "KB_PYTHON": ""}):
            return remote.evaluate_candidate(problem, src, trials=30)

    def test_replaying_the_reference_is_rejected(self):
        self.assertEqual(self.ev("19_ReLU", TORCH_RELU)["category"], "runtime_reject:reference_copy")

    def test_mutating_an_input_is_rejected(self):
        res = self.ev("19_ReLU", relu_src("        return x.clamp_(min=0)"))
        self.assertEqual(res["category"], "runtime_reject:input_mutated")

    def test_skipping_work_is_incorrect(self):
        self.assertEqual(self.ev("19_ReLU", relu_src("        return x.clone()"))["category"], "incorrect")

    def test_a_fused_pytorch_rewrite_is_correct_and_timed_on_mps(self):
        res = self.ev("94_MSELoss", LEGIT_MSE)
        self.assertTrue(res["correct"], res.get("reasons"))
        self.assertIn("Apple GPU (MPS), not CUDA", res["device"])
        self.assertGreaterEqual(res["timings"]["trials"], 30)


if __name__ == "__main__":
    unittest.main()
