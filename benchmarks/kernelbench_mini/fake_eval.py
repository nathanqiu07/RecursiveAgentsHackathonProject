"""A GPU-free stand-in for evaluate.py, for tests and `--offline` runs. Never used for a reported number.

It runs the REAL static checks, then models the runtime by a tag the scripted candidate carries:
  // fake: behaviour=<exact|identity|lowprec|cache> speedup=<x> compile=<ok|fail>
Correctness is modelled on signed (gauss) inputs at the evaluator's mixed scales, against a pure-Python
version of the reference: e.g. an "identity" ReLU matches max(x, 0) on all-positive torch.rand-style inputs
but fails on signed ones, which is exactly why evaluate.py draws torch.randn. Output schema = evaluate.py's.
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Callable

from benchmarks.kernelbench_mini.static_checks import check

TAG = re.compile(r"fake:\s*(.*)")
SCALES = (1.0, 0.1, 0.5, 0.01, 1.0)
ATOL = RTOL = 1e-4
REFERENCE: dict[str, Callable[[float], float]] = {
    "19_ReLU": lambda x: max(x, 0.0),
    "26_GELU_": lambda x: 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0))),
}


def _tag(src: str) -> dict[str, str]:
    m = TAG.search(src)
    if not m:
        return {}
    return dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)


def _behave(name: str, ref: Callable[[float], float], x: float, call: int) -> float:
    if name == "identity":
        return x
    if name == "lowprec":
        return float(f"{ref(x):.3g}")  # fp16 keeps ~3 significant digits
    if name == "cache":
        return ref(x) if call == 0 else ref(0.5)  # replays a stale output after the first call
    return ref(x)


class FakeEvaluator:
    def __init__(self, signed: bool = True, seed: int = 42) -> None:
        self.signed, self.seed = signed, seed
        self.calls: list[tuple[str, str]] = []

    def __call__(self, problem: str, src: str) -> dict[str, Any]:
        self.calls.append((problem, src))
        res: dict[str, Any] = {"problem": problem, "compiled": False, "correct": False, "speedup": 0.0, "score": 0.0,
                               "suspicious": False, "category": None, "reasons": [], "timings": {}, "device": "fake"}
        hits = check(src)
        if hits:
            res["reasons"] = [f"static_reject:{h['rule']}: {h['detail']}" for h in hits]
            res["category"] = f"static_reject:{hits[0]['rule']}"
            return res
        tag = _tag(src)
        if tag.get("compile") == "fail":
            res["reasons"].append("compile_error: RuntimeError: Error building extension (modelled)")
            res["category"] = "compile_error"
            return res
        res["compiled"] = True
        ref = REFERENCE.get(problem, lambda x: x * 1.0)
        rng = random.Random(self.seed)
        for k, scale in enumerate(SCALES):
            xs = [rng.gauss(0.0, scale) if self.signed else rng.random() * scale for _ in range(256)]
            for x in xs:
                want, got = ref(x), _behave(tag.get("behaviour", "exact"), ref, x, k)
                if abs(got - want) > ATOL + RTOL * abs(want):
                    res["reasons"].append(f"incorrect on {'signed' if self.signed else 'unsigned'} input set {k} "
                                          f"(scale {scale:g}): values differ, x={x:.4g} got {got:.4g} want {want:.4g}")
                    res["category"] = "incorrect"
                    return res
        speedup = float(tag.get("speedup", "1.0"))
        res.update(correct=True, speedup=speedup, score=min(speedup, 10.0),
                   timings={"ref_ms_median": 0.1, "cand_ms_median": round(0.1 / speedup, 5), "trials": 50})
        if speedup > 10.0:
            res["suspicious"] = True
            res["category"] = "suspicious"
            res["reasons"].append(f"suspicious: speedup {speedup:.2f}x > 10x, capped in the score")
        elif speedup <= 1.0:
            res["category"] = "slower"
        return res
