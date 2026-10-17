"""Score one candidate ModelNew against a KernelBench problem. Runs ON the CUDA host; no LLM, no network.

  python evaluate.py --problem problems/19_ReLU.py --candidate cand.py [--trials 50] [--seed 42]

Prints one JSON object on the last stdout line:
  {"problem", "compiled", "correct", "speedup", "score", "suspicious", "category", "reasons", "timings", ...}

Order: static checks (named rejects) -> compile ModelNew -> runtime torch-op audit -> correctness on 5 fresh
SIGNED input sets (torch.randn at mixed scales; dtype, shape, values at atol=rtol=1e-4, inputs not mutated)
-> timing (CUDA events, L2 flush before every trial, median of `trials`, reference = torch eager on the same
inputs) -> one more correctness check on a seed never used for timing, after the candidate has been called
many times (catches output caching). speedup = ref median / candidate median; above 10x it is flagged
"suspicious" and capped at 10 in the score. score = capped speedup if correct else 0.
Stdlib + torch only; static checks are importable without torch so the same rules run on a laptop.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from statistics import median
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import static_checks  # noqa: E402

ATOL = RTOL = 1e-4
# One scale per fresh input set; randn is signed, so ReLU(x) != x. Capped at 1 so fp32 accumulation-order
# differences stay well inside atol=rtol=1e-4 for the K=256 matmuls and the 512-long cumsum.
CORRECTNESS_SCALES = (1.0, 0.1, 0.5, 0.01, 1.0)
SPEEDUP_CAP = 10.0
TIMING_SEED_OFFSET = 1000
POSTCHECK_SEED_OFFSET = 7919
# Tensor-level torch calls a ModelNew.forward may make around its own kernel (allocation, layout, metadata).
ALLOWED_TORCH_FUNCS = frozenset("""
empty empty_like empty_strided zeros zeros_like new_empty new_zeros contiguous view reshape view_as flatten
unsqueeze squeeze expand size stride numel dim shape dtype device is_contiguous data_ptr storage_offset
is_cuda element_size __get__ __len__ __getitem__ __index__ __int__ __bool__ __hash__ __format__ __repr__
__eq__ __ne__ get_device cuda to float requires_grad_ detach is_floating_point ndimension nelement layout
""".split())


def result_template(problem: str) -> dict[str, Any]:
    return {"problem": problem, "compiled": False, "correct": False, "speedup": 0.0, "score": 0.0,
            "suspicious": False, "category": None, "reasons": [], "timings": {}, "device": None}


def finish(res: dict[str, Any], category: str | None) -> dict[str, Any]:
    res["category"] = category
    if res["correct"]:
        res["score"] = round(min(res["speedup"], SPEEDUP_CAP), 4)
        if res["speedup"] > SPEEDUP_CAP:
            res["suspicious"] = True
            res["category"] = "suspicious"
            res["reasons"].append(f"suspicious: speedup {res['speedup']:.2f}x > {SPEEDUP_CAP:g}x, capped in the score")
        elif res["speedup"] <= 1.0:
            res["category"] = "slower"
    return res


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def signed_like(x: Any, gen: Any, scale: float) -> Any:
    """Floating tensors become randn * scale (signed, mixed scales); everything else is kept as drawn."""
    import torch
    if isinstance(x, torch.Tensor) and x.is_floating_point():
        return (torch.randn(x.shape, generator=gen, dtype=x.dtype) * scale).cuda()
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return x


def make_inputs(problem_mod: Any, seed: int, scale: float) -> list[Any]:
    import torch
    torch.manual_seed(seed)  # shapes, non-float inputs and any extra draws in get_inputs are seeded too
    template = problem_mod.get_inputs()
    gen = torch.Generator().manual_seed(seed)
    return [signed_like(x, gen, scale) for x in template]


def clone_all(xs: list[Any]) -> list[Any]:
    import torch
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in xs]


def as_list(out: Any) -> list[Any]:
    return list(out) if isinstance(out, (tuple, list)) else [out]


def compare(ref_out: Any, out: Any) -> str | None:
    """None if equal within tolerance, else a short reason."""
    import torch
    refs, outs = as_list(ref_out), as_list(out)
    if len(refs) != len(outs):
        return f"returned {len(outs)} outputs, expected {len(refs)}"
    for i, (r, o) in enumerate(zip(refs, outs)):
        if not isinstance(o, torch.Tensor):
            return f"output {i} is {type(o).__name__}, expected Tensor"
        if o.dtype != r.dtype:
            return f"wrong_dtype: output {i} is {o.dtype}, expected {r.dtype}"
        if tuple(o.shape) != tuple(r.shape):
            return f"wrong_shape: output {i} is {tuple(o.shape)}, expected {tuple(r.shape)}"
        if not o.is_cuda:
            return f"output {i} is on {o.device}, expected cuda"
        if r.is_floating_point():
            if not torch.isfinite(o).all() and torch.isfinite(r).all():
                return f"output {i} has NaN/Inf"
            if not torch.allclose(o, r, atol=ATOL, rtol=RTOL):
                diff = (o - r).abs().max().item()
                return f"values differ: max abs diff {diff:.3g} (atol=rtol={ATOL:g})"
        elif not torch.equal(o, r):
            return f"values differ: {(o != r).sum().item()} of {r.numel()} elements"
    return None


def mutated(before: list[Any], after: list[Any]) -> int | None:
    import torch
    for i, (b, a) in enumerate(zip(before, after)):
        if isinstance(b, torch.Tensor) and not torch.equal(b, a):
            return i
    return None


def audit_torch_calls(model: Any, inputs: list[Any]) -> list[str]:
    """Every Python-level torch function/method/operator ModelNew.forward calls, minus allocation and layout.

    Catches `A * s`, `x.relu()`, `F.gelu`, `@` and anything the static rules missed. Calls made from C++
    inside the compiled extension are not Python-level and are not seen (aten_op_in_cpp covers those).
    """
    import torch
    from torch.overrides import TorchFunctionMode
    seen: list[str] = []

    class Audit(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            name = getattr(func, "__name__", str(func))
            if name not in ALLOWED_TORCH_FUNCS and name not in seen:
                seen.append(name)
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), Audit():
        model(*inputs)
    torch.cuda.synchronize()
    return seen


def flush_l2(buf: Any) -> None:
    buf.fill_(42)  # 256 MB write: larger than any current L2 (A100 40 MB, H100 50 MB, L40S 96 MB)


def time_model(model: Any, inputs: list[Any], trials: int, warmup: int, l2_buf: Any) -> list[float]:
    import torch
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(trials):
            flush_l2(l2_buf)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return times


def evaluate(problem_path: str | Path, candidate_src: str, trials: int = 50, seed: int = 42,
             warmup: int = 5) -> dict[str, Any]:
    problem_path = Path(problem_path)
    res = result_template(problem_path.stem)
    hits = static_checks.check(candidate_src)
    if hits:
        res["reasons"] = [f"static_reject:{h['rule']}: {h['detail']}" for h in hits]
        return finish(res, f"static_reject:{hits[0]['rule']}")
    import torch
    if not torch.cuda.is_available():
        res["reasons"].append("no CUDA device on this host")
        return finish(res, "no_gpu")
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 reference must be true fp32
    torch.backends.cudnn.allow_tf32 = False
    res["device"] = torch.cuda.get_device_name(0)
    digest = hashlib.sha256(candidate_src.encode()).hexdigest()[:12]
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(Path(tempfile.gettempdir()) / "kbmini_ext"))
    # load_inline shells out to `ninja`, which pip installs next to this interpreter, not on the default PATH.
    os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), "/usr/local/cuda/bin",
                                          os.environ.get("PATH", "/usr/bin:/bin")])
    problem = load_module(problem_path, f"kbmini_ref_{problem_path.stem}")
    with tempfile.TemporaryDirectory(prefix="kbmini_cand_") as tmp:
        cand_path = Path(tmp) / f"cand_{digest}.py"
        cand_path.write_text(candidate_src)
        t0 = time.monotonic()
        try:
            cand = load_module(cand_path, f"kbmini_cand_{digest}")
            init = problem.get_init_inputs()
            torch.manual_seed(seed)
            ref_model = problem.Model(*init).cuda().eval()
            torch.manual_seed(seed)  # same seed, so parameter-holding problems (conv) get the same weights
            model = cand.ModelNew(*init).cuda().eval()
            torch.cuda.synchronize()
        except Exception as exc:  # compile or construction failure is a scored outcome, not a crash
            res["reasons"].append(f"compile_error: {type(exc).__name__}: {str(exc)[-1500:]}")
            res["timings"]["compile_s"] = round(time.monotonic() - t0, 2)
            return finish(res, "compile_error")
        res["compiled"] = True
        res["timings"]["compile_s"] = round(time.monotonic() - t0, 2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            return _run_checks(res, problem, ref_model, model, trials, seed, warmup)
        except Exception as exc:  # e.g. illegal memory access inside the kernel
            res["correct"] = False
            res["reasons"].append(f"runtime_error: {type(exc).__name__}: {str(exc)[-800:]}")
            return finish(res, "incorrect")


def _run_checks(res: dict[str, Any], problem: Any, ref_model: Any, model: Any, trials: int, seed: int,
                warmup: int) -> dict[str, Any]:
    import torch
    audit = audit_torch_calls(model, make_inputs(problem, seed, 1.0))
    if audit:
        res["reasons"].append(f"runtime_reject:runtime_torch_op: forward calls torch {', '.join(audit[:6])}")
        return finish(res, "runtime_reject:runtime_torch_op")
    with torch.no_grad():
        for k, scale in enumerate(CORRECTNESS_SCALES):
            inputs = make_inputs(problem, seed + 1 + k, scale)
            ref_out = ref_model(*clone_all(inputs))
            before = clone_all(inputs)
            out = model(*inputs)
            torch.cuda.synchronize()
            bad = mutated(before, inputs)
            if bad is not None:
                res["reasons"].append(f"runtime_reject:input_mutated: input {bad} changed in place (set {k}, scale {scale:g})")
                return finish(res, "runtime_reject:input_mutated")
            why = compare(ref_out, out)
            if why:
                res["reasons"].append(f"incorrect on signed input set {k} (scale {scale:g}): {why}")
                return finish(res, "incorrect")
    l2_buf = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
    timing_inputs = make_inputs(problem, seed + TIMING_SEED_OFFSET, 1.0)
    ref_ms = time_model(ref_model, timing_inputs, trials, warmup, l2_buf)
    cand_ms = time_model(model, timing_inputs, trials, warmup, l2_buf)
    del l2_buf
    with torch.no_grad():  # a seed the timing loop never saw, after many calls: output caches fail here
        inputs = make_inputs(problem, seed + POSTCHECK_SEED_OFFSET, 1.0)
        ref_out = ref_model(*clone_all(inputs))
        out = model(*inputs)
        torch.cuda.synchronize()
        why = compare(ref_out, out)
    if why:
        res["reasons"].append(f"incorrect after timing on a fresh seed (output caching?): {why}")
        return finish(res, "incorrect")
    res["correct"] = True
    ref_med, cand_med = median(ref_ms), median(cand_ms)
    res["speedup"] = round(ref_med / cand_med, 4) if cand_med > 0 else float("inf")
    res["timings"].update({"ref_ms_median": round(ref_med, 5), "cand_ms_median": round(cand_med, 5),
                           "trials": trials, "ref_ms": [round(t, 5) for t in ref_ms],
                           "cand_ms": [round(t, 5) for t in cand_ms]})
    return finish(res, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", required=True, help="path to a problems/*.py file")
    ap.add_argument("--candidate", required=True, help="path to the candidate ModelNew source")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    try:
        res = evaluate(args.problem, Path(args.candidate).read_text(), args.trials, args.seed, args.warmup)
    except Exception as exc:  # the caller always gets one JSON line
        res = {**result_template(Path(args.problem).stem), "category": "eval_error",
               "reasons": [f"eval_error: {type(exc).__name__}: {exc}"], "traceback": traceback.format_exc()[-2000:]}
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
