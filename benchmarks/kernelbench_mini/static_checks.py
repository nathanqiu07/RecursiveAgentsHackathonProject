"""Static checks on a candidate `ModelNew` source, run before anything is compiled. Stdlib only.

Our AST rules run first, so a known hack gets a specific, stable rule name; KernelBench's own
pattern checker (vendored in kb_static_checker.py, MIT) runs after them as a second net and its
hits are named `kb_<check>`. `check(src)` returns a list of {"rule", "detail"}; empty means pass.

Rules (all reject):
  syntax_error      the file does not parse
  no_modelnew       no `class ModelNew`
  reference_model   uses the reference `Model` class (subclass, instance or call)
  reference_op      a torch compute op, a compute Tensor method, `@`, torch.ops/aten/compile,
                    or an nn layer's forward, called outside ModelNew.__init__
  functional_call   any torch.nn.functional / F.* call
  try_except        any try/except (or contextlib.suppress): the classic silent fallback
  cuda_stream       side CUDA streams, in Python or in the C++/CUDA source
  timer_patch       assigning to time.*, torch.cuda.Event/synchronize/elapsed_time and friends
  monkey_patch      assigning to any other torch/numpy/builtins attribute (e.g. allow_tf32)
  global_cache      empty module-level containers or ones written at call time, class-level
                    containers, global/nonlocal, lru_cache
  instance_cache    ModelNew stores state on self outside __init__ (output caching)
  low_precision     fp16/bf16/TF32 types or conversions in an fp32 problem
  aten_op_in_cpp    the C++ source calls an ATen compute op (at::matmul, x.sum(), ...)
  vendor_library    cuBLAS / cuDNN / cuTENSOR instead of a hand-written kernel
  introspection     frame/gc/inspect/exec tricks that could read the evaluator's reference output
  forbidden_import  subprocess, ctypes, sockets, pickle, importlib, threading, multiprocessing

device="mps" (Apple GPU via PyTorch MPS, where load_inline CUDA cannot build): PyTorch ops and Metal kernels
via torch.mps.compile_shader are the allowed tools, so the CUDA-only rules (reference_op on torch compute ops
and Tensor methods, functional_call, aten_op_in_cpp, vendor_library, kb_cuda_impl, kb_torch_computation_ops)
are off. Every hack rule stays on, and the evaluator adds a runtime reference_copy reject.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import kb_static_checker as kb  # noqa: E402

# torch.<op>(...) and <tensor>.<op>(...) that compute the answer instead of a custom kernel.
COMPUTE_OPS = frozenset("""
matmul mm bmm addmm addmv addbmm baddbmm mv dot vdot einsum tensordot inner outer linear bilinear
relu relu_ gelu silu selu elu leaky_relu hardtanh hardsigmoid softplus softsign mish sigmoid tanh logit
softmax log_softmax exp exp_ expm1 log log1p log2 sqrt rsqrt pow square abs neg reciprocal erf erfc sign
clamp clamp_ clip max min maximum minimum amax amin argmax argmin aminmax sum nansum mean nanmean prod
cumsum cumsum_ cumprod cummax cummin logcumsumexp norm var std var_mean std_mean logsumexp
layer_norm rms_norm group_norm batch_norm instance_norm normalize dropout
conv1d conv2d conv3d conv_transpose1d conv_transpose2d conv_transpose3d convolution unfold fold
max_pool1d max_pool2d max_pool3d avg_pool1d avg_pool2d avg_pool3d
mse_loss l1_loss smooth_l1_loss huber_loss cross_entropy nll_loss kl_div binary_cross_entropy
where mul mul_ multiply add add_ sub sub_ subtract div div_ divide true_divide fmod remainder
sort topk kthvalue median mode cdist triu tril diag diagonal matrix_power
""".split())
TORCH_SUBMODULES = frozenset({"linalg", "special", "fft", "ops", "_C", "_ops", "jit", "compile", "_dynamo", "_inductor",
                              "_refs", "_prims", "_decomp"})
# On MPS, PyTorch ops are allowed, but these still hide work from the runtime audit or compile behind it.
MPS_BANNED_SUBMODULES = frozenset({"ops", "_C", "_ops", "jit", "compile", "_dynamo", "_inductor", "_refs", "_prims",
                                   "_decomp"})
MPS_OFF_RULES = frozenset({"functional_call", "aten_op_in_cpp", "vendor_library", "kb_cuda_impl",
                           "kb_torch_computation_ops"})
MPS_RULES = """Rules (all reject):
  syntax_error      the file does not parse
  no_modelnew       no `class ModelNew`
  reference_model   uses the reference `Model` class (subclass, instance or call)
  reference_op      torch.ops/aten/_C, torch.compile/jit/_dynamo/_inductor called outside ModelNew.__init__
  reference_copy    (runtime) forward issues exactly the reference's sequence of torch ops: no optimization
  try_except        any try/except (or contextlib.suppress): the classic silent fallback
  cuda_stream       side streams
  timer_patch       assigning to time.*, torch.mps/cuda synchronize, Event, elapsed_time and friends
  monkey_patch      assigning to any other torch/numpy/builtins attribute
  global_cache      empty module-level containers or ones written at call time, class-level
                    containers, global/nonlocal, lru_cache
  instance_cache    ModelNew stores or mutates state on self outside __init__ (output caching): assigning
                    self.x, in-place ops or out= on self tensors, self tensors handed to a Metal kernel
  low_precision     fp16/bf16 types or conversions in an fp32 problem
  introspection     frame/gc/inspect/exec tricks that could read the evaluator's reference output
  forbidden_import  subprocess, ctypes, sockets, pickle, importlib, threading, multiprocessing
  input_mutated     (runtime) an input changed in place
  output caching    (runtime) a final correctness check on a fresh seed after timing"""
NN_CONTAINERS = frozenset({"Module", "Parameter", "ParameterList", "ParameterDict", "ModuleList", "ModuleDict", "init",
                           "parameter"})
MUTABLE_FACTORIES = frozenset({"dict", "list", "set", "defaultdict", "OrderedDict", "Counter", "deque",
                               "WeakKeyDictionary", "WeakValueDictionary"})
MUTATORS = frozenset({"append", "extend", "insert", "update", "setdefault", "add", "__setitem__", "pop",
                      "popitem", "appendleft"})
PATCH_ROOTS = frozenset({"torch", "time", "triton", "numpy", "np", "builtins", "nn", "F", "math", "os", "sys"})
TIMER_WORDS = ("time", "Event", "synchronize", "elapsed", "perf_counter", "record", "monotonic", "process_time")
FORBIDDEN_IMPORTS = frozenset({"subprocess", "ctypes", "socket", "urllib", "requests", "http", "pickle", "importlib",
                               "threading", "multiprocessing", "concurrent", "asyncio", "signal", "shutil"})
INTROSPECTION_MODULES = frozenset({"inspect", "gc", "builtins", "traceback", "dis", "code", "codeop"})
INTROSPECTION_NAMES = frozenset({"__import__", "exec", "eval", "compile", "globals", "locals", "vars", "breakpoint"})
INTROSPECTION_ATTRS = frozenset({"_getframe", "f_back", "f_locals", "f_globals", "__subclasses__", "__code__",
                                 "__globals__", "__builtins__", "get_objects", "get_referrers", "currentframe", "stack"})

CPP_STREAM = re.compile(r"cudaStreamCreate|getStreamFromPool|cudaStreamNonBlocking|cudaStreamPerThread|"
                        r"cudaStreamLegacy|cudaLaunchHostFunc|cudaStreamAddCallback|cudaGraphLaunch")
CPP_ATEN = re.compile(r"\b(?:at|torch)::(?:native::|_ops::|cuda::blas::)?(" + "|".join(sorted(
    COMPUTE_OPS - {"where", "sign", "abs", "max", "min", "exp", "log", "sqrt", "pow"} | {"matmul_out", "mm_out",
                                                                                       "cudnn_convolution"})) + r")\s*\(")
CPP_TENSOR_METHOD = re.compile(r"[\w\)\]]\.(matmul|mm|bmm|relu|gelu|softmax|log_softmax|sum|mean|cumsum|argmax|argmin|"
                               r"amax|amin|sigmoid|tanh|clamp|mul|div|add|sub|pow|sqrt|exp|norm|var|std|logsumexp|"
                               r"conv2d|layer_norm|t|transpose|mm_out)\s*\(")
CPP_VENDOR = re.compile(r"\bcublas|\bcudnn|\bcutensor|\bcusparse", re.IGNORECASE)
LOW_PRECISION = re.compile(r"\b(__half2?|half2|__nv_bfloat16\w*|__nv_half\w*|__float2half\w*|__half2float|"
                           r"__float2bfloat16\w*|__bfloat162float|at::Half|at::BFloat16|at::kHalf|at::kBFloat16|"
                           r"torch::kHalf|torch::kFloat16|torch::kBFloat16|c10::Half|c10::BFloat16|"
                           r"precision::tf32|__float_to_tf32|cvt\.rna\.tf32|float16|bfloat16)\b|"
                           r"\.half\s*\(|allow_tf32|set_float32_matmul_precision|allow_fp16_reduced|allow_bf16_reduced")
METAL_LOW_PRECISION = re.compile(r"\b(?:half|bfloat)[234]?\b")  # Metal shader types, matched in string constants


def _chain(node: ast.AST) -> list[str]:
    """`torch.nn.functional.relu` -> ["torch", "nn", "functional", "relu"]; [] if not a plain name chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return []


def _receiver_root(node: ast.AST) -> ast.AST:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node


def _is_mutable_value(v: ast.AST | None) -> bool:
    if isinstance(v, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)):
        return True
    return isinstance(v, ast.Call) and (_chain(v.func)[-1:] or [""])[0] in MUTABLE_FACTORIES


def _is_empty_container(v: ast.AST | None) -> bool:
    """Caches start empty; a non-empty literal at module level is configuration (flags, include paths)."""
    if isinstance(v, ast.Dict):
        return not v.keys
    if isinstance(v, (ast.List, ast.Set)):
        return not v.elts
    return isinstance(v, ast.Call) and not v.args and (_chain(v.func)[-1:] or [""])[0] in MUTABLE_FACTORIES


TENSOR_METHODS = frozenset({"clone", "to", "empty_like", "zeros_like", "ones_like", "new_empty", "new_zeros",
                            "new_ones", "contiguous", "detach", "Parameter", "mps", "float"})


def _is_tensor_call(v: ast.AST | None) -> bool:
    """A call that yields a tensor a kernel could write: torch.<factory>(...), nn.Parameter, .clone(), .to() ..."""
    if not isinstance(v, ast.Call):
        return False
    c = _chain(v.func)
    if c[:1] == ["torch"] and len(c) >= 2 and c[1] not in ("nn", "mps"):
        return True
    last = c[-1] if c else (v.func.attr if isinstance(v.func, ast.Attribute) else "")
    return last in TENSOR_METHODS


def _is_loader_call(v: ast.AST | None) -> bool:
    return isinstance(v, ast.Call) and (_chain(v.func)[-1:] or [""])[0] in ("load_inline", "load", "compile_shader")


class _Visitor(ast.NodeVisitor):
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.hits: list[dict] = []
        self.ext_names: set[str] = set()  # names bound to a compiled extension (calls on them are the kernel)
        self.ext_attrs: set[str] = set()  # self.<attr> bound to a compiled extension
        self.layer_attrs: set[str] = set()  # self.<attr> bound to an nn compute layer in __init__
        self.buffer_attrs: set[str] = set()  # self.<attr> bound to a tensor (torch factory, Parameter, buffer) in __init__
        self.module_containers: set[str] = set()  # module-level mutable containers (caches if mutated)
        self.scope: list[str] = []  # "module", "class:<name>", "func:<name>"

    def hit(self, rule: str, node: ast.AST, detail: str) -> None:
        self.hits.append({"rule": rule, "detail": f"line {getattr(node, 'lineno', '?')}: {detail}"})

    # --- scope bookkeeping -------------------------------------------------------------
    def in_modelnew_init(self) -> bool:
        return "class:ModelNew" in self.scope and "func:__init__" in self.scope

    def in_function(self) -> bool:
        return any(s.startswith("func:") for s in self.scope)

    def in_modelnew_method(self) -> bool:
        return "class:ModelNew" in self.scope and self.in_function()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for b in node.bases:
            if "Model" in _chain(b):
                self.hit("reference_model", node, f"class {node.name} inherits from the reference Model")
        if not self.in_function():
            for stmt in node.body:
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and _is_mutable_value(stmt.value):
                    self.hit("global_cache", stmt, f"class-level mutable container in {node.name}")
        self.scope.append(f"class:{node.name}")
        self.generic_visit(node)
        self.scope.pop()

    def _visit_func(self, node: ast.AST, name: str) -> None:
        for d in getattr(node, "decorator_list", []):
            last = (_chain(d.func if isinstance(d, ast.Call) else d)[-1:] or [""])[0]
            if last in ("lru_cache", "cache", "cached_property", "memoize"):
                self.hit("global_cache", node, f"@{last} on {name}")
        self.scope.append(f"func:{name}")
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_func(node, "<lambda>")

    # --- statements ---------------------------------------------------------------------
    def visit_Try(self, node: ast.AST) -> None:
        self.hit("try_except", node, "try/except can silently fall back to the reference")
        self.generic_visit(node)

    visit_TryStar = visit_Try  # type: ignore[assignment]

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            c = _chain(item.context_expr.func if isinstance(item.context_expr, ast.Call) else item.context_expr)
            if c and c[-1] == "suppress":
                self.hit("try_except", node, "contextlib.suppress swallows kernel errors")
            if c and c[-1] in ("stream", "Stream", "StreamContext"):
                self.hit("cuda_stream", node, f"`with {'.'.join(c)}` runs work on a side stream")
        self.generic_visit(node)

    def visit_Global(self, node: ast.AST) -> None:
        self.hit("global_cache", node, "global/nonlocal state survives across calls")

    visit_Nonlocal = visit_Global  # type: ignore[assignment]

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self._check_import(node, a.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._check_import(node, node.module or "")
        for a in node.names:
            if a.name in INTROSPECTION_NAMES | INTROSPECTION_ATTRS:
                self.hit("introspection", node, f"imports {a.name}")
            if a.name == "Model":
                self.hit("reference_model", node, "imports the reference Model")

    def _check_import(self, node: ast.AST, name: str) -> None:
        root = name.split(".")[0]
        if root in FORBIDDEN_IMPORTS:
            self.hit("forbidden_import", node, f"imports {name}")
        if root in INTROSPECTION_MODULES or name == "sys._getframe":
            self.hit("introspection", node, f"imports {name}")

    def _assign_targets(self, node: ast.AST, targets: list[ast.AST], value: ast.AST | None) -> None:
        for t in targets:
            for sub in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                self._check_target(node, sub, value)

    def _check_target(self, node: ast.AST, t: ast.AST, value: ast.AST | None) -> None:
        chain = _chain(t.value if isinstance(t, ast.Subscript) else t)
        if isinstance(t, ast.Subscript) and chain[:1] and chain[0] in self.module_containers and self.in_function():
            self.hit("global_cache", node, f"writes into module-level `{chain[0]}` at call time")
        if len(chain) >= 2 and chain[0] in PATCH_ROOTS and chain[:2] != ["os", "environ"]:
            rule = "timer_patch" if any(w in ".".join(chain) for w in TIMER_WORDS) else "monkey_patch"
            self.hit(rule, node, f"assigns to {'.'.join(chain)}")
        if not self.in_function() and isinstance(t, ast.Name) and _is_mutable_value(value) \
                and not any(s.startswith("class:") for s in self.scope):
            self.module_containers.add(t.id)
            if _is_empty_container(value):
                self.hit("global_cache", node, f"module-level empty container `{t.id}` (a cache)")
        if isinstance(t, ast.Name) and _is_loader_call(value):
            self.ext_names.add(t.id)
        if len(chain) == 2 and chain[0] == "self":
            if self.in_modelnew_init():
                if _is_loader_call(value) or (isinstance(value, ast.Name) and value.id in self.ext_names):
                    self.ext_attrs.add(chain[1])
                vc = _chain(value.func) if isinstance(value, ast.Call) else []
                if len(vc) >= 2 and vc[-2] == "nn" and vc[-1] not in NN_CONTAINERS:
                    self.layer_attrs.add(chain[1])
                if _is_tensor_call(value):
                    self.buffer_attrs.add(chain[1])
            elif self.in_modelnew_method():
                self.hit("instance_cache", node, f"ModelNew writes self.{chain[1]} outside __init__")

    def visit_Assign(self, node: ast.Assign) -> None:
        self._assign_targets(node, node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._assign_targets(node, [node.target], node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._assign_targets(node, [node.target], None)
        self.generic_visit(node)

    # --- expressions --------------------------------------------------------------------
    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult) and self.in_function() and not self.in_modelnew_init() \
                and self.device != "mps":
            self.hit("reference_op", node, "`@` is torch.matmul")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "Model" and isinstance(node.ctx, ast.Load):
            self.hit("reference_model", node, "references the reference Model class")
        if node.id in INTROSPECTION_NAMES and isinstance(node.ctx, ast.Load):
            self.hit("introspection", node, f"uses {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in INTROSPECTION_ATTRS:
            self.hit("introspection", node, f"uses .{node.attr}")
        chain = _chain(node)
        if chain[:1] == ["torch"] and len(chain) >= 3 and chain[1] == "cuda" and chain[2] in ("Stream", "stream",
                                                                                               "set_stream",
                                                                                               "ExternalStream"):
            self.hit("cuda_stream", node, f"uses {'.'.join(chain)}")
        if node.attr in ("wait_stream", "record_stream", "Stream", "ExternalStream") and chain[:1] != ["torch"]:
            self.hit("cuda_stream", node, f"uses .{node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        chain = _chain(node.func)
        if chain == ["self", "register_buffer"] and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            self.buffer_attrs.add(node.args[0].value)
        if "functional" in chain or (len(chain) == 2 and chain[0] == "F"):
            self.hit("functional_call", node, f"calls {'.'.join(chain)}")
        elif chain[-1:] == ["setattr"] or chain[-1:] == ["delattr"]:
            target = _chain(node.args[0]) if node.args else []
            if target[:1] and target[0] in PATCH_ROOTS:
                name = ".".join(target)
                rule = "timer_patch" if any(w in name for w in TIMER_WORDS) else "monkey_patch"
                self.hit(rule, node, f"{chain[-1]} on {name}")
        elif self.in_function() and not self.in_modelnew_init():
            self._check_compute_call(node, chain)
        self.generic_visit(node)

    def _check_compute_call(self, node: ast.Call, chain: list[str]) -> None:
        if len(chain) == 2 and chain[0] in self.module_containers and chain[1] in MUTATORS:
            self.hit("global_cache", node, f"mutates module-level `{chain[0]}` at call time")
            return
        if self.device == "mps":  # PyTorch ops are the allowed tool on MPS; only work-hiding torch paths are banned
            if chain[:1] == ["torch"] and len(chain) >= 2 and chain[1] in MPS_BANNED_SUBMODULES:
                self.hit("reference_op", node, f"calls {'.'.join(chain)}")
            self._check_mps_state(node, chain)
            return
        if chain[:1] == ["torch"] and len(chain) >= 2:
            if chain[1] in TORCH_SUBMODULES or chain[-1] in COMPUTE_OPS:
                self.hit("reference_op", node, f"calls {'.'.join(chain)}")
            elif chain[1] == "nn" and len(chain) >= 3 and chain[2] not in NN_CONTAINERS:
                self.hit("reference_op", node, f"builds {'.'.join(chain)} in forward")
            return
        if len(chain) >= 2 and chain[0] == "nn" and chain[1] not in NN_CONTAINERS:
            self.hit("reference_op", node, f"builds {'.'.join(chain)} in forward")
            return
        if len(chain) == 2 and chain[0] == "self" and chain[1] in self.layer_attrs:
            self.hit("reference_op", node, f"runs nn layer self.{chain[1]}")
            return
        if isinstance(node.func, ast.Attribute):
            recv = node.func.value
            recv_chain = _chain(recv)
            if recv_chain[:1] == ["self"] and len(recv_chain) == 2:
                if recv_chain[1] in self.ext_attrs:
                    return
                if recv_chain[1] in self.layer_attrs and node.func.attr in ("forward", "_conv_forward", "__call__"):
                    self.hit("reference_op", node, f"runs nn layer self.{recv_chain[1]}.{node.func.attr}")
                    return
            if len(recv_chain) == 1 and recv_chain[0] in self.ext_names:
                return
            if node.func.attr in COMPUTE_OPS:
                self.hit("reference_op", node, f"calls Tensor.{node.func.attr}")


    def _check_mps_state(self, node: ast.Call, chain: list[str]) -> None:
        """With torch ops allowed, a content-keyed cache needs persistent state: block writes to self in forward."""
        if not self.in_modelnew_method() or self.in_modelnew_init():
            return
        if len(chain) >= 3 and chain[0] == "self" and chain[1] not in self.ext_attrs \
                and (chain[-1] in MUTATORS or (chain[-1].endswith("_") and not chain[-1].startswith("_"))):
            self.hit("instance_cache", node, f"mutates {'.'.join(chain[:-1])} in place via .{chain[-1]}()")
        for kw in node.keywords:
            if kw.arg == "out" and _chain(kw.value)[:1] == ["self"]:
                self.hit("instance_cache", node, f"writes out= into {'.'.join(_chain(kw.value))}")
        is_kernel = (len(chain) == 2 and chain[0] in self.ext_names) or \
                    (len(chain) == 3 and chain[0] == "self" and chain[1] in self.ext_attrs)
        if is_kernel:
            for a in list(node.args) + [k.value for k in node.keywords]:
                ac = _chain(a)
                if len(ac) == 2 and ac[0] == "self" and ac[1] in self.buffer_attrs:
                    self.hit("instance_cache", node, f"hands ModelNew state self.{ac[1]} to a kernel that can write it")


def _string_constants(tree: ast.AST) -> str:
    return "\n".join(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str))


def check(src: str, precision: str = "fp32", device: str = "cuda") -> list[dict]:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [{"rule": "syntax_error", "detail": f"line {exc.lineno}: {exc.msg}"}]
    if not any(isinstance(n, ast.ClassDef) and n.name == "ModelNew" for n in ast.walk(tree)):
        return [{"rule": "no_modelnew", "detail": "define `class ModelNew(nn.Module)`"}]
    v = _Visitor(device)
    # Two passes: the first learns which names are compiled extensions and nn layers.
    v.visit(tree)
    ext_names, ext_attrs, layer_attrs, containers = v.ext_names, v.ext_attrs, v.layer_attrs, v.module_containers
    buffer_attrs = v.buffer_attrs
    v = _Visitor(device)
    v.ext_names, v.ext_attrs, v.layer_attrs, v.buffer_attrs = ext_names, ext_attrs, layer_attrs, buffer_attrs
    v.module_containers = containers
    v.visit(tree)
    hits = v.hits
    strings = _string_constants(tree)
    for rule, rx, what in (("cuda_stream", CPP_STREAM, "side stream in the CUDA source"),
                           ("aten_op_in_cpp", CPP_ATEN, "ATen compute op in the C++ source"),
                           ("aten_op_in_cpp", CPP_TENSOR_METHOD, "Tensor compute method in the C++ source"),
                           ("vendor_library", CPP_VENDOR, "vendor math library")):
        m = rx.search(strings)
        if m:
            hits.append({"rule": rule, "detail": f"{what}: {m.group(0)!r}"})
    if precision == "fp32":
        m = LOW_PRECISION.search(src)
        if m:
            hits.append({"rule": "low_precision", "detail": f"fp32 problem, found {m.group(0)!r}"})
        m = METAL_LOW_PRECISION.search(strings) if device == "mps" else None
        if m:
            hits.append({"rule": "low_precision", "detail": f"fp32 problem, Metal source uses {m.group(0)!r}"})
    kb_checks = list(kb.STRICT_CHECKS) + ["cuda_impl", "stream_injection", "torch_computation_ops", "precision_downgrade"]
    for name in kb_checks:
        fn = kb.CHECK_FUNCTIONS[name]
        bad, msg = fn(src, precision) if name in kb.PRECISION_DEPENDENT_CHECKS else fn(src)
        if bad:
            hits.append({"rule": f"kb_{name}", "detail": msg})
    if device == "mps":
        hits = [h for h in hits if h["rule"] not in MPS_OFF_RULES]
    seen, out = set(), []
    for h in hits:  # one entry per (rule, detail), first occurrence order
        key = (h["rule"], h["detail"])
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def rules_summary(device: str = "cuda") -> str:
    """The rule list as the kernel role sees it in its prompt."""
    if device == "mps":
        return MPS_RULES
    doc = __doc__ or ""
    doc = doc[:doc.index("device=\"mps\"")] if "device=\"mps\"" in doc else doc
    return doc[doc.index("Rules (all reject):"):].strip()


if __name__ == "__main__":
    import json
    print(json.dumps(check(Path(sys.argv[1]).read_text()), indent=1))
