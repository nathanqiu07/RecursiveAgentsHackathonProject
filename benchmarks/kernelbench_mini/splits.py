"""The 12 problems, their short retrieval texts, and a seeded, disjoint train 4 / dev 3 / held-out 5 split.

Held-out is frozen like QAPLIB's: once splits.json exists, a different held-out list is refused.
Run: .venv/bin/python -m benchmarks.kernelbench_mini.splits
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from team.types import Task  # noqa: E402

HERE = Path(__file__).resolve().parent
PROBLEMS_DIR = HERE / "problems"
SPLITS_JSON = HERE / "splits.json"
SEED = 20261017
SPLIT_SIZES = (4, 3, 5)

# id -> (file, category, retrieval text). Shapes are the reduced ones in problems/.
PROBLEMS: dict[str, tuple[str, str, str]] = {
    "kb_matmul_square": ("1_Square_matrix_multiplication_.py", "matmul",
                         "fp32 square matmul C=A@B, 256x256x256; shared-memory tiling, register blocking"),
    "kb_matmul_tb": ("17_Matmul_with_transposed_B.py", "matmul",
                     "fp32 matmul with transposed B, C=A@B.T, M=256 K=256 N=384; tiled GEMM, B row-major reads"),
    "kb_sum_dim": ("47_Sum_reduction_over_a_dimension.py", "reduction",
                   "sum reduction over dim 1 of a 16x256x255 tensor, keepdim; strided column reduction"),
    "kb_softmax": ("23_Softmax.py", "softmax",
                   "row softmax over dim 1, 64x8192 fp32; max-subtract, warp/block reductions, online softmax"),
    "kb_rmsnorm": ("36_RMSNorm_.py", "norm",
                   "RMSNorm over the feature dim 1 of 8x64x64x64, eps 1e-5; per-position reduction across channels"),
    "kb_relu": ("19_ReLU.py", "elementwise",
                "elementwise ReLU 64x16384 fp32; memory-bound, vectorized float4; inputs are signed"),
    "kb_gelu": ("26_GELU_.py", "elementwise",
                "elementwise exact GELU (erf) 64x16384 fp32; memory-bound, erff, vectorized loads"),
    "kb_conv2d": ("63_conv_standard_2D__square_input__square_kernel.py", "conv",
                  "conv2d 3x3, 8->32 channels, 4x8x64x64 input, no bias, stride 1, no padding; direct conv, weight reuse"),
    "kb_cumsum": ("89_cumsum.py", "scan",
                  "inclusive cumsum along dim 1 of 128x512 fp32; block scan, warp shuffles"),
    "kb_argmax": ("51_Argmax_over_a_dimension.py", "reduction",
                  "argmax over dim 1 of 16x256x255 fp32, int64 output; first index on ties"),
    "kb_mse": ("94_MSELoss.py", "loss",
               "MSE loss mean((p-t)^2) over 128x4096 fp32 to a scalar; grid-wide reduction"),
    "kb_scale_copy": ("5_Matrix_scalar_multiplication.py", "copy",
                      "matrix times python scalar, 1024x2048 fp32 out-of-place; pure memory copy bandwidth"),
}


class FrozenSplitError(RuntimeError):
    pass


def make_splits(seed: int = SEED, sizes: tuple[int, int, int] = SPLIT_SIZES) -> dict[str, list[str]]:
    """Pure function of the seed: shuffle the ids, deal train/dev/held-out in order."""
    if sum(sizes) > len(PROBLEMS):
        raise ValueError(f"split sizes {sizes} exceed {len(PROBLEMS)} problems")
    ids = sorted(PROBLEMS)
    random.Random(seed).shuffle(ids)
    a, b, c = sizes
    return {"train": sorted(ids[:a]), "dev": sorted(ids[a:a + b]), "heldout": sorted(ids[a + b:a + b + c])}


def heldout_hash(names: list[str]) -> str:
    blob = "\n".join(f"{n}:{hashlib.sha256((PROBLEMS_DIR / PROBLEMS[n][0]).read_bytes()).hexdigest()}"
                     for n in sorted(names))
    return hashlib.sha256(blob.encode()).hexdigest()


def freeze(splits: dict[str, list[str]], path: Path = SPLITS_JSON) -> bool:
    record = {"seed": SEED, **splits, "heldout_sha256": heldout_hash(splits["heldout"])}
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("heldout") != splits["heldout"] or old.get("heldout_sha256") != record["heldout_sha256"]:
            raise FrozenSplitError(f"{path} exists with a different held-out set; refusing to overwrite it")
        return False
    path.write_text(json.dumps(record, indent=1) + "\n")
    return True


def make_task(pid: str, split: str) -> Task:
    file, category, text = PROBLEMS[pid]
    ref_src = (PROBLEMS_DIR / file).read_text()
    return Task(id=pid, split=split, text=text,
                payload={"problem": Path(file).stem, "file": file, "category": category, "ref_src": ref_src})


def load_tasks(seed: int = SEED, sizes: tuple[int, int, int] = SPLIT_SIZES) -> tuple[list[Task], list[Task], list[Task]]:
    s = make_splits(seed, sizes)
    return tuple([make_task(pid, split) for pid in s[split]] for split in ("train", "dev", "heldout"))  # type: ignore[return-value]


def main() -> int:
    s = make_splits()
    try:
        wrote = freeze(s)
    except FrozenSplitError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(("wrote" if wrote else "unchanged") + f" {SPLITS_JSON.name}: " + json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
