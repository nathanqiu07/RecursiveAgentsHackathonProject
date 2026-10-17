"""Exact, independent verifier for QAPLIB answers.

The score must never come from an LLM, so this module only parses a permutation and
recomputes the objective with Python integers (no float or int64 overflow risk).
cost(p) = sum_{i,j} A[i][j] * B[p[i]][p[j]], with p[i] the 0-based location of facility i.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
INSTANCES = HERE / "instances"
_INT = re.compile(r"-?\d+")


def parse_dat(path: str | Path) -> tuple[int, list[list[int]], list[list[int]]]:
    """QAPLIB .dat is n, then A (n*n), then B (n*n); line breaks are not meaningful, so tokenize."""
    nums = [int(t) for t in _INT.findall(Path(path).read_text())]
    n = nums[0]
    if len(nums) != 1 + 2 * n * n:
        raise ValueError(f"{path}: expected {1 + 2 * n * n} integers for n={n}, found {len(nums)}")
    flat_a, flat_b = nums[1:1 + n * n], nums[1 + n * n:]
    return n, [flat_a[i * n:(i + 1) * n] for i in range(n)], [flat_b[i * n:(i + 1) * n] for i in range(n)]


def parse_sln(path: str | Path) -> tuple[int, int, list[int]]:
    """QAPLIB .sln is n, the objective value, then a 1-based permutation (formatting varies)."""
    nums = [int(t) for t in _INT.findall(Path(path).read_text())]
    n, value = nums[0], nums[1]
    return n, value, [x - 1 for x in nums[2:2 + n]]


def cost(a: list[list[int]], b: list[list[int]], perm: list[int]) -> int:
    """Exact objective; perm is 0-based."""
    total = 0
    for i, row in enumerate(a):
        b_row = b[perm[i]]
        total += sum(a_ij * b_row[perm[j]] for j, a_ij in enumerate(row) if a_ij)
    return total


def parse_permutation(text: str, n: int) -> list[int]:
    """Accept 0- or 1-based answers; the base is unambiguous because a valid one is exactly 0..n-1 or 1..n.

    Reads the last `PERMUTATION:` line if present, else every integer in the text.
    """
    lines = [ln for ln in text.splitlines() if "PERMUTATION:" in ln.upper()]
    source = lines[-1].upper().split("PERMUTATION:", 1)[1] if lines else text
    values = [int(t) for t in _INT.findall(source)]
    if len(values) != n:
        raise ValueError(f"expected {n} integers, got {len(values)}")
    if sorted(values) == list(range(n)):
        return values
    if sorted(values) == list(range(1, n + 1)):
        return [v - 1 for v in values]
    raise ValueError("not a permutation of 0..n-1 or 1..n (duplicates or out-of-range values)")


def load_index() -> dict[str, dict[str, Any]]:
    data = json.loads((INSTANCES / "index.json").read_text())
    return {e["name"]: e for e in data["instances"]}


def verify(name: str, answer: str | list[int], bks: int | None = None, dat_path: str | Path | None = None) -> dict[str, Any]:
    """Score = bks / cost for minimization, so 1.0 is the best known and >1.0 is a new best."""
    if bks is None or dat_path is None:
        entry = load_index()[name]
        bks = entry["bks"] if bks is None else bks
        dat_path = dat_path or INSTANCES / f"{name}.dat"
    n, a, b = parse_dat(dat_path)
    try:
        text = answer if isinstance(answer, str) else " ".join(str(x) for x in answer)
        perm = parse_permutation(text, n)
    except ValueError as exc:
        return {"status": "invalid", "error": str(exc), "cost": None, "bks": bks, "score": 0.0, "gap_pct": None}
    value = cost(a, b, perm)
    score = bks / value if value > 0 else (1.0 if bks == 0 else 0.0)
    gap = (value - bks) / bks * 100.0 if bks else 0.0
    return {"status": "ok", "cost": value, "bks": bks, "score": score, "gap_pct": gap, "perm": perm}
