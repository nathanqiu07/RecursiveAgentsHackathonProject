"""Deterministic train/dev/held-out split, stratified by family and size, with a frozen held-out set.

Sorting by (family, n) and dealing each consecutive block of 5 as 2 train / 1 dev / 2 held-out
(order shuffled by the seed) keeps every family and size range represented in all three splits.
Held-out is frozen: once splits.json exists, a different held-out list is refused, so the
before/after score is always measured on the same unseen problems.
Run: .venv/bin/python -m benchmarks.qaplib.splits
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPLITS_JSON = HERE / "splits.json"
SPLITS_MD = HERE / "SPLITS.md"
SEED = 20261017
PATTERN = ["train", "train", "dev", "heldout", "heldout"]


class FrozenSplitError(RuntimeError):
    pass


def make_splits(entries: list[dict[str, Any]], seed: int = SEED) -> dict[str, list[str]]:
    """Pure function of the index entries and the seed."""
    rng = random.Random(seed)
    ordered = sorted(entries, key=lambda e: (e["family"], e["n"], e["name"]))
    out: dict[str, list[str]] = {"train": [], "dev": [], "heldout": []}
    for start in range(0, len(ordered), len(PATTERN)):
        block = ordered[start:start + len(PATTERN)]
        labels = PATTERN[:]
        rng.shuffle(labels)
        for entry, label in zip(block, labels):
            out[label].append(entry["name"])
    return {k: sorted(v) for k, v in out.items()}


def heldout_hash(entries_by_name: dict[str, dict[str, Any]], names: list[str]) -> str:
    """Hash of names and file hashes, so a changed instance file also counts as a changed held-out set."""
    blob = "\n".join(f"{n}:{entries_by_name[n]['sha256']}" for n in sorted(names))
    return hashlib.sha256(blob.encode()).hexdigest()


def freeze(splits: dict[str, list[str]], entries: list[dict[str, Any]], path: Path = SPLITS_JSON) -> bool:
    """Write splits once; return False if an identical file exists; raise if held-out would change."""
    by_name = {e["name"]: e for e in entries}
    record = {"seed": SEED, **splits, "heldout_sha256": heldout_hash(by_name, splits["heldout"])}
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("heldout") != splits["heldout"] or old.get("heldout_sha256") != record["heldout_sha256"]:
            raise FrozenSplitError(f"{path} exists with a different held-out set; refusing to overwrite it")
        if old == record:
            return False
        raise FrozenSplitError(f"{path} exists with different train/dev lists; delete it deliberately to re-split")
    path.write_text(json.dumps(record, indent=1) + "\n")
    return True


def load_splits(path: Path = SPLITS_JSON) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_markdown(splits: dict[str, list[str]], entries: list[dict[str, Any]], path: Path = SPLITS_MD) -> None:
    by_name = {e["name"]: e for e in entries}
    lines = ["# QAPLIB splits", "",
             f"Seed {SEED}. Stratified by family and size (blocks of 5 sorted by family, n: 2 train / 1 dev / 2 held-out).",
             "Held-out is frozen: `splits.py` refuses to overwrite a `splits.json` whose held-out set differs.", "",
             f"Held-out hash (sha256 over `name:file_sha256`): `{heldout_hash(by_name, splits['heldout'])}`", ""]
    for split in ("train", "dev", "heldout"):
        names = splits[split]
        sizes = [by_name[n]["n"] for n in names]
        lines += [f"## {split} ({len(names)} instances, n {min(sizes)}..{max(sizes)})", "",
                  "| name | family | n | BKS | proven optimal | sha256 (.dat) |", "|---|---|---|---|---|---|"]
        lines += [f"| {n} | {by_name[n]['family']} | {by_name[n]['n']} | {by_name[n]['bks']} | "
                  f"{'yes' if by_name[n]['bks_proven_optimal'] else 'no'} | `{by_name[n]['sha256'][:16]}` |" for n in names]
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    entries = json.loads((HERE / "instances" / "index.json").read_text())["instances"]
    splits = make_splits(entries)
    try:
        wrote = freeze(splits, entries)
    except FrozenSplitError as exc:
        print(f"ERROR: {exc}")
        return 1
    write_markdown(splits, entries)
    print(("wrote" if wrote else "unchanged") + f" {SPLITS_JSON.name}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
