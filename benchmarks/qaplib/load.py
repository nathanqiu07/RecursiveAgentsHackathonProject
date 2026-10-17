"""Turn a split into `team.types.Task`s; the text carries family and size so the brain retrieves by them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.qaplib.splits import load_splits  # noqa: E402
from benchmarks.qaplib.verify import INSTANCES  # noqa: E402
from team.types import Task  # noqa: E402


def load_tasks(split: str) -> list[Task]:
    if split not in ("train", "dev", "heldout"):
        raise ValueError(f"unknown split {split!r}; expected train, dev or heldout")
    index = {e["name"]: e for e in json.loads((INSTANCES / "index.json").read_text())["instances"]}
    tasks = []
    for name in load_splits()[split]:
        e = index[name]
        dat = INSTANCES / f"{name}.dat"
        if not dat.exists():
            raise FileNotFoundError(f"{dat} missing; run: .venv/bin/python -m benchmarks.qaplib.fetch")
        text = f"QAP facility layout, family {e['family']}, n={e['n']}"
        if e.get("comment"):
            text += f". {e['comment']}"
        tasks.append(Task(id=f"qaplib-{name}", split=split, text=text,
                          payload={"name": name, "path": str(dat), "n": e["n"], "bks": e["bks"]}))
    return tasks
