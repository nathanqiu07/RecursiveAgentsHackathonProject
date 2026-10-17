"""Score a finished QAPLIB run's frozen version + learned brain on the held-out instances its final report skipped.

The dashboard ranks a config only when it has a verified permutation for all 18 held-out instances
(benchmarks/qaplib/splits.json); the live runs sample 6 of them (n <= 50) to fit the time budget. This runs the same
team (same version file, same brain, read-only) on the rest and saves each solution where team.evolve saves the
run's own, under the same config label, so the publisher re-verifies all 18 from scratch.

  CRUSOE_API_KEY=... .venv/bin/python reports/qaplib_live/fill_heldout.py runs/qaplib-evolve-3 0 "v0 + final brain"
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.qaplib.load import load_tasks  # noqa: E402
from benchmarks.qaplib.team_runner import run_team  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.crusoe import CrusoeLLM  # noqa: E402
from team.evolve import TeamVersion  # noqa: E402


def main(run_dir: str, version_number: str, label: str, workers: int = 3) -> int:
    run = Path(run_dir).resolve()
    version = TeamVersion.load(run / "versions" / f"v{version_number}.json")
    # The cold arm reads an empty brain, as Evolver.final_report's does; every other arm reads the run's learned one.
    cold = "empty brain" in label
    brain = SharedBrain(tempfile.mkdtemp()) if cold else SharedBrain(run / "brain")
    out = run / "solutions" / (re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "run")
    out.mkdir(parents=True, exist_ok=True)
    have = {json.loads(f.read_text())["solution"]["instance"] for f in out.glob("*.json")}
    tasks = [t for t in load_tasks("heldout") if t.payload["name"] not in have]
    print(f"{run.name} {label}: {len(have)} held-out solutions saved, running {len(tasks)}: "
          + ", ".join(t.payload["name"] for t in tasks), flush=True)
    llm = CrusoeLLM()

    def one(task):
        try:
            r = run_team(version, brain, task, llm)
        except Exception as e:  # noqa: BLE001 - one failed call costs one instance, reported below
            print(f"  {task.payload['name']}: run_team failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            return task, None
        if isinstance(r.solution, dict) and "solution" in r.solution:
            record = {**r.solution, "task_id": r.task_id, "config": label, "version": version.version,
                      "split": task.split, "score": r.score, "passed": r.passed, "filled_after_run": True}
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", r.task_id)
            (out / f"{name}.json").write_text(json.dumps(record, default=str))
        print(f"  {task.payload['name']} (n={task.payload['n']}): score {r.score:.4f} cost ${r.cost_usd:.5f} "
              f"failures {r.failures}", flush=True)
        return task, r

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, tasks))
    usage = llm.ledger.summary()
    (run / "reports").mkdir(exist_ok=True)
    (run / "reports" / f"heldout_fill_{out.name}.json").write_text(json.dumps(
        {"label": label, "version": version.version, "filled": [t.payload["name"] for t, r in results if r],
         "failed": [t.payload["name"] for t, r in results if r is None], "usage": usage}, indent=1))
    saved = [json.loads(f.read_text()) for f in out.glob("*.json")]
    scores = [d["score"] for d in saved]
    print(f"{len(saved)}/18 held-out solutions saved; mean score over them {sum(scores) / max(1, len(scores)):.4f}; "
          f"fill cost ${usage['total']['cost_usd']:.5f}; wall {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3))
