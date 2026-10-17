"""Offline demo: 3 generations of the toy team with a scripted reviewer, then the held-out ablation.

Usage: python3 scripts/demo_offline.py [output_dir]   (default: a fresh temp dir)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team.bench import REVIEWER_SCRIPT, make_tasks, toy_run_team, toy_v0  # noqa: E402
from team.brain import SharedBrain  # noqa: E402
from team.evolve import Evolver, markdown_table  # noqa: E402
from team.fake_llm import ScriptedLLM  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="demo_offline_"))
    train, dev, heldout = make_tasks(seed=0)
    llm = ScriptedLLM({"reviewer": REVIEWER_SCRIPT})
    evolver = Evolver(toy_run_team, llm, "reviewer-model", SharedBrain(out / "brain"), out / "versions",
                      initial=toy_v0(), reports_dir=out / "reports")
    final = evolver.run(3, train, dev, heldout)
    for h in evolver.history:
        print(f"gen {h['generation']}: v{h['parent']} -> v{h['version']} accepted={h['accepted']} "
              f"train={h['train_mean']:.3f} dev={h['dev_mean']:.3f} brain_active={h['brain']['active']}")
    print("\nchangelog:\n  " + "\n  ".join(final.changelog))
    rows = evolver.final_report(heldout)
    print(f"\nHeld-out ({len(heldout)} unseen tasks):\n")
    print(markdown_table(rows))
    print(f"artifacts: {out}")


if __name__ == "__main__":
    main()
