"""The one seam between the domain-agnostic core and a real domain (DoneTwin data-hall layout).

The Evolver calls `run_team(version, brain, task, llm) -> RunResult` for every run. A DoneTwin
implementation must:

1. For each enabled agent in `version.roles()` (Spec, Layout, Inspector), call
   `context, ids = brain.read(role, task.text)` and prepend `context` to that agent's system
   prompt, after `version.prompts[role]` and the steps in `version.playbooks[role]`. Collect every
   returned id into `brain_ids_read`; credit assignment depends on it. Use
   `brain.stable_prefix(role)` instead if provider prefix caching matters more than relevance.
2. Take everything else from the version: models per role from `version.config["models"]`,
   temperatures, `max_iters`, and any other knob the Reviewer may tune via `config_set`.
3. Score with the independent geometry verifier, never an LLM: `score` in 0..1, `passed` iff
   every hard check holds. LLM self-assessment may only fill `claimed_done`.
4. Return a RunResult with `failures` = one line per failing check, formatted "Role: message"
   so it becomes a pitfall for the role that owns the check (the message should state the rule
   and the measured value, e.g. "Layout: hot aisle 1.0 m < required 1.2 m"), plus iterations,
   calls, tokens and cost summed over all agents.
5. Optionally put agent-written memory proposals in `notes` as JSON
   {"proposals": [{"kind": ..., "roles": [...], "text": "<=280 chars"}]}; the Evolver attaches
   this run's id and verifier score as evidence and the Curator decides what lands.

Never write to the brain from inside `run_team`; the Evolver credits and proposes after the run,
and evaluation runs (dev gate, held-out report) must stay read-only.
"""
from __future__ import annotations

from typing import Callable

from team.brain import SharedBrain
from team.evolve import TeamVersion
from team.types import LLM, RunResult, Task

RunTeam = Callable[[TeamVersion, SharedBrain, Task, LLM], RunResult]


def donetwin_run_team(version: TeamVersion, brain: SharedBrain, task: Task, llm: LLM) -> RunResult:
    """Run the DoneTwin Spec -> Layout -> Inspector loop on one task, scored by the geometry verifier."""
    raise NotImplementedError("DoneTwin integration lands with its code; see this module's docstring")
