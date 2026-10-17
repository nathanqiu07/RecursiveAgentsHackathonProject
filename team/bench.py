"""Deterministic toy domain so the brain and evolver can be tested and demoed without a network.

Each task needs three rules. A role satisfies a rule only if the rule's code (e.g. "[F3]") is in
its context: the role prompt, its playbook, or what it read from the Shared Brain. Most failures
reveal the rule (so the brain can learn it); two never do (only a playbook edit fixes them); the
Inspector role does not exist in v0 (only a team-structure edit fixes its rules).
"""
from __future__ import annotations

import itertools
import json
import random

from team.brain import SharedBrain
from team.evolve import TeamVersion
from team.types import LLM, RunResult, Task

# code: (responsible role, topic word used in task text, rule, verifier reveals the rule on failure)
FACTS: dict[str, tuple[str, str, str, bool]] = {
    "F1": ("Layout", "aisle", "hot aisle width must be at least 1.2 m", True),
    "F2": ("Layout", "tiles", "rack rows must align to the 600 mm floor tile grid", True),
    "F3": ("Spec", "power", "rack power density is capped at 40 kW", True),
    "F4": ("Spec", "cooling", "liquid cooling is required above 30 kW per rack", True),
    "F5": ("Inspector", "egress", "every row needs an exit within 45 m of travel", True),
    "F6": ("Inspector", "fire", "a fire suppression zone must not span two halls", False),
    "F7": ("Layout", "cables", "cable trays must run perpendicular to rack rows", False),
    "F8": ("Spec", "redundancy", "power feeds must be 2N redundant", True),
}
MYTH = "[M0] aisles may shrink to 0.8 m to fit more racks"


def make_tasks(seed: int = 0) -> tuple[list[Task], list[Task], list[Task]]:
    """Disjoint train/dev/held-out splits; held-out tasks are new rule combinations."""
    combos = list(itertools.combinations(sorted(FACTS), 3))
    random.Random(seed).shuffle(combos)

    def build(split: str, chunk: list[tuple[str, ...]]) -> list[Task]:
        return [Task(id=f"{split}-{i}", split=split, payload={"required": list(c)},
                     text="Lay out a data hall covering " + ", ".join(FACTS[k][1] for k in c))
                for i, c in enumerate(chunk)]
    return build("train", combos[:12]), build("dev", combos[12:22]), build("heldout", combos[22:34])


def toy_v0() -> TeamVersion:
    return TeamVersion(
        version=0, parent=None,
        prompts={"Spec": "You turn the request into a data hall spec.", "Layout": "You place racks and aisles."},
        playbooks={"Layout": [f"[F1] {FACTS['F1'][2]}"]},
        config={"roles_enabled": ["Spec", "Layout"], "models": {}, "temperature": 0.2, "brain_budget": 3000},
        changelog=["v0: initial team"])


def toy_run_team(version: TeamVersion, brain: SharedBrain, task: Task, llm: LLM) -> RunResult:
    """Each enabled role reads the brain, calls the LLM once; an independent check scores the rules."""
    seen: dict[str, str] = {}
    ids: set[str] = set()
    prompt_tokens = completion_tokens = 0
    for role in version.roles():
        context, read_ids = brain.read(role, task.text, version.config.get("brain_budget", 3000))
        ids.update(read_ids)
        system = "\n".join([version.prompts.get(role, ""), *version.playbooks.get(role, []), context])
        reply = llm.chat(version.config.get("models", {}).get(role, "toy"),
                         [{"role": "system", "content": system}, {"role": "user", "content": task.text}],
                         temperature=version.config.get("temperature", 0.2), role=role)
        prompt_tokens += reply.prompt_tokens
        completion_tokens += reply.completion_tokens
        seen[role] = system
    failures, proposals = [], []
    for code in task.payload["required"]:
        role, _, rule, revealed = FACTS[code]
        if f"[{code}]" in seen.get(role, ""):
            continue
        if revealed:
            failures.append(f"{role}: [{code}] {rule}")
            proposals.append({"kind": "check_template", "roles": ["Inspector"], "text": f"verify [{code}] before sign-off"})
        else:
            failures.append(f"{role}: check {code} failed (verifier gave no detail)")
    required = len(task.payload["required"])
    score = (required - len(failures)) / required
    if any(MYTH[:4] in s for s in seen.values()):
        score = max(0.0, score - 0.34)
        failures.append(f"Layout: [F1] {FACTS['F1'][2]}")
    return RunResult(task_id=task.id, version=version.version, score=round(score, 4), passed=not failures,
                     claimed_done=True, iterations=1, calls=len(seen), prompt_tokens=prompt_tokens,
                     completion_tokens=completion_tokens, cost_usd=round((prompt_tokens + completion_tokens) * 2e-7, 8),
                     failures=failures, brain_ids_read=sorted(ids), notes=json.dumps({"proposals": proposals}))


def _mutations(*items: dict) -> str:
    return json.dumps({"mutations": list(items)})


# One scripted reviewer reply per generation: a real gain, a harmful edit, a no-op, and invalid ones.
REVIEWER_SCRIPT: list[str] = [
    _mutations({"type": "add_role", "role": "Inspector", "prompt": "You check the layout against every rule.",
                "rationale": "Inspector-owned checks fail on every run because no role owns them"},
               {"type": "playbook_add", "role": "Layout", "step": MYTH, "rationale": "denser halls"},
               {"type": "config_set", "key": "temperature", "value": 0.1, "rationale": "less variance"}),
    _mutations({"type": "playbook_add", "role": "Layout", "step": f"[F7] {FACTS['F7'][2]}",
                "rationale": "check F7 fails repeatedly without detail; vendor guide says trays run perpendicular"},
               {"type": "prompt_edit", "role": "Spec", "text": "You turn the request into a precise spec.",
                "rationale": "clarity"},
               {"type": "delete_everything", "rationale": "invalid on purpose"}),
    _mutations({"type": "playbook_add", "role": "Inspector", "step": f"[F6] {FACTS['F6'][2]}",
                "rationale": "check F6 fails repeatedly; fire code limits zones to one hall"},
               {"type": "playbook_remove", "role": "Layout", "index": 99, "rationale": "invalid index"}),
]
