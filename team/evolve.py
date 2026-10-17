"""Version N+1: the team reviews its runs, proposes edits to itself, and keeps only measured gains.

Fast memory (the Shared Brain) learns every run; slow self-modification (TeamVersion) changes only
through a Gate on dev tasks. `promote_brain` links the two: proven brain entries become playbook
steps and leave the brain once the Gate accepts the version that absorbed them.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from team.brain import MAX_TEXT, SharedBrain
from team.types import LLM, RunResult, Task

RunTeam = Callable[["TeamVersion", SharedBrain, Task, LLM], RunResult]

MUTATIONS: dict[str, tuple[str, ...]] = {
    "prompt_edit": ("role", "text"),
    "playbook_add": ("role", "step"),
    "playbook_remove": ("role", "index"),
    "config_set": ("key", "value"),
    "add_role": ("role", "prompt"),
    "remove_role": ("role",),
    "promote_brain": ("entry_ids", "role"),
    "tool_set": ("tool", "enabled"),
}
OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {"tool_set": ("params",)}
# Owned by add_role/remove_role and tool_set, so config_set cannot bypass their checks or invent tools.
RESERVED_KEYS = ("roles_enabled", "tools", "tool_catalog")


@dataclass
class TeamVersion:
    version: int
    parent: int | None
    prompts: dict[str, str]
    playbooks: dict[str, list[str]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    changelog: list[str] = field(default_factory=list)

    def roles(self) -> list[str]:
        return list(self.config.get("roles_enabled") or sorted(self.prompts))

    def key(self) -> str:
        """Content hash, so identical candidates share cached dev results."""
        body = json.dumps([self.prompts, self.playbooks, self.config], sort_keys=True)
        return hashlib.sha1(body.encode()).hexdigest()[:12]

    def save(self, versions_dir: str | Path) -> Path:
        path = Path(versions_dir) / f"v{self.version}.json"
        path.write_text(json.dumps(asdict(self), indent=1, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TeamVersion":
        return cls(**json.loads(Path(path).read_text()))


def tool_catalog(v: TeamVersion, tools: list[str] | None = None) -> list[str]:
    """Tools a tool_set may name: an explicit allowlist, else the version's own `tool_catalog`."""
    return list(v.config.get("tool_catalog", []) if tools is None else tools)


def validate_mutation(m: Any, v: TeamVersion, brain: SharedBrain, tools: list[str] | None = None) -> str | None:
    if not isinstance(m, dict) or m.get("type") not in MUTATIONS:
        return f"unknown mutation type {m.get('type') if isinstance(m, dict) else m!r}"
    kind = m["type"]
    missing = [f for f in MUTATIONS[kind] if f not in m]
    if missing:
        return f"{kind} missing {missing}"
    role = m.get("role")
    if kind == "add_role":
        if not isinstance(role, str) or not role or role in v.prompts:
            return f"add_role needs a new role name, got {role!r}"
    elif "role" in MUTATIONS[kind] and role not in v.prompts:
        return f"unknown role {role!r}"
    for name in ("text", "step", "prompt"):
        if name in MUTATIONS[kind] and not (isinstance(m[name], str) and m[name].strip()):
            return f"{name} must be a non-empty string"
    if kind == "playbook_add" and len(m["step"]) > MAX_TEXT:
        return f"step longer than {MAX_TEXT} chars"
    if kind == "playbook_remove" and not (isinstance(m["index"], int) and 0 <= m["index"] < len(v.playbooks.get(role, []))):
        return f"playbook index {m['index']!r} out of range"
    if kind == "remove_role" and len(v.roles()) <= 1:
        return "cannot remove the last role"
    if kind == "config_set" and (not isinstance(m["key"], str) or m["key"] in RESERVED_KEYS):
        return "config key must be a string other than roles_enabled, tools or tool_catalog (use add_role/remove_role/tool_set)"
    if kind == "tool_set":
        catalog = tool_catalog(v, tools)
        if m["tool"] not in catalog:
            return f"unknown tool {m['tool']!r}, catalog is {catalog}"
        if not isinstance(m["enabled"], bool):
            return "enabled must be true or false"
        params = m.get("params") or {}
        if not isinstance(params, dict) or "enabled" in params:
            return "params must be an object without an enabled key"
    if kind == "promote_brain":
        ids = m["entry_ids"]
        live = {e.id for e in brain.get(ids if isinstance(ids, list) else []) if e.live}
        if not ids or not isinstance(ids, list) or set(ids) - live:
            return f"entry_ids must be live brain entries, got {ids!r}"
    return None


def apply_mutation(v: TeamVersion, m: dict, brain: SharedBrain, number: int) -> TeamVersion:
    c = TeamVersion(version=number, parent=v.version, prompts=dict(v.prompts),
                    playbooks={r: list(s) for r, s in v.playbooks.items()},
                    config=json.loads(json.dumps(v.config)), changelog=list(v.changelog))
    kind, role, roles = m["type"], m.get("role"), c.roles()
    if kind == "prompt_edit":
        c.prompts[role] = m["text"]
    elif kind == "playbook_add":
        c.playbooks.setdefault(role, []).append(m["step"])
    elif kind == "playbook_remove":
        c.playbooks[role].pop(m["index"])
    elif kind == "config_set":
        c.config[m["key"]] = m["value"]
    elif kind == "add_role":
        c.prompts[role] = m["prompt"]
        c.config["roles_enabled"] = roles + [role]
    elif kind == "remove_role":
        c.prompts.pop(role)
        c.playbooks.pop(role, None)
        c.config["roles_enabled"] = [r for r in roles if r != role]
    elif kind == "promote_brain":
        c.playbooks.setdefault(role, []).extend(e.text for e in brain.get(m["entry_ids"]))
    elif kind == "tool_set":
        c.config.setdefault("tools", {})[m["tool"]] = {"enabled": m["enabled"], **(m.get("params") or {})}
    detail = {k: m[k] for k in MUTATIONS[kind] + OPTIONAL_FIELDS.get(kind, ()) if k in m}
    c.changelog.append(f"v{number}: {kind} {json.dumps(detail)} - {m.get('rationale', '')}".strip())
    return c


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(match.group(0)) if match else {}
        except ValueError:
            return {}


class Reviewer:
    """Turns recent runs into at most `max_mutations` validated self-edits."""

    SYSTEM = ("You review an agent team's recent runs and propose at most 3 changes to its own prompts, "
              "playbooks, tools, config or roles. Each change is tested on dev tasks and kept only if the score rises, "
              "so propose concrete, testable edits that target the most frequent failures. "
              'Reply with JSON only: {"mutations": [{"type": ..., "rationale": ..., <fields>}]}. Types and fields: '
              + json.dumps(MUTATIONS) + "; optional fields: " + json.dumps(OPTIONAL_FIELDS)
              + ". tool_set may only name a tool listed in tool_catalog; params are passed to that tool.")

    def __init__(self, llm: LLM, model: str, max_mutations: int = 3, tool_catalog: list[str] | None = None):
        self.llm, self.model, self.max_mutations = llm, model, max_mutations
        self.tool_catalog = tool_catalog
        self.dropped: list[dict] = []

    def build_prompt(self, version: TeamVersion, results: list[RunResult], brain: SharedBrain) -> str:
        failures = Counter(f for r in results for f in r.failures)
        summary = {
            "version": {k: v for k, v in asdict(version).items() if k != "changelog"},
            "recent_changes": version.changelog[-5:],
            "runs": {"n": len(results), "mean_score": round(mean(r.score for r in results), 3) if results else 0,
                     "pass_rate": round(mean(r.passed for r in results), 3) if results else 0,
                     "cost_usd": round(sum(r.cost_usd for r in results), 4)},
            "tool_catalog": tool_catalog(version, self.tool_catalog),
            "tool_settings": version.config.get("tools", {}),
            "failures_by_count": failures.most_common(15),
            "brain_stats": brain.stats(),
            "brain_top_helping": [{"id": e.id, "roles": e.roles, "text": e.text, "helps": e.helps, "hurts": e.hurts}
                                  for e in brain.top(5)],
        }
        return json.dumps(summary, indent=1)

    def propose(self, version: TeamVersion, results: list[RunResult], brain: SharedBrain) -> list[dict]:
        reply = self.llm.chat(self.model, [{"role": "system", "content": self.SYSTEM},
                                           {"role": "user", "content": self.build_prompt(version, results, brain)}],
                              temperature=0.3, json_mode=True, role="reviewer")
        raw = _parse_json(reply.text).get("mutations", [])
        valid, self.dropped = [], []
        for m in raw if isinstance(raw, list) else []:
            error = validate_mutation(m, version, brain, self.tool_catalog)
            if error is None and len(valid) >= self.max_mutations:
                error = f"over the limit of {self.max_mutations} mutations"
            (self.dropped.append({"mutation": m, "reason": error}) if error else valid.append(m))
        return valid


class Gate:
    """Paired dev comparison: keep a candidate only for a real gain with no lost passes."""

    def __init__(self, min_gain: float = 0.05):
        self.min_gain = min_gain

    def decide(self, parent: list[RunResult], candidate: list[RunResult], min_gain: float | None = None) -> dict:
        """`min_gain` overrides the default, e.g. 0.0 for consolidation, whose payoff is a leaner brain."""
        threshold = self.min_gain if min_gain is None else min_gain
        p, c = {r.task_id: r for r in parent}, {r.task_id: r for r in candidate}
        ids = sorted(p.keys() & c.keys())
        if not ids:
            return {"n": 0, "accepted": False, "reason": "no paired dev tasks"}
        pm, cm = mean(p[i].score for i in ids), mean(c[i].score for i in ids)
        pp, cp = sum(p[i].passed for i in ids), sum(c[i].passed for i in ids)
        gain = cm - pm
        accepted = gain >= threshold - 1e-9 and cp >= pp
        reason = ("accepted" if accepted else f"gain {gain:+.3f} < {threshold}" if gain < threshold - 1e-9
                  else f"passed dropped {pp} -> {cp}")
        return {"n": len(ids), "parent_mean": round(pm, 4), "cand_mean": round(cm, 4), "gain": round(gain, 4),
                "parent_passed": pp, "cand_passed": cp, "accepted": accepted, "reason": reason}


def proposals_from_run(result: RunResult, run_id: str) -> list[dict]:
    """Verifier failures become pitfalls; agents may add proposals as JSON {"proposals": [...]} in notes.

    A failure line "Role: message" targets that role; anything else targets "all".
    Each proposal cites this run and its verifier score as evidence.
    """
    evidence = [{"run_id": run_id, "task_id": result.task_id,
                 "measurement": f"score={result.score:.3f} passed={result.passed}"}]
    out = []
    for line in result.failures:
        role, sep, msg = line.partition(": ")
        if not sep or " " in role:
            role, msg = "all", line
        out.append({"op": "add", "kind": "pitfall", "roles": [role], "text": msg.strip()[:MAX_TEXT], "evidence": evidence})
    notes = _parse_json(result.notes) if result.notes.strip().startswith("{") else {}
    for p in notes.get("proposals", []) if isinstance(notes.get("proposals"), list) else []:
        if isinstance(p, dict):
            out.append({"op": "add", **p, "evidence": list(p.get("evidence") or []) + evidence})
    return out


class Evolver:
    """Runs generations: learn on train, settle disputes and review, gate candidates on dev; held-out only in final_report."""

    def __init__(self, run_team: RunTeam, llm: LLM, reviewer_model: str, brain: SharedBrain,
                 versions_dir: str | Path, max_workers: int = 4, seed: int = 0,
                 initial: TeamVersion | None = None, gate: Gate | None = None,
                 reports_dir: str | Path | None = None, max_disputes: int = 2):
        self.run_team, self.llm, self.brain, self.seed = run_team, llm, brain, seed
        self.max_disputes = max_disputes
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path(reports_dir) if reports_dir else self.versions_dir.parent / "reports"
        self.max_workers = max_workers
        self.reviewer = Reviewer(llm, reviewer_model)
        self.gate = gate or Gate()
        self.v0 = initial or TeamVersion.load(self.versions_dir / "v0.json")
        self.v0.save(self.versions_dir)
        self.current = self.v0
        self.history: list[dict] = []
        self._cache: dict[tuple, RunResult] = {}
        self._last_score: dict[str, float] = {}
        self._dispute_tests: Counter = Counter()
        self._heldout: frozenset[str] = frozenset()
        self._final = False
        self._lock = threading.Lock()

    def _log(self, record: dict) -> None:
        with self._lock, open(self.versions_dir / "ledger.jsonl", "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def _one(self, version: TeamVersion, brain: SharedBrain, task: Task, run_id: str | None) -> RunResult:
        if task.id in self._heldout and not self._final:
            raise RuntimeError(f"held-out task {task.id} used before final_report")
        if run_id is None:  # evaluation: read-only on the brain, cached per brain state
            key = (version.key(), task.id, self.seed, str(brain.dir), brain.version(), tuple(sorted(brain._hidden)))
            if key not in self._cache:
                self._cache[key] = self.run_team(version, brain, task, self.llm)
            return self._cache[key]
        result = self.run_team(version, brain, task, self.llm)
        with self._lock:
            baseline = self._last_score.get(task.id)
            self._last_score[task.id] = result.score
        if baseline is not None:
            brain.credit(result, baseline)
        proposals = proposals_from_run(result, run_id)
        if proposals:
            brain.propose(proposals, run_id)
        return result

    def _batch(self, version: TeamVersion, brain: SharedBrain, tasks: list[Task], tag: str | None = None) -> list[RunResult]:
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(lambda t: self._one(version, brain, t, f"{tag}-{t.id}" if tag else None), tasks))

    def _settle_disputes(self, g: int, version: TeamVersion, dev_tasks: list[Task]) -> list[dict]:
        """Scheduled test run: each side of a dispute runs on dev with the other hidden; the higher mean wins.

        Least-tested pairs go first, so a pair that keeps tying cannot starve the rest of the cap.
        """
        if not dev_tasks:
            return []
        out = []
        for a, b in sorted(self.brain.disputes(), key=lambda p: (self._dispute_tests[p], p))[: self.max_disputes]:
            self._dispute_tests[(a, b)] += 1
            means = {keep: mean(r.score for r in self._batch(version, self.brain.without([hide]), dev_tasks))
                     for keep, hide in ((a, b), (b, a))}
            gain = means[a] - means[b]
            winner, loser = (None, None) if abs(gain) < 1e-9 else (a, b) if gain > 0 else (b, a)
            if winner:
                self.brain.resolve_dispute(winner, loser, f"dev mean {means[winner]:.2f} vs {means[loser]:.2f} "
                                                          f"on {len(dev_tasks)} tasks", run_id=f"g{g}-dispute")
            record = {"generation": g, "dispute": [a, b], "means": {k: round(v, 4) for k, v in means.items()},
                      "winner": winner}
            self._log(record)
            out.append(record)
        return out

    def run(self, generations: int, train_tasks: list[Task], dev_tasks: list[Task],
            heldout_tasks: list[Task]) -> TeamVersion:
        splits = [{t.id for t in s} for s in (train_tasks, dev_tasks, heldout_tasks)]
        if splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2]:
            raise ValueError("train, dev and held-out task ids must be disjoint")
        self._heldout = frozenset(splits[2])
        for g in range(generations):
            parent = self.current
            train = self._batch(parent, self.brain, train_tasks, tag=f"v{parent.version}g{g}")
            pruned = self.brain.prune()
            disputes = self._settle_disputes(g, parent, dev_tasks)
            mutations = self.reviewer.propose(parent, train, self.brain)
            for d in self.reviewer.dropped:
                self._log({"generation": g, "parent": parent.version, "mutation": d["mutation"],
                           "accepted": False, "reason": f"invalid: {d['reason']}"})
            parent_dev = self._batch(parent, self.brain, dev_tasks)
            best = None
            for m in mutations:
                cand = apply_mutation(parent, m, self.brain, parent.version + 1)
                is_consolidation = m["type"] == "promote_brain"
                view = self.brain.without(m["entry_ids"]) if is_consolidation else self.brain
                decision = self.gate.decide(parent_dev, self._batch(cand, view, dev_tasks),
                                            min_gain=0.0 if is_consolidation else None)
                self._log({"generation": g, "parent": parent.version, "mutation": m, **decision})
                if decision["accepted"] and (best is None or decision["gain"] > best[1]["gain"]):
                    best = (cand, decision, m)
            if best:
                cand, decision, m = best
                cand.changelog[-1] += f" [dev gain {decision['gain']:+.3f}]"
                cand.save(self.versions_dir)
                if m["type"] == "promote_brain":
                    self.brain.retire(m["entry_ids"], f"consolidated into v{cand.version}", run_id=f"g{g}")
                self.current = cand
            self.history.append({"generation": g, "parent": parent.version, "version": self.current.version,
                                 "train_mean": round(mean(r.score for r in train), 4) if train else 0.0,
                                 "dev_mean": round(mean(r.score for r in parent_dev), 4) if parent_dev else 0.0,
                                 "mutations": len(mutations), "accepted": best[2]["type"] if best else None,
                                 "pruned": len(pruned), "disputes_settled": sum(d["winner"] is not None for d in disputes),
                                 "brain": self.brain.stats()})
        return self.current

    def _row(self, label: str, version: TeamVersion, brain: SharedBrain, tasks: list[Task]) -> dict:
        rs = self._batch(version, brain, tasks)
        n = max(1, len(rs))
        return {"config": label, "version": version.version, "n": len(rs),
                "mean_score": round(sum(r.score for r in rs) / n, 4), "pass_rate": round(sum(r.passed for r in rs) / n, 4),
                "mean_iterations": round(sum(r.iterations for r in rs) / n, 3),
                "tokens": sum(r.prompt_tokens + r.completion_tokens for r in rs),
                "cost_usd": round(sum(r.cost_usd for r in rs), 6)}

    def final_report(self, heldout_tasks: list[Task]) -> list[dict]:
        """Ablation on unseen tasks: cold v0, v0 + learned brain, final version + learned brain."""
        self._final = True
        with tempfile.TemporaryDirectory() as tmp:
            rows = [self._row("v0 + empty brain (cold)", self.v0, SharedBrain(tmp), heldout_tasks),
                    self._row("v0 + final brain", self.v0, self.brain, heldout_tasks),
                    self._row(f"v{self.current.version} + final brain", self.current, self.brain, heldout_tasks)]
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "before_after.json").write_text(json.dumps({"rows": rows, "history": self.history}, indent=1))
        (self.reports_dir / "before_after.md").write_text(markdown_table(rows))
        return rows


def markdown_table(rows: list[dict]) -> str:
    lines = ["| Config | Mean score | Pass rate | Mean iters | Tokens | Cost (USD) |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['config']} | {r['mean_score']:.3f} | {r['pass_rate']:.0%} | {r['mean_iterations']:.2f} | "
              f"{r['tokens']} | {r['cost_usd']:.4f} |" for r in rows]
    return "\n".join(lines) + "\n"
