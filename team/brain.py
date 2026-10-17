"""Shared Brain: an evidence-gated, curated memory every agent reads before work and feeds after.

`brain.json` is the source of truth, `BRAIN.md` a human-readable render, `ledger.jsonl` an
append-only log of every write and why. Agents never edit entries: they submit proposals and
the Curator decides, so the brain stays lean, deduplicated and free of silent contradictions.
"""
from __future__ import annotations

import copy
import difflib
import fcntl
import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from team.types import LLM, RunResult

KINDS = ("fact", "lesson", "pitfall", "check_template", "playbook_step")
OPS = ("add", "update", "retire")
MAX_TEXT = 280
_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_STOP = {"the", "and", "for", "with", "must", "are", "not", "per", "that", "this", "into", "from"}

Judge = Callable[[str, str], str]


@dataclass
class Entry:
    id: str
    kind: str
    roles: list[str]
    text: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    created_run: str = ""
    last_validated_run: str = ""
    uses: int = 0
    helps: int = 0
    hurts: int = 0
    status: str = "active"
    supersedes: list[str] = field(default_factory=list)
    disputed_with: list[str] = field(default_factory=list)  # ids it contradicted at equal support
    touched: int = 0  # brain version at last create/validate: recency without parsing run ids

    @property
    def live(self) -> bool:
        return self.status != "retired"


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in _STOP}


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def has_evidence(evidence: Any) -> bool:
    """Evidence must cite a run and a measurement, or the claim is unfalsifiable."""
    return isinstance(evidence, list) and any(
        isinstance(e, dict) and e.get("run_id") and e.get("measurement") not in (None, "")
        for e in evidence)


def _strength(e: Entry) -> int:
    return len(e.evidence) + e.helps


def llm_judge(llm: LLM, model: str) -> Judge:
    """Contradiction judge backed by an LLM; anything unparseable counts as compatible."""
    def judge(a_text: str, b_text: str) -> str:
        reply = llm.chat(model, [
            {"role": "system", "content": 'Compare two rules. Reply with JSON {"verdict": "contradict"|"duplicate"|"compatible"} only.'},
            {"role": "user", "content": f"A: {a_text}\nB: {b_text}"},
        ], temperature=0.0, max_tokens=20, json_mode=True, role="judge")
        try:
            verdict = json.loads(reply.text).get("verdict", "")
        except (ValueError, AttributeError):
            verdict = reply.text.strip().strip('"').lower()
        return verdict if verdict in ("contradict", "duplicate", "compatible") else "compatible"
    return judge


class Curator:
    """Applies proposals to the entry table: evidence gate, dedup, contradiction resolution."""

    def __init__(self, judge: Judge | None = None, dup_ratio: float = 0.85,
                 judge_floor: float = 0.3, max_judged: int = 3):
        self.judge = judge
        self.dup_ratio = dup_ratio
        self.judge_floor = judge_floor
        self.max_judged = max_judged

    def merge(self, entries: dict[str, Entry], proposals: list[dict], run_id: str,
              version: int) -> list[dict]:
        return [self._apply(entries, p, run_id, version) for p in proposals]

    def _apply(self, entries: dict[str, Entry], p: dict, run_id: str, version: int) -> dict:
        op = p.get("op", "add")
        rec = {"run_id": run_id, "op": op, "id": p.get("id"), "text": str(p.get("text", ""))[:80]}
        error = self._invalid(entries, p, op)
        if error:
            return {**rec, "decision": "rejected", "reason": error}
        if op == "retire":
            entries[p["id"]].status = "retired"
            return {**rec, "decision": "retired", "reason": p.get("reason", "evidence-backed retire")}
        if op == "update":
            e = entries[p["id"]]
            e.text, e.kind, e.roles = p.get("text") or e.text, p.get("kind") or e.kind, p.get("roles") or e.roles
            self._absorb(e, p["evidence"], run_id, version)
            return {**rec, "decision": "updated", "reason": "evidence-backed update"}
        return self._add(entries, p, run_id, version, rec)

    def _invalid(self, entries: dict[str, Entry], p: dict, op: str) -> str | None:
        if op not in OPS:
            return f"unknown op {op!r}"
        if op != "add" and p.get("id") not in entries:
            return f"unknown id {p.get('id')!r}"
        if not has_evidence(p.get("evidence")):
            return "no evidence citing a run_id and a measurement"
        text = p.get("text")
        if op == "add" or text:
            if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
                return f"text must be 1..{MAX_TEXT} chars"
        if (op == "add" or p.get("kind")) and p.get("kind") not in KINDS:
            return f"kind must be one of {KINDS}"
        roles = p.get("roles")
        if (op == "add" or roles) and not (isinstance(roles, list) and roles and all(isinstance(r, str) for r in roles)):
            return "roles must be a non-empty list of strings"
        return None

    def _add(self, entries: dict[str, Entry], p: dict, run_id: str, version: int, rec: dict) -> dict:
        text = p["text"].strip()
        live = sorted((e for e in entries.values() if e.live), key=lambda e: e.id)
        ranked = sorted(((similarity(e.text, text), e) for e in live), key=lambda x: (-x[0], x[1].id))
        for ratio, e in ranked[: self.max_judged]:
            if ratio < self.judge_floor:
                break
            # Same wording but different numbers is a different rule, never a duplicate.
            if ratio >= self.dup_ratio and _NUMBER.findall(e.text) == _NUMBER.findall(text):
                verdict = "duplicate"
            elif self.judge:
                verdict = self.judge(e.text, text)
            else:
                continue
            if verdict == "duplicate":
                self._absorb(e, p["evidence"], run_id, version)
                e.roles = sorted(set(e.roles) | set(p["roles"]))
                return {**rec, "id": e.id, "decision": "merged", "reason": f"duplicate (ratio {ratio:.2f})"}
            if verdict == "contradict":
                new = self._create(entries, p, text, run_id, version)
                return {**rec, "id": new.id, **self._resolve(e, new)}
        new = self._create(entries, p, text, run_id, version)
        return {**rec, "id": new.id, "decision": "added", "reason": "new evidence-backed entry"}

    @staticmethod
    def _resolve(old: Entry, new: Entry) -> dict:
        if _strength(old) == _strength(new):
            old.status = new.status = "disputed"
            old.disputed_with.append(new.id)
            new.disputed_with.append(old.id)
            return {"decision": "disputed", "reason": f"contradicts {old.id}, equal support"}
        win, lose = (old, new) if _strength(old) > _strength(new) else (new, old)
        lose.status = "retired"
        win.supersedes.append(lose.id)
        return {"decision": "contradiction", "reason": f"{win.id} supersedes {lose.id} (more support)"}

    @staticmethod
    def _create(entries: dict[str, Entry], p: dict, text: str, run_id: str, version: int) -> Entry:
        salt = 0
        while True:
            eid = hashlib.sha1(f"{p['kind']}|{_norm(text)}|{salt}".encode()).hexdigest()[:8]
            if eid not in entries:
                break
            salt += 1
        entries[eid] = Entry(id=eid, kind=p["kind"], roles=sorted(set(p["roles"])), text=text,
                             evidence=list(p["evidence"]), created_run=run_id,
                             last_validated_run=run_id, touched=version)
        return entries[eid]

    @staticmethod
    def _absorb(e: Entry, evidence: list[dict], run_id: str, version: int) -> None:
        e.evidence.extend(ev for ev in evidence if ev not in e.evidence)
        e.last_validated_run, e.touched = run_id, version


class SharedBrain:
    """File-backed brain safe for concurrent threads and processes (flock + version counter)."""

    def __init__(self, path_dir: str | Path, curator: Curator | None = None):
        self.dir = Path(path_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.curator = curator or Curator()
        self._hidden: frozenset[str] = frozenset()
        self._tlock = threading.RLock()
        with self._locked(write=True):
            if not (self.dir / "brain.json").exists():
                self._save(0, {})

    @contextmanager
    def _locked(self, write: bool) -> Iterator[None]:
        with self._tlock, open(self.dir / "brain.lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _load(self) -> tuple[int, dict[str, Entry]]:
        data = json.loads((self.dir / "brain.json").read_text())
        return data["version"], {k: Entry(**v) for k, v in data["entries"].items()}

    def _save(self, version: int, entries: dict[str, Entry]) -> None:
        payload = {"version": version, "entries": {k: asdict(entries[k]) for k in sorted(entries)}}
        tmp = self.dir / "brain.json.tmp"
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
        os.replace(tmp, self.dir / "brain.json")
        (self.dir / "BRAIN.md").write_text(self._render_md(version, entries))

    def _mutate(self, fn: Callable[[dict[str, Entry], int], list[dict]]) -> list[dict]:
        """Every write is one locked read-modify-write that bumps the version and hits the ledger."""
        with self._locked(write=True):
            version, entries = self._load()
            records = fn(entries, version + 1)
            self._save(version + 1, entries)
            with open(self.dir / "ledger.jsonl", "a") as fh:
                for r in records:
                    fh.write(json.dumps({"ts": round(time.time(), 3), "brain_version": version + 1, **r}) + "\n")
        return records

    def _snapshot(self) -> tuple[int, dict[str, Entry]]:
        with self._locked(write=False):
            return self._load()

    def _visible(self, e: Entry, role: str) -> bool:
        return e.live and e.id not in self._hidden and (role in e.roles or "all" in e.roles)

    def without(self, ids: list[str]) -> "SharedBrain":
        """A read view that hides some entries, e.g. to test a version that absorbed them."""
        view = copy.copy(self)
        view._hidden = self._hidden | frozenset(ids)
        return view

    def version(self) -> int:
        return self._snapshot()[0]

    def read(self, role: str, task_text: str, budget_chars: int = 3000) -> tuple[str, list[str]]:
        """Entries for this role ranked by net help, task relevance, recency, then id (deterministic)."""
        _, entries = self._snapshot()
        query = _tokens(task_text)

        def rank(e: Entry) -> tuple:
            words = _tokens(e.text)
            relevance = len(query & words) / max(1, len(words))
            return (-round(e.helps - e.hurts + 2.0 * relevance, 6), -e.touched, e.id)

        return self._render(role, sorted((e for e in entries.values() if self._visible(e, role)), key=rank), budget_chars)

    def stable_prefix(self, role: str, budget_chars: int = 3000) -> str:
        """Task-independent render (kind, id order): byte-identical until the brain changes, so prefix-cacheable."""
        _, entries = self._snapshot()
        ordered = sorted((e for e in entries.values() if self._visible(e, role)),
                         key=lambda e: (KINDS.index(e.kind), e.id))
        return self._render(role, ordered, budget_chars)[0]

    @staticmethod
    def _render(role: str, ordered: list[Entry], budget_chars: int) -> tuple[str, list[str]]:
        if not ordered:
            return "", []
        text, ids = f"## Shared brain for {role} (evidence-backed; DISPUTED = unresolved conflict)\n", []
        for e in ordered:
            line = f"- [{e.kind}]{' [DISPUTED]' if e.status == 'disputed' else ''} {e.text} (id {e.id})\n"
            if len(text) + len(line) > budget_chars:
                break
            text += line
            ids.append(e.id)
        return (text, ids) if ids else ("", [])

    @staticmethod
    def _render_md(version: int, entries: dict[str, Entry]) -> str:
        lines = [f"# Shared Brain (version {version})", ""]
        for status in ("active", "disputed"):
            group = sorted((e for e in entries.values() if e.status == status), key=lambda e: (KINDS.index(e.kind), e.id))
            lines += [f"## {status.title()} ({len(group)})", ""]
            lines += [f"- `{e.id}` **{e.kind}** [{', '.join(e.roles)}] {e.text} "
                      f"(+{e.helps}/-{e.hurts}, {e.uses} uses, {len(e.evidence)} evidence)" for e in group]
            lines.append("")
        lines.append(f"Retired: {sum(not e.live for e in entries.values())}")
        return "\n".join(lines) + "\n"

    def propose(self, proposals: list[dict], run_id: str) -> list[dict]:
        """The only write path for agents; the Curator decides what lands."""
        return self._mutate(lambda entries, v: self.curator.merge(entries, proposals, run_id, v))

    def credit(self, result: RunResult, baseline_score: float, eps: float = 1e-9) -> list[dict]:
        ids = sorted(set(result.brain_ids_read))
        effect = ("helps" if result.score > baseline_score + eps
                  else "hurts" if result.score < baseline_score - eps else None)

        def fn(entries: dict[str, Entry], _v: int) -> list[dict]:
            for i in ids:
                if i in entries:
                    entries[i].uses += 1
                    if effect:
                        setattr(entries[i], effect, getattr(entries[i], effect) + 1)
            return [{"op": "credit", "ids": ids, "task_id": result.task_id, "score": result.score,
                     "baseline": baseline_score, "decision": effect or "tie", "reason": "paired score vs baseline"}]
        return self._mutate(fn)

    def prune(self, max_active: int = 40, min_uses: int = 3) -> list[dict]:
        """Retire entries that keep hurting, then the lowest-ranked ones beyond capacity."""
        def fn(entries: dict[str, Entry], _v: int) -> list[dict]:
            out = []
            for e in sorted((e for e in entries.values() if e.live), key=lambda e: e.id):
                if e.uses >= min_uses and e.hurts > e.helps + 1:
                    e.status = "retired"
                    out.append({"op": "prune", "id": e.id, "decision": "retired",
                                "reason": f"hurts {e.hurts} > helps {e.helps} + 1 after {e.uses} uses"})
            live = sorted((e for e in entries.values() if e.live), key=lambda e: (e.helps - e.hurts, e.touched, e.id))
            for e in live[: max(0, len(live) - max_active)]:
                e.status = "retired"
                out.append({"op": "prune", "id": e.id, "decision": "retired", "reason": f"over capacity {max_active}"})
            return out
        return self._mutate(fn)

    def retire(self, ids: list[str], reason: str, run_id: str = "system") -> list[dict]:
        def fn(entries: dict[str, Entry], _v: int) -> list[dict]:
            out = []
            for i in ids:
                if i in entries and entries[i].live:
                    entries[i].status = "retired"
                    out.append({"op": "retire", "run_id": run_id, "id": i, "decision": "retired", "reason": reason})
            return out
        return self._mutate(fn)

    def disputes(self) -> list[tuple[str, str]]:
        """Open contradictions as sorted (a, b) id pairs; only a measured test run may settle them."""
        _, entries = self._snapshot()
        return sorted({tuple(sorted((e.id, o))) for e in entries.values() if e.status == "disputed"
                       for o in e.disputed_with if o in entries and entries[o].status == "disputed"})

    def resolve_dispute(self, winner_id: str, loser_id: str, measurement: str, run_id: str) -> list[dict]:
        """The winner goes active with the measurement as evidence; the loser retires, superseded by it."""
        evidence = [{"run_id": run_id, "measurement": measurement}]

        def fn(entries: dict[str, Entry], version: int) -> list[dict]:
            rec = {"op": "resolve_dispute", "run_id": run_id}
            win, lose = entries.get(winner_id), entries.get(loser_id)
            if not (win and lose and win.status == lose.status == "disputed" and loser_id in win.disputed_with):
                return [{**rec, "id": winner_id, "decision": "rejected",
                         "reason": f"no open dispute between {winner_id} and {loser_id}"}]
            if not has_evidence(evidence):
                return [{**rec, "id": winner_id, "decision": "rejected", "reason": "no evidence citing a run_id and a measurement"}]
            lose.status = "retired"
            win.supersedes.append(loser_id)
            Curator._absorb(win, evidence, run_id, version)
            still_open = any(entries[o].status == "disputed" for o in win.disputed_with if o in entries)
            win.status = "disputed" if still_open else "active"
            return [{**rec, "id": winner_id, "decision": win.status, "reason": f"won dispute against {loser_id}: {measurement}"},
                    {**rec, "id": loser_id, "decision": "retired", "reason": f"lost dispute to {winner_id}: {measurement}"}]
        return self._mutate(fn)

    def get(self, ids: list[str]) -> list[Entry]:
        _, entries = self._snapshot()
        return [entries[i] for i in ids if i in entries]

    def top(self, n: int = 5) -> list[Entry]:
        _, entries = self._snapshot()
        return sorted((e for e in entries.values() if e.live), key=lambda e: (e.hurts - e.helps, -e.uses, e.id))[:n]

    def stats(self) -> dict:
        version, entries = self._snapshot()
        live = [e for e in entries.values() if e.live]
        count = max(1, len(live))
        return {"version": version, "active": sum(e.status == "active" for e in live),
                "disputed": sum(e.status == "disputed" for e in live),
                "retired": len(entries) - len(live), "total_chars": sum(len(e.text) for e in live),
                "avg_helps": round(sum(e.helps for e in live) / count, 3),
                "avg_hurts": round(sum(e.hurts for e in live) / count, 3)}
