"""Bridge from the Evolver to SA's paper Library (library/, FastAPI on :8001).

SA's design: a research agent submits a paper (`POST /papers`), and Library's verifier records a
VerificationReport `{verdict: verified | not_verified | inconclusive, summary, evidence}` on it.
Here the Reviewer is the research agent: each mutation is submitted as a paper whose text states
its claim as a verification contract (conditions, acceptance criterion, reproduction). The Gate
is the verifier: its paired dev measurement becomes that paper's report.

Library's HTTP API has no endpoint for an external verdict; only its worker writes reports, by
leasing a paper and calling `PaperQueue.finish_verification` on the shared SQLite database. The
Gate does the same, through SA's own PaperQueue. If SA's worker already took the paper, or the
database is not on this machine, the report is submitted as a linked verification-report paper.

Every call is best effort: the loop keeps running, with a logged warning, when the Library is down.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger("sa_library")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8001"
DEFAULT_DB = REPO / "library" / "data" / "library.sqlite3"
WORKER_ID = "evolve-gate-verifier"
MAX_FIELD = 10_000  # SA's Notes limit, which also bounds report summary and evidence items


def _clip(text: str, limit: int = MAX_FIELD) -> str:
    text = " ".join(str(text).split()) if "\n" not in str(text) else str(text).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _num(x: Any, fmt: str = "+.4f") -> str:
    return "n/a" if not isinstance(x, (int, float)) or isinstance(x, bool) else format(x, fmt)


def _pct(x: Any) -> str:
    return "n/a" if not isinstance(x, (int, float)) or isinstance(x, bool) else f"{x:+.1f}%"


def row_key(generation: Any, parent: Any = None, mutation: Any = None, dispute: Any = None) -> str:
    """Stable id for one ledger row, shared by the live hook and the backfill so neither re-publishes."""
    body = ({"generation": generation, "dispute": list(dispute)} if dispute is not None
            else {"generation": generation, "parent": parent, "mutation": mutation})
    return hashlib.sha1(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- SA schema: PaperSubmission

def claim_submission(benchmark: str, generation: int, parent_version: int, mutation: dict,
                     context: dict | None = None) -> dict:
    """One Reviewer mutation as SA's PaperSubmission; `notes` carries the machine-readable contract."""
    ctx = context or {}
    claim = mutation.get("claim") or {}
    change = {k: v for k, v in mutation.items() if k != "claim"}
    target = change.get("role") or change.get("key") or change.get("tool") or ""
    min_gain, min_cut = ctx.get("min_gain"), ctx.get("min_cost_cut")
    claimed = []
    if isinstance(claim.get("dev_gain"), (int, float)):
        claimed.append(f"raises the mean dev score by {claim['dev_gain']:+.4f}")
    if isinstance(claim.get("cost_change_pct"), (int, float)):
        claimed.append(f"changes mean cost per task by {claim['cost_change_pct']:+.1f}%")
    conditions = [f"Benchmark {benchmark}", f"parent team version v{parent_version}", f"generation {generation}",
                  f"paired comparison on {ctx['n_dev']} dev tasks" if ctx.get("n_dev") else "paired comparison on the dev split",
                  "dev tasks are disjoint from train and held-out tasks"]
    if ctx.get("seed") is not None:
        conditions.append(f"seed {ctx['seed']}")
    if ctx.get("reviewer_model"):
        conditions.append(f"reviewer model {ctx['reviewer_model']}")
    train = (f" In this generation the parent scored a mean {ctx['train_mean']:.4f} on {ctx.get('n_train', '?')} train "
             "tasks; that run is the evidence the Reviewer read." if isinstance(ctx.get("train_mean"), (int, float)) else "")
    acceptance = ("Accept if the paired mean dev gain is at least "
                  f"{min_gain if min_gain is not None else 'the Gate minimum'} with no fewer passed dev tasks, or if the "
                  "gain is non-negative, no passed task is lost, and mean cost per task falls by at least "
                  f"{f'{min_cut:.0%}' if isinstance(min_cut, (int, float)) else 'the Gate cost cut'}. The claim is "
                  "reproduced when the change is accepted and every claimed benefit is measured with the claimed sign.")
    text = "\n\n".join([
        f"Abstract. The Reviewer agent of a self-improving agent team proposes one typed change to team version "
        f"v{parent_version} on the {benchmark} benchmark and claims a measurable effect on the dev split.",
        f"Change. {json.dumps(change, sort_keys=True, default=str)}",
        f"Rationale. {claim.get('why') or 'None given.'}",
        "Methods. " + "; ".join(conditions) + ". Both versions run the same dev tasks with the same seed and Shared "
        "Brain; every score comes from the benchmark's deterministic non-LLM scorer, never from a model." + train,
        "Claim. Applying this change " + (" and ".join(claimed) if claimed else "has no numeric claim attached") + ".",
        "Acceptance criteria. " + acceptance,
        "Reproduction. Apply the change with team/evolve.py apply_mutation to the parent version, run parent and "
        "candidate on the identical dev tasks, and compare paired mean scores, passed counts and mean cost per task "
        "(team/evolve.py Gate.decide).",
    ])
    contract = {"contract": "evolve-gate/v1", "benchmark": benchmark, "generation": generation,
                "parent_version": parent_version, "mutation": change, "claim": claim,
                "acceptance": {"min_gain": min_gain, "min_cost_cut": min_cut, "no_lost_passes": True},
                "verifier": "paired dev rerun scored by the benchmark's non-LLM scorer"}
    notes = json.dumps(contract, sort_keys=True, default=str)
    if len(notes) > MAX_FIELD:
        contract["mutation"] = {"type": change.get("type"), "truncated": True}
        notes = json.dumps(contract, sort_keys=True, default=str)[:MAX_FIELD]
    detail = next((change[k] for k in ("step", "text", "prompt", "value", "enabled", "entry_ids", "index") if k in change), "")
    detail = " ".join(str(detail).split())
    detail = f": {detail[:90]}{'...' if len(detail) > 90 else ''}" if detail != "" else ""
    return {"title": _clip(f"[{benchmark}] g{generation} v{parent_version}: {change.get('type')}"
                           f"{f' {target}' if target else ''}{detail}", 500),
            "paper_text": text, "authors": ["Reviewer agent (team/evolve.py)"],
            "submitted_by": "evolve-reviewer", "notes": notes}


def dispute_submission(benchmark: str, record: dict, n_dev: int | None = None, texts: dict | None = None) -> dict:
    a, b = record["dispute"]
    texts = texts or {}
    entries = "".join(f"\n\nEntry {k}: {texts[k]}" for k in (a, b) if texts.get(k))
    text = (f"Abstract. Shared Brain entries {a} and {b} contradict each other on the {benchmark} benchmark; a "
            f"scheduled test run decides which one the team keeps.{entries}\n\nMethods. Run the current team version on "
            f"{n_dev or 'the'} dev tasks once with {b} hidden and once with {a} hidden, same tasks and seed, scored by the "
            "benchmark's non-LLM scorer.\n\nClaim. One entry yields a strictly higher mean dev score than the other.\n\n"
            "Acceptance criteria. The entry whose run has the higher mean wins; equal means leave the dispute open.")
    return {"title": _clip(f"[{benchmark}] g{record['generation']} dispute test: {a} vs {b}", 500), "paper_text": text,
            "authors": ["Shared Brain (team/brain.py)"], "submitted_by": "evolve-brain",
            "notes": json.dumps({"contract": "evolve-dispute/v1", "benchmark": benchmark, **record}, default=str)[:MAX_FIELD]}


# ---------------------------------------------------------------- SA schema: VerificationReport

def gate_report(mutation: dict, decision: dict, record: dict | None = None) -> dict:
    """The Gate's paired dev measurement as SA's VerificationReport, claimed vs measured."""
    record = record or {}
    claim = mutation.get("claim") or {}
    accepted = bool(decision.get("accepted"))
    if str(decision.get("reason", "")).startswith("invalid"):
        return {"verdict": "inconclusive",
                "summary": _clip(f"Gate REJECTED before measurement ({decision['reason']}); the change never ran."),
                "evidence": ["Verifier: team/evolve.py validate_mutation (schema and bounds check)"]}
    outcome = record.get("verdict") or ("reproduced" if accepted else "not reproduced")
    verdict = "verified" if outcome == "reproduced" else "inconclusive" if accepted else "not_verified"
    measured = record.get("measured") or {"dev_gain": decision.get("gain"),
                                          "cost_change_pct": decision.get("cost_change_pct")}
    summary = (f"Gate {'ACCEPTED' if accepted else 'REJECTED'} ({decision.get('reason')}); claim {outcome}. "
               f"Claimed dev gain {_num(claim.get('dev_gain'))}, cost {_pct(claim.get('cost_change_pct'))}; measured "
               f"dev gain {_num(measured.get('dev_gain'))}, cost {_pct(measured.get('cost_change_pct'))} on "
               f"{decision.get('n')} paired dev tasks.")
    evidence = [f"Dev mean: parent {_num(decision.get('parent_mean'), '.4f')}, candidate "
                f"{_num(decision.get('cand_mean'), '.4f')} over n={decision.get('n')} paired tasks",
                f"Passed dev tasks: parent {decision.get('parent_passed')}, candidate {decision.get('cand_passed')}"]
    if "parent_cost" in decision:
        evidence.append(f"Mean cost per task ({decision.get('cost_unit')}): parent {decision.get('parent_cost')}, "
                        f"candidate {decision.get('cand_cost')} ({_pct(decision.get('cost_change_pct'))})")
    evidence.append("Verifier: team/evolve.py Gate, a paired dev rerun scored by the benchmark's non-LLM scorer")
    return {"verdict": verdict, "summary": _clip(summary), "evidence": [_clip(e) for e in evidence]}


def dispute_report(record: dict) -> dict:
    a, b = record["dispute"]
    means = record.get("means") or {}
    winner = record.get("winner")
    loser = (b if winner == a else a) if winner else None
    summary = (f"Entry {winner} wins: dev mean {_num(means.get(winner), '.4f')} vs {_num(means.get(loser), '.4f')} "
               f"for {loser}; {loser} is retired." if winner else
               f"Tie: dev means {_num(means.get(a), '.4f')} ({a}) and {_num(means.get(b), '.4f')} ({b}); the dispute stays open.")
    return {"verdict": "verified" if winner else "inconclusive", "summary": summary,
            "evidence": [f"Mean dev score keeping {k} (other hidden): {_num(v, '.4f')}" for k, v in means.items()]
            + ["Verifier: team/evolve.py Evolver._settle_disputes, scored by the benchmark's non-LLM scorer"]}


# ---------------------------------------------------------------- transport

class LibraryClient:
    """SA's Library HTTP API: POST /papers, GET /papers, GET /papers/{id}, GET /health."""

    def __init__(self, url: str = DEFAULT_URL, api_key: str | None = None, timeout: float = 5.0):
        self.url, self.api_key, self.timeout = url.rstrip("/"), api_key, timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc

    def health(self) -> dict:
        return self._request("GET", "/health")

    def submit(self, submission: dict) -> dict:
        return self._request("POST", "/papers", submission)

    def get(self, paper_id: str) -> dict:
        return self._request("GET", f"/papers/{paper_id}")

    def list_all(self) -> list[dict]:
        papers, offset = [], 0
        while True:
            page = self._request("GET", f"/papers?limit=100&offset={offset}")
            papers += page.get("papers", [])
            offset += 100
            if offset >= page.get("total", 0):
                return papers


class QueueVerdicts:
    """Writes a report the way SA's Library worker does: lease the paper, then PaperQueue.finish_verification."""

    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self._queue = None

    def queue(self):
        if self._queue is None:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Library database not on this machine: {self.db_path}")
            root = str(REPO / "library")
            if root not in sys.path:
                sys.path.append(root)
            from paper_library.queue import PaperQueue  # SA's code, unmodified
            self._queue = PaperQueue(self.db_path)
        return self._queue

    def finish(self, paper_id: str, report: dict) -> dict:
        queue = self.queue()
        token, now = secrets.token_urlsafe(32), time.time()
        # Same lease fields as PaperQueue.claim_for_verification, but for this paper only: that method
        # takes the oldest pending paper, which may belong to another loop or to a harness researcher.
        with queue.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            leased = db.execute(
                """UPDATE papers SET status='planning', started_at=COALESCE(started_at,?), worker_id=?,
                   lease_token=?, lease_until=?, compute_key=COALESCE(compute_key,?)
                   WHERE id=? AND status='pending_verification'""",
                (datetime.fromtimestamp(now, timezone.utc).isoformat(), WORKER_ID, token, now + 120,
                 str(uuid4()), paper_id)).rowcount
        if leased != 1:
            raise LookupError(f"paper {paper_id} is not pending_verification in {self.db_path}")
        return queue.finish_verification(paper_id, token, report=report)


# ---------------------------------------------------------------- publisher used by the Evolver

class NullLibrary:
    """No Library configured: every hook is a no-op."""

    def publish_claims(self, generation, parent_version, mutations, context=None):
        return [None] * len(mutations)

    def post_verdict(self, paper_id, mutation, decision, record=None):
        return None

    def publish_invalid(self, generation, parent_version, mutation, reason, context=None):
        return None

    def publish_dispute(self, record, n_dev=None, texts=None):
        return None


class LibraryPublisher(NullLibrary):
    def __init__(self, url: str = DEFAULT_URL, benchmark: str = "evolve", state_path: str | Path | None = None,
                 db_path: str | Path | None = None, api_key: str | None = None, timeout: float = 5.0,
                 client: Any = None, verdicts: Any = None):
        self.client = client or LibraryClient(url, api_key, timeout)
        self.verdicts = verdicts or QueueVerdicts(db_path or os.environ.get("SA_LIBRARY_DB") or DEFAULT_DB)
        self.benchmark = benchmark
        self.state_path = Path(state_path) if state_path else None
        self.papers: dict[str, dict] = {}  # row key -> {"paper_id", "verdict", ...}
        self.counts = {"papers": 0, "verdicts": 0, "failures": 0}
        if self.state_path and self.state_path.exists():
            for line in self.state_path.read_text().splitlines():
                entry = json.loads(line)
                self.papers.setdefault(entry["key"], {}).update(entry)

    def _remember(self, key: str, **fields: Any) -> None:
        self.papers.setdefault(key, {}).update(fields)
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "a") as fh:
                fh.write(json.dumps({"key": key, **fields}, default=str) + "\n")

    def _safe(self, what: str, fn, *args):
        try:
            return fn(*args)
        except Exception as exc:  # the loop must survive a down or misbehaving Library
            self.counts["failures"] += 1
            log.warning("SA Library %s failed at %s (%s: %s); the loop continues", what,
                        getattr(self.client, "url", "?"), type(exc).__name__, exc)
            return None

    def check(self) -> bool:
        return self._safe("health check", self.client.health) is not None

    def _publish(self, key: str, submission: dict, **meta: Any) -> str | None:
        known = self.papers.get(key, {}).get("paper_id")
        if known:
            return known
        paper = self._safe("paper submission", self.client.submit, submission)
        if not paper or not paper.get("id"):
            return None
        self.counts["papers"] += 1
        self._remember(key, paper_id=paper["id"], title=submission["title"], **meta)
        return paper["id"]

    def _verdict(self, key: str, paper_id: str | None, report: dict) -> str | None:
        if not paper_id or self.papers.get(key, {}).get("verdict"):
            return self.papers.get(key, {}).get("verdict")
        try:
            self.verdicts.finish(paper_id, report)
            where = "report"
        except Exception as exc:
            log.warning("SA Library report on %s not written through PaperQueue (%s: %s); "
                        "submitting it as a linked verification-report paper", paper_id, type(exc).__name__, exc)
            linked = {"title": _clip(f"Verification report for {paper_id}: {report['verdict']}", 500),
                      "paper_text": f"Verification report for Library paper {paper_id}.\n\nVerdict: {report['verdict']}."
                                    f"\n\n{report['summary']}\n\nEvidence.\n" + "\n".join(f"- {e}" for e in report["evidence"]),
                      "authors": ["Gate verifier (team/evolve.py)"], "submitted_by": WORKER_ID,
                      "notes": json.dumps({"verifies": paper_id, **report})[:MAX_FIELD]}
            if not self._safe("verification-report submission", self.client.submit, linked):
                return None
            where = "linked_paper"
        self.counts["verdicts"] += 1
        self._remember(key, paper_id=paper_id, verdict=report["verdict"], verdict_via=where)
        return report["verdict"]

    def publish_claims(self, generation, parent_version, mutations, context=None):
        return [self._publish(row_key(generation, parent_version, m),
                              claim_submission(self.benchmark, generation, parent_version, m, context),
                              kind="mutation", generation=generation)
                for m in mutations]

    def post_verdict(self, paper_id, mutation, decision, record=None, generation=None, parent_version=None):
        key = next((k for k, v in self.papers.items() if v.get("paper_id") == paper_id), None) if paper_id else None
        if key is None and generation is not None:
            key = row_key(generation, parent_version, mutation)
        return self._verdict(key or str(paper_id), paper_id, gate_report(mutation, decision, record))

    def publish_invalid(self, generation, parent_version, mutation, reason, context=None):
        key = row_key(generation, parent_version, mutation)
        pid = self._publish(key, claim_submission(self.benchmark, generation, parent_version, mutation, context),
                            kind="mutation", generation=generation)
        return self._verdict(key, pid, gate_report(mutation, {"accepted": False, "reason": f"invalid: {reason}"
                                                              if not str(reason).startswith("invalid") else reason}))

    def publish_dispute(self, record, n_dev=None, texts=None):
        key = row_key(record["generation"], dispute=record["dispute"])
        pid = self._publish(key, dispute_submission(self.benchmark, record, n_dev, texts),
                            kind="dispute", generation=record["generation"])
        return self._verdict(key, pid, dispute_report(record))


def from_env(versions_dir: str | Path) -> NullLibrary:
    """LibraryPublisher when SA_LIBRARY_URL is set (benchmark name from SA_LIBRARY_BENCHMARK), else a no-op."""
    url = os.environ.get("SA_LIBRARY_URL", "").strip()
    if not url:
        return NullLibrary()
    versions_dir = Path(versions_dir)
    benchmark = os.environ.get("SA_LIBRARY_BENCHMARK") or versions_dir.resolve().parent.name
    publisher = LibraryPublisher(url, benchmark, state_path=versions_dir / "library.jsonl",
                                 api_key=os.environ.get("SA_LIBRARY_API_KEY"))
    if publisher.check():
        log.warning("SA Library: publishing %s papers and Gate verdicts to %s", benchmark, url)
    return publisher


def add_library_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--library", metavar="URL", default=None,
                    help=f"publish each mutation as a paper and each Gate outcome as its verdict to SA's Library "
                         f"(default with --live: $SA_LIBRARY_URL or {DEFAULT_URL})")
    ap.add_argument("--no-library", action="store_true", help="do not publish to SA's Library")


def configure_from_args(args: argparse.Namespace, benchmark: str) -> str | None:
    """Sets SA_LIBRARY_URL / SA_LIBRARY_BENCHMARK for the Evolver. Live runs publish by default; offline runs opt in."""
    if getattr(args, "no_library", False):
        os.environ.pop("SA_LIBRARY_URL", None)
        return None
    url = args.library or (os.environ.get("SA_LIBRARY_URL") or DEFAULT_URL if getattr(args, "live", False) else None)
    if url:
        os.environ["SA_LIBRARY_URL"] = url
        os.environ.setdefault("SA_LIBRARY_BENCHMARK", benchmark)
    return url
