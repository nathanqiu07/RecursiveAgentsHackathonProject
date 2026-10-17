"""Durable run ownership and audit events; interrupted runs never replay publication."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .models import RunRequest

TERMINAL = {"submitted", "completed", "failed", "cancelled", "needs_attention"}


class LostLease(Exception):
    pass


class RunCancelled(Exception):
    pass


class RunConflict(Exception):
    pass


class RunStore:
    def __init__(self, path, now=time.time):
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.now = now
        self.worker_id = None
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                id TEXT PRIMARY KEY, request TEXT NOT NULL, status TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT '{}', result TEXT, error TEXT,
                compute_job_id TEXT, sandbox_id TEXT,
                cleanup_required INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                allocation_started INTEGER NOT NULL DEFAULT 0,
                lease_token TEXT, lease_until REAL, next_check REAL NOT NULL DEFAULT 0,
                created_at REAL NOT NULL, started_at REAL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES agent_runs(id),
                kind TEXT NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_run_keys (
                key TEXT PRIMARY KEY, request TEXT NOT NULL,
                run_id TEXT NOT NULL REFERENCES agent_runs(id)
            );
            CREATE TABLE IF NOT EXISTS agent_workers (
                id TEXT PRIMARY KEY, last_seen_at REAL NOT NULL
            );
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @staticmethod
    def view(row):
        if row is None:
            return None
        value = dict(row)
        for key in ("request", "state", "result"):
            value[key] = json.loads(value[key]) if value[key] is not None else None
        for key in ("cleanup_required", "cancel_requested", "allocation_started"):
            value[key] = bool(value[key])
        value.pop("lease_token", None)
        value.pop("lease_until", None)
        return value

    @staticmethod
    def _validate_key(idempotency_key):
        if idempotency_key is not None and (
            not idempotency_key
            or len(idempotency_key) > 200
            or any(not 33 <= ord(char) <= 126 for char in idempotency_key)
        ):
            raise ValueError("Idempotency-Key must contain 1–200 visible ASCII characters")

    def replay(self, request, idempotency_key):
        self._validate_key(idempotency_key)
        if idempotency_key is None:
            return None
        existing = self.db.execute(
            "SELECT request,run_id FROM agent_run_keys WHERE key=?", (idempotency_key,)
        ).fetchone()
        if existing:
            # Normalize older saved contracts with newly added optional defaults.
            original = RunRequest.model_validate_json(existing["request"])
            if original != request:
                raise RunConflict("This launch key was already used for a different request")
            return self.get(existing["run_id"])
        return None

    def create(self, request: RunRequest, idempotency_key=None, *, original_request=None):
        self._validate_key(idempotency_key)
        run_id = str(uuid4())
        now = self.now()
        encoded = json.dumps(request.model_dump(mode="json"), sort_keys=True, allow_nan=False)
        identity = original_request if original_request is not None else request
        original_encoded = json.dumps(identity.model_dump(mode="json"), sort_keys=True)
        with self.transaction():
            repeated = self.replay(identity, idempotency_key)
            if repeated:
                return repeated
            self.db.execute(
                "INSERT INTO agent_runs(id,request,status,created_at,updated_at) VALUES(?,?,?,?,?)",
                (run_id, encoded, "queued", now, now),
            )
            if idempotency_key is not None:
                self.db.execute(
                    "INSERT INTO agent_run_keys(key,request,run_id) VALUES(?,?,?)",
                    (idempotency_key, original_encoded, run_id),
                )
        return self.get(run_id)

    def touch_worker(self):
        if self.worker_id:
            self.db.execute(
                "INSERT INTO agent_workers(id,last_seen_at) VALUES(?,?) "
                "ON CONFLICT(id) DO UPDATE SET last_seen_at=excluded.last_seen_at",
                (self.worker_id, self.now()),
            )

    def remove_worker(self):
        if self.worker_id:
            self.db.execute("DELETE FROM agent_workers WHERE id=?", (self.worker_id,))

    def status(self):
        workers = self.db.execute(
            "SELECT count(CASE WHEN last_seen_at>? THEN 1 END),max(last_seen_at) "
            "FROM agent_workers",
            (self.now() - 120,),
        ).fetchone()
        counts = dict(
            self.db.execute("SELECT status,count(*) FROM agent_runs GROUP BY status").fetchall()
        )
        return {
            "workers_online": workers[0],
            "last_worker_seen_at": workers[1],
            "queued_runs": counts.get("queued", 0),
            "running_runs": counts.get("running", 0),
        }

    def get(self, run_id):
        return self.view(
            self.db.execute("SELECT * FROM agent_runs WHERE id=?", (run_id,)).fetchone()
        )

    def research_proposals(self, exclude_run_id):
        """Complete peer method snapshot, including unpublished and interrupted work.

        Exclude changing execution/status fields so unrelated progress cannot invalidate review.
        Keep older runs: an interrupted experiment is not permission to copy its method.
        """
        snapshot = []
        rows = self.db.execute(
            "SELECT id,request,state FROM agent_runs WHERE id<>? ORDER BY id",
            (exclude_run_id,),
        ).fetchall()
        for row in rows:
            request, state = json.loads(row["request"]), json.loads(row["state"])
            proposal = state.get("research_proposal")
            if request.get("role") == "researcher" and proposal:
                snapshot.append({"run_id": row["id"], "proposal": proposal})
        return snapshot

    def save_reviewed_state(self, run_id, token, state, peer_snapshot):
        """Commit a review only if no peer changed its proposal during model review."""
        with self.transaction():
            self.owned(run_id, token)
            if self.research_proposals(run_id) != peer_snapshot:
                return False
            self.db.execute(
                "UPDATE agent_runs SET state=?,updated_at=? WHERE id=? AND lease_token=?",
                (json.dumps(state, allow_nan=False), self.now(), run_id, token),
            )
        return True

    def list(self, limit=50, offset=0):
        rows = self.db.execute(
            "SELECT * FROM agent_runs ORDER BY created_at DESC,rowid DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        total = self.db.execute("SELECT count(*) FROM agent_runs").fetchone()[0]
        return {
            "runs": [self.view(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def cancel(self, run_id):
        with self.transaction():
            run = self.get(run_id)
            if run is None:
                return None
            if run["status"] == "queued":
                self.db.execute(
                    "UPDATE agent_runs SET status='cancelled',cancel_requested=1,updated_at=? "
                    "WHERE id=?",
                    (self.now(), run_id),
                )
            elif run["status"] == "running":
                self.db.execute(
                    "UPDATE agent_runs SET cancel_requested=1,updated_at=? WHERE id=?",
                    (self.now(), run_id),
                )
        return self.get(run_id)

    def claim(self, include_queued=True):
        with self.transaction():
            now = self.now()
            row = self.db.execute(
                "SELECT * FROM agent_runs WHERE next_check<=? "
                "AND (lease_until IS NULL OR lease_until<=?) "
                "AND (status='running' OR (? AND status='queued') OR cleanup_required=1) "
                "ORDER BY created_at,id LIMIT 1",
                (now, now, bool(include_queued)),
            ).fetchone()
            if row is None:
                return None
            token = str(uuid4())
            interrupted = row["status"] == "running"
            self.db.execute(
                "UPDATE agent_runs SET status=CASE WHEN status='queued' THEN 'running' ELSE status "
                "END,lease_token=?,lease_until=?,started_at=coalesce(started_at,?),updated_at=? "
                "WHERE id=?",
                (token, now + 120, now, now, row["id"]),
            )
            return self.get(row["id"]), token, interrupted

    def owned(self, run_id, token):
        row = self.db.execute(
            "SELECT * FROM agent_runs WHERE id=? AND lease_token=? AND lease_until>?",
            (run_id, token, self.now()),
        ).fetchone()
        if row is None:
            raise LostLease()
        return self.view(row)

    def heartbeat(self, run_id, token):
        with self.transaction():
            run = self.owned(run_id, token)
            self.db.execute(
                "UPDATE agent_runs SET lease_until=?,updated_at=? WHERE id=? AND lease_token=?",
                (self.now() + 120, self.now(), run_id, token),
            )
            self.touch_worker()
        if run["cancel_requested"]:
            raise RunCancelled()

    def update(self, run_id, token, **fields):
        allowed = {
            "state",
            "result",
            "error",
            "compute_job_id",
            "sandbox_id",
            "cleanup_required",
            "allocation_started",
        }
        if not fields or set(fields) - allowed:
            raise ValueError("Invalid run update")
        values = []
        for key, value in fields.items():
            values.append(
                json.dumps(value, allow_nan=False) if key in {"state", "result"} else value
            )
        assignments = ",".join(f"{key}=?" for key in fields)
        with self.transaction():
            self.owned(run_id, token)
            self.db.execute(
                f"UPDATE agent_runs SET {assignments},updated_at=? WHERE id=? AND lease_token=?",
                (*values, self.now(), run_id, token),
            )

    def event(self, run_id, token, kind, data):
        with self.transaction():
            self.owned(run_id, token)
            self.db.execute(
                "INSERT INTO agent_events(run_id,kind,data,created_at) VALUES(?,?,?,?)",
                (run_id, kind, json.dumps(data, allow_nan=False), self.now()),
            )

    def events(self, run_id, after=0, limit=100):
        rows = self.db.execute(
            "SELECT sequence,kind,data,created_at FROM agent_events WHERE run_id=? AND sequence>? "
            "ORDER BY sequence LIMIT ?",
            (run_id, after, limit),
        ).fetchall()
        return [{**dict(row), "data": json.loads(row["data"])} for row in rows]

    def finish(self, run_id, token, status, error=None, result=None):
        if status not in TERMINAL:
            raise ValueError("Invalid terminal status")
        with self.transaction():
            self.owned(run_id, token)
            self.db.execute(
                "UPDATE agent_runs SET status=?,error=?,result=?,lease_token=NULL,lease_until=NULL,"
                "next_check=?,updated_at=? WHERE id=? AND lease_token=?",
                (
                    status,
                    error,
                    json.dumps(result, allow_nan=False),
                    self.now() + 5,
                    self.now(),
                    run_id,
                    token,
                ),
            )
