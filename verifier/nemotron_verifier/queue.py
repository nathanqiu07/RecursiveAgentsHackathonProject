import json
import sqlite3
import time
import uuid
import threading
from functools import wraps
from pathlib import Path


def synchronized(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return guarded


class Queue:
    def __init__(self, path):
        self.lock = threading.RLock()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("""CREATE TABLE IF NOT EXISTS evaluations (
            id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, kind TEXT NOT NULL,
            definition TEXT NOT NULL, status TEXT NOT NULL, result TEXT, created REAL NOT NULL
        )""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS benchmark_identities (
            benchmark_id TEXT PRIMARY KEY, identity TEXT NOT NULL
        )""")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(evaluations)")}
        for name, declaration in [("node_id", "TEXT"), ("started", "REAL"),
                                  ("cancel_requested", "INTEGER NOT NULL DEFAULT 0")]:
            if name not in columns:
                self.db.execute(f"ALTER TABLE evaluations ADD COLUMN {name} {declaration}")

    @synchronized
    def bind_benchmark(self, benchmark_id, identity):
        self.db.execute("INSERT OR IGNORE INTO benchmark_identities VALUES (?,?)", (benchmark_id, identity))
        bound = self.db.execute("SELECT identity FROM benchmark_identities WHERE benchmark_id=?", (benchmark_id,)).fetchone()
        if bound["identity"] != identity:
            raise ValueError("Benchmark is bound to different hardware, model, suite, or reference settings; create a new benchmark")

    @synchronized
    def baseline_job(self, identity):
        rows = self.db.execute("SELECT * FROM evaluations WHERE kind='baseline' ORDER BY created DESC").fetchall()
        for row in rows:
            if json.loads(row["definition"])["identity"] == identity:
                return self.view(row)
        return None

    @synchronized
    def submit_baseline(self, identity, deployment, node_id=None, pool_identity=None):
        # A successful reference is immutable. Only unsuccessful attempts can retry.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.baseline_job(identity)
            if existing and existing["status"] in {"queued", "running", "completed"}:
                result = existing
            else:
                attempts = self.db.execute("SELECT COUNT(*) FROM evaluations WHERE kind='baseline' AND request_key LIKE ?",
                                           ("baseline:" + identity + ":%",)).fetchone()[0]
                definition = {"deployment": deployment, "identity": identity}
                if node_id:
                    definition.update(node_id=node_id, pool_identity=pool_identity)
                result = self.submit(f"baseline:{identity}:{attempts}", "baseline", definition)
            self.db.execute("COMMIT")
            return result
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @synchronized
    def submit(self, key, kind, definition):
        serialized = json.dumps(definition, sort_keys=True)
        self.db.execute("INSERT OR IGNORE INTO evaluations (id,request_key,kind,definition,status,result,created) VALUES (?,?,?,?,'queued',NULL,?)",
                        (str(uuid.uuid4()), key, kind, serialized, time.time()))
        row = self.db.execute("SELECT * FROM evaluations WHERE request_key=?", (key,)).fetchone()
        if row["definition"] != serialized or row["kind"] != kind:
            raise ValueError("Idempotency key already belongs to a different request")
        return self.view(row)

    @staticmethod
    def view(row):
        result = json.loads(row["result"]) if row["result"] else {}
        return {"id": row["id"], "status": row["status"], "created_at": row["created"],
                "node_id": row["node_id"], "started_at": row["started"],
                "cancel_requested": bool(row["cancel_requested"]), **result}

    @synchronized
    def list(self, limit=100):
        rows = self.db.execute("SELECT * FROM evaluations ORDER BY created DESC,id LIMIT ?", (limit,)).fetchall()
        return [self.view(row) for row in rows]

    @synchronized
    def cancel(self, job_id):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM evaluations WHERE id=?", (job_id,)).fetchone()
            if row is not None and row["status"] == "queued":
                self.db.execute("UPDATE evaluations SET status='cancelled',cancel_requested=1,result=? WHERE id=?",
                                (json.dumps({"feedback": "Cancelled before a GPU was assigned. No score recorded."}), job_id))
            elif row is not None and row["status"] == "running":
                self.db.execute("UPDATE evaluations SET cancel_requested=1 WHERE id=?", (job_id,))
            result = self.get(job_id)
            self.db.execute("COMMIT")
            return result
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @synchronized
    def claim_node(self, node_id, pool_identity, node_identity):
        """One durable assignment per node; all three scheduler threads share SQLite."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            busy = self.db.execute("SELECT 1 FROM evaluations WHERE status='running' AND node_id=?", (node_id,)).fetchone()
            selected = None
            if not busy:
                rows = self.db.execute("SELECT * FROM evaluations WHERE status='queued' ORDER BY created,id").fetchall()
                for row in rows:
                    definition = json.loads(row["definition"])
                    if definition.get("pool_identity") != pool_identity:
                        continue
                    if row["kind"] == "baseline":
                        eligible = definition.get("node_id") == node_id and definition["identity"] == node_identity
                    else:
                        eligible = self.baseline(node_identity) is not None
                    if eligible:
                        selected = row
                        self.db.execute("UPDATE evaluations SET status='running',node_id=?,started=? WHERE id=?",
                                        (node_id, time.time(), row["id"]))
                        break
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": selected["id"], "kind": selected["kind"], "created": selected["created"],
                "definition": json.loads(selected["definition"])} if selected else None

    @synchronized
    def requeue(self, job_id):
        """Only a confirmed busy response before VM acceptance is safe to retry."""
        return self.db.execute("UPDATE evaluations SET status='queued',node_id=NULL,started=NULL "
                               "WHERE id=? AND status='running' AND cancel_requested=0", (job_id,)).rowcount == 1

    @synchronized
    def interrupted(self):
        return [dict(row) for row in self.db.execute("SELECT id,node_id FROM evaluations WHERE status='running'")]

    @synchronized
    def get(self, job_id):
        row = self.db.execute("SELECT * FROM evaluations WHERE id=?", (job_id,)).fetchone()
        return self.view(row) if row else None

    @synchronized
    def baseline(self, identity):
        rows = self.db.execute("SELECT * FROM evaluations WHERE kind='baseline' "
                               "AND status='completed' ORDER BY created DESC").fetchall()
        for row in rows:
            result = json.loads(row["result"])
            if result["metrics"]["identity"] == identity:
                return result["metrics"]
        return None

    @synchronized
    def claim(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM evaluations WHERE status='queued' "
                                   "ORDER BY created,id LIMIT 1").fetchone()
            if row:
                self.db.execute("UPDATE evaluations SET status='running' WHERE id=?", (row["id"],))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return {"id": row["id"], "kind": row["kind"],
                "definition": json.loads(row["definition"])} if row else None

    @synchronized
    def finish(self, job_id, status, result):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT cancel_requested FROM evaluations WHERE id=?", (job_id,)).fetchone()
            if row and row[0] and status in {"completed", "rejected"}:
                status, result = "cancelled", {"feedback": "Evaluation cancelled. No score recorded."}
            self.db.execute("UPDATE evaluations SET status=?,result=? WHERE id=? AND status='running'",
                            (status, json.dumps(result, allow_nan=False), job_id))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @synchronized
    def recover(self):
        # Starting a worker again never silently repeats costly interrupted evaluation.
        self.db.execute("UPDATE evaluations SET status='failed',result=? WHERE status='running'",
                        (json.dumps({"feedback": "Worker interrupted; no score recorded. Submit a new evaluation."}),))
