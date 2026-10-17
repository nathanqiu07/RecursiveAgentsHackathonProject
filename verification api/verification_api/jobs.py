"""Durable FIFO queue with exclusive claims and renewable worker leases."""

import json
import math
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl, model_validator

LEASE_SECONDS = 120


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def validate_repository_path(value: str, field_name: str):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or "\0" in value:
        raise ValueError(f"{field_name} must be a relative path inside the repository")


class JobSubmission(StrictModel):
    repository: HttpUrl
    commit: str = Field(pattern=r"^(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})$")
    manifest_path: str = Field(default="verification.json", min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_source(self):
        if self.repository.username or self.repository.password:
            raise ValueError("Use host-managed Git credentials, not credentials in the URL")
        if self.repository.query or self.repository.fragment:
            raise ValueError("Repository URL must not contain query parameters or fragments")
        validate_repository_path(self.manifest_path, "manifest_path")
        return self


class BenchmarkSpec(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    workload: str = Field(min_length=1, max_length=100)
    concurrency: int = Field(ge=1, le=4096)
    num_requests: int = Field(ge=1, le=1000000)
    input_tokens: int | None = Field(default=None, ge=1, le=10000000)
    output_tokens: int = Field(ge=1, le=1000000)
    warmup_requests: int = Field(ge=0, le=10000)
    parameters: dict[str, Any] = Field(default_factory=dict)


class JobManifest(StrictModel):
    schema_version: Literal[1]
    runner_script: str = Field(min_length=1, max_length=512)
    parameters: dict[str, Any] = Field(default_factory=dict)
    engine: dict[str, Any] | None = None
    benchmarks: list[BenchmarkSpec] | None = Field(default=None, min_length=1, max_length=1000)
    timeout_seconds: int = Field(ge=1, le=86400)

    @model_validator(mode="after")
    def unique_benchmarks(self):
        validate_repository_path(self.runner_script, "runner_script")
        if self.benchmarks and len({spec.id for spec in self.benchmarks}) != len(self.benchmarks):
            raise ValueError("Benchmark IDs must be unique")
        json.dumps(self.model_dump(), allow_nan=False)
        return self


# Existing benchmark integrations may keep importing the earlier name.
BenchmarkManifest = JobManifest


class BenchmarkResult(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    status: Literal["completed", "failed"]
    metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_metrics(self):
        if any(not math.isfinite(value) or value < 0 for value in self.metrics.values()):
            raise ValueError("Metrics must be finite nonnegative numbers")
        if self.status == "failed" and not self.error:
            raise ValueError("Failed benchmark results require an error")
        return self


class RunReport(StrictModel):
    schema_version: Literal[1]
    result: dict[str, Any] | None = None
    benchmarks: list[BenchmarkResult] | None = Field(default=None, min_length=1, max_length=1000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    worker: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_report(self):
        if self.result is None and self.benchmarks is None:
            raise ValueError("A report requires result or benchmarks")
        if self.benchmarks and len({item.id for item in self.benchmarks}) != len(self.benchmarks):
            raise ValueError("Result IDs must be unique")
        json.dumps(self.model_dump(), allow_nan=False)
        return self


class ClaimRequest(StrictModel):
    node_id: str = Field(
        min_length=1, max_length=100, validation_alias=AliasChoices("node_id", "worker_id")
    )


class NodeRegistration(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    config: dict[str, Any]

    @model_validator(mode="after")
    def valid_config(self):
        json.dumps(self.config, allow_nan=False)
        return self


class NodeInfo(NodeRegistration):
    status: Literal["idle", "busy", "offline"]
    active_job_id: str | None
    created_at: float
    updated_at: float
    last_seen: float


class NodeList(StrictModel):
    nodes: list[NodeInfo]


class LeaseRequest(StrictModel):
    lease_token: str


class FinishRequest(LeaseRequest):
    status: Literal["completed", "failed"]
    report: dict[str, Any] | None = None
    error: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.status == "completed" and self.report is None:
            raise ValueError("Completed jobs require a report")
        if self.status == "failed" and not self.error:
            raise ValueError("Failed jobs require an error")
        if self.report is not None:
            RunReport.model_validate(self.report)
        return self


def validate_report(report, manifest):
    parsed = RunReport.model_validate(report)
    if manifest.benchmarks is not None:
        ids = {entry.id for entry in (parsed.benchmarks or [])}
        if ids != {spec.id for spec in manifest.benchmarks}:
            raise ValueError("Report must include each requested benchmark ID exactly once")
    return parsed.model_dump(exclude_none=True)


class QueueConflict(Exception):
    pass


class Queue:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, payload TEXT NOT NULL, worker_id TEXT,
                lease_token TEXT, lease_until REAL, report TEXT, error TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at)")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            if "node_config" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN node_config TEXT")
            db.execute("""CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY, config TEXT NOT NULL, created_at REAL NOT NULL,
                updated_at REAL NOT NULL, last_seen REAL NOT NULL
            )""")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def public(row):
        if row is None:
            return None
        data = dict(row)
        data.pop("lease_token", None)
        data["submission"] = json.loads(data.pop("payload"))
        data["report"] = json.loads(data["report"]) if data["report"] else None
        data["node_id"] = data["worker_id"]
        data["node_config"] = json.loads(data["node_config"]) if data["node_config"] else None
        return data

    @staticmethod
    def node_view(db, row):
        if row is None:
            return None
        node = dict(row)
        node["config"] = json.loads(node["config"])
        active = db.execute(
            "SELECT id FROM jobs WHERE worker_id=? AND status='running' AND lease_until>=? LIMIT 1",
            (node["id"], time.time()),
        ).fetchone()
        node["active_job_id"] = active["id"] if active else None
        node["status"] = (
            "offline"
            if node["last_seen"] < time.time() - LEASE_SECONDS
            else "busy"
            if active
            else "idle"
        )
        return node

    def register_node(self, registration: NodeRegistration):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self.expire(db)
            if db.execute(
                "SELECT id FROM jobs WHERE worker_id=? AND status='running' LIMIT 1",
                (registration.id,),
            ).fetchone():
                raise QueueConflict(
                    "Cannot register or change configuration while the node is busy"
                )
            now = time.time()
            db.execute(
                """INSERT INTO nodes(id,config,created_at,updated_at,last_seen) VALUES(?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET config=excluded.config,
                   updated_at=excluded.updated_at, last_seen=excluded.last_seen""",
                (registration.id, json.dumps(registration.config, allow_nan=False), now, now, now),
            )
            return self.node_view(
                db, db.execute("SELECT * FROM nodes WHERE id=?", (registration.id,)).fetchone()
            )

    def list_nodes(self):
        with self.connection() as db:
            return [
                self.node_view(db, row) for row in db.execute("SELECT * FROM nodes ORDER BY id")
            ]

    def get_node(self, node_id: str):
        with self.connection() as db:
            return self.node_view(
                db, db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
            )

    def node_heartbeat(self, node_id: str):
        with self.connection() as db:
            updated = db.execute("UPDATE nodes SET last_seen=? WHERE id=?", (time.time(), node_id))
            if updated.rowcount != 1:
                return None
            return self.node_view(
                db, db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
            )

    def submit(self, submission: JobSubmission):
        job_id = str(uuid4())
        now = time.time()
        with self.connection() as db:
            db.execute(
                "INSERT INTO jobs(id,status,created_at,updated_at,payload) VALUES(?,?,?,?,?)",
                (job_id, "queued", now, now, submission.model_dump_json()),
            )
            return self.public(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    @staticmethod
    def expire(db):
        # Do not retry automatically: the old worker might still be using the GPU.
        db.execute(
            """UPDATE jobs SET status='failed', error='worker_lease_expired', updated_at=?,
               lease_token=NULL, lease_until=NULL WHERE status='running' AND lease_until < ?""",
            (time.time(), time.time()),
        )

    def get(self, job_id: str):
        with self.connection() as db:
            self.expire(db)
            return self.public(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def claim(self, worker_id: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self.expire(db)
            node = db.execute("SELECT config FROM nodes WHERE id=?", (worker_id,)).fetchone()
            if node is None:
                raise QueueConflict("Node must register its ID and config before claiming jobs")
            db.execute("UPDATE nodes SET last_seen=? WHERE id=?", (time.time(), worker_id))
            if db.execute(
                "SELECT id FROM jobs WHERE worker_id=? AND status='running' LIMIT 1", (worker_id,)
            ).fetchone():
                return None
            row = db.execute(
                "SELECT id FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            token = str(uuid4())
            now = time.time()
            db.execute(
                """UPDATE jobs SET status='running', worker_id=?, lease_token=?, lease_until=?,
                   node_config=?,
                   updated_at=? WHERE id=?""",
                (worker_id, token, now + LEASE_SECONDS, node["config"], now, row["id"]),
            )
            result = self.public(
                db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
            )
            result["lease_token"] = token
            return result

    def heartbeat(self, job_id: str, token: str):
        with self.connection() as db:
            now = time.time()
            updated = db.execute(
                """UPDATE jobs SET lease_until=?, updated_at=? WHERE id=? AND status='running'
                   AND lease_token=? AND lease_until>=?""",
                (now + LEASE_SECONDS, now, job_id, token, now),
            )
            if updated.rowcount != 1:
                raise QueueConflict("Job is not running with this lease")
            db.execute(
                "UPDATE nodes SET last_seen=? WHERE id=(SELECT worker_id FROM jobs WHERE id=?)",
                (now, job_id),
            )
        return {"lease_seconds": LEASE_SECONDS}

    def finish(self, job_id: str, outcome: FinishRequest):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise QueueConflict("Job does not exist")
            now = time.time()
            updated = db.execute(
                """UPDATE jobs SET status=?, report=?, error=?, lease_token=NULL,
                   lease_until=NULL, updated_at=? WHERE id=? AND status='running'
                   AND lease_token=? AND lease_until>=?""",
                (
                    outcome.status,
                    json.dumps(outcome.report, allow_nan=False)
                    if outcome.report is not None
                    else None,
                    outcome.error,
                    now,
                    job_id,
                    outcome.lease_token,
                    now,
                ),
            )
            if updated.rowcount != 1:
                raise QueueConflict("Job is not running with this lease")
            db.execute(
                "UPDATE nodes SET last_seen=? WHERE id=(SELECT worker_id FROM jobs WHERE id=?)",
                (now, job_id),
            )
            return self.public(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def cancel(self, job_id: str):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return None
            if row["status"] not in {"queued", "running"}:
                raise QueueConflict("Only queued or running jobs can be cancelled")
            db.execute(
                """UPDATE jobs SET status='cancelled', lease_token=NULL, lease_until=NULL,
                   updated_at=? WHERE id=?""",
                (time.time(), job_id),
            )
            return self.public(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
