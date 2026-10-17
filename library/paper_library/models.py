"""Library-owned verification records and generic compute-job metadata."""

import json
import math
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    field_validator,
    model_validator,
)

MAX_OPAQUE_JSON_BYTES = 512 * 1024
MAX_OPAQUE_JSON_DEPTH = 32
ManagerStatus = Literal[
    "pending_verification", "planning", "verifying", "blocked", "completed", "failed"
]
ComputeStatus = Literal["queued", "provisioning", "running", "completed", "failed", "cancelled"]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
PaperText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000_000)
]
Notes = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=10_000)]


def validate_opaque_json(value):
    """Bound stored JSON while keeping persistence independent of planning schemas."""

    def check(item, depth=1):
        kind = type(item)
        if kind in (dict, list):
            if depth > MAX_OPAQUE_JSON_DEPTH:
                raise ValueError("JSON metadata nesting exceeds the Library limit")
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError("JSON object keys must be strings")
                    check(child, depth + 1)
            else:
                for child in item:
                    check(child, depth + 1)
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("JSON metadata numbers must be finite")
        elif kind not in (str, int, bool, type(None)):
            raise ValueError("Metadata must contain JSON values only")

    check(value)
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ValueError("Metadata must contain finite JSON values") from exc
    if len(encoded) > MAX_OPAQUE_JSON_BYTES:
        raise ValueError("JSON metadata must not exceed 512 KiB")
    return value


class PaperSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Title
    paper_url: HttpUrl | None = None
    paper_text: PaperText | None = None
    authors: list[Name] = Field(default_factory=list, max_length=100)
    submitted_by: Name | None = None
    notes: Notes | None = None

    @model_validator(mode="after")
    def require_paper(self):
        if self.paper_url is None and self.paper_text is None:
            raise ValueError("Provide paper_url or paper_text")
        if self.paper_url is not None and (self.paper_url.username or self.paper_url.password):
            raise ValueError("paper_url must not contain embedded credentials")
        return self


class StoredPaperSubmission(PaperSubmission):
    # Preserve old records as data, without importing the manager's requirement models.
    verification_requirements: list[dict[str, Any]] = Field(default_factory=list, max_length=50)

    @field_validator("verification_requirements", mode="before")
    @classmethod
    def bounded_legacy_requirements(cls, value):
        return validate_opaque_json(value)


class VerificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["verified", "not_verified", "inconclusive"]
    summary: Notes
    evidence: list[Notes] = Field(default_factory=list, max_length=100)


class SemanticNoveltyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["novel", "not_novel", "inconclusive"]
    rationale: Notes
    reviewed_paper_ids: list[UUID] = Field(max_length=200)
    source_paper_ids: list[UUID] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def valid_sources(self):
        for values in (self.reviewed_paper_ids, self.source_paper_ids):
            if len(values) != len(set(values)):
                raise ValueError("Novelty source IDs must be unique")
        if not set(self.source_paper_ids) <= set(self.reviewed_paper_ids):
            raise ValueError("Novelty sources must belong to reviewed papers")
        if self.verdict == "not_novel" and not self.source_paper_ids:
            raise ValueError("A non-novel decision must cite an existing Library paper")
        return self


class NoveltyReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["novel", "not_novel", "inconclusive"]
    rationale: Notes
    source_paper_ids: list[UUID] = Field(default_factory=list, max_length=100)
    snapshot_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_count: int = Field(ge=0)
    reviewed_at: str
    method: Literal["semantic", "exact_duplicate", "precheck"]

    @field_validator("source_paper_ids")
    @classmethod
    def unique_sources(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Novelty source IDs must be unique")
        return value


class Paper(BaseModel):
    id: UUID
    status: ManagerStatus
    created_at: str
    submission: StoredPaperSubmission
    started_at: str | None = None
    completed_at: str | None = None
    worker_id: str | None = None
    report: VerificationReport | None = None
    experimental_report: VerificationReport | None = None
    novelty: NoveltyReview | None = None
    error: str | None = None
    plan: dict[str, Any] | None = None
    blocked_reason: str | None = None
    cleanup_required: bool = False
    node_id: str | None = None
    sandbox_id: str | None = None
    resource_manager_job_id: UUID | None = None
    resource_manager_status: ManagerStatus | None = None
    compute_job_id: UUID | None = None
    compute_job_status: ComputeStatus | None = None

    @field_validator("plan", mode="before")
    @classmethod
    def bounded_plan(cls, value):
        return validate_opaque_json(value)


class ComputeJobUpdate(BaseModel):
    # Compute responses cannot supply Library-owned plans, reports, or scientific verdicts.
    model_config = ConfigDict(extra="ignore", strict=True)

    id: UUID
    status: ComputeStatus
    cleanup_required: bool = False
    node_id: Name | None = None
    sandbox_id: Name | None = None

    @field_validator("id", mode="before")
    @classmethod
    def parse_manager_uuid(cls, value):
        if not isinstance(value, (str, UUID)):
            raise ValueError("Manager identifiers must be UUID strings")
        try:
            return UUID(str(value))
        except ValueError as exc:
            raise ValueError("Manager identifiers must be UUID strings") from exc


class PaperList(BaseModel):
    papers: list[Paper]
    total: int
    limit: int
    offset: int
