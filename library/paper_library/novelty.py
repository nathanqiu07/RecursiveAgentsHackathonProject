"""Library-owned novelty review against a complete, immutable content snapshot."""

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone

from .models import NoveltyReview, SemanticNoveltyDecision

MAX_NOVELTY_PAPERS = 200
MAX_NOVELTY_CHARS = 100_000


@dataclass(frozen=True)
class NoveltySnapshot:
    fingerprint: str
    candidate_fingerprint: str
    count: int
    candidate: dict
    papers: tuple[dict, ...]
    problem: str | None
    duplicate_ids: tuple[str, ...]


def normalized_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def capture_snapshot(db, paper_id):
    """Stream every Library row, even when semantic input exceeds the review budget."""
    candidate = db.execute("SELECT submission FROM papers WHERE id=?", (paper_id,)).fetchone()
    if candidate is None:
        raise ValueError("Candidate paper does not exist")
    encoded = candidate["submission"]
    submission = json.loads(encoded)
    text = submission.get("paper_text")
    readable = isinstance(text, str) and bool(text.strip())
    normalized = normalized_text(text) if readable else None
    problem = None if readable else "Candidate full paper text is required for novelty review."
    candidate_view = {"id": paper_id, "title": submission.get("title"), "paper_text": text}
    used = len(json.dumps(candidate_view, ensure_ascii=False))
    if used > MAX_NOVELTY_CHARS:
        problem = "The complete Library comparison exceeds the bounded novelty context limit."
    digest = hashlib.sha256(b"library-novelty-snapshot-v1\n")
    papers, duplicate_ids, count = [], [], 0
    for row in db.execute(
        "SELECT id,submission FROM papers WHERE id<>? ORDER BY sequence", (paper_id,)
    ):
        count += 1
        raw = row["submission"]
        digest.update(json.dumps([row["id"], raw], ensure_ascii=False).encode())
        digest.update(b"\n")
        try:
            prior = json.loads(raw)
            prior_text = prior.get("paper_text")
            if not isinstance(prior_text, str) or not prior_text.strip():
                problem = problem or "An existing Library paper has no readable full text."
            elif normalized is not None and normalized_text(prior_text) == normalized:
                if len(duplicate_ids) < 100:
                    duplicate_ids.append(row["id"])
            view = {"id": row["id"], "title": prior.get("title"), "paper_text": prior_text}
            used += len(json.dumps(view, ensure_ascii=False))
            if count > MAX_NOVELTY_PAPERS or used > MAX_NOVELTY_CHARS:
                problem = problem or (
                    "The complete Library comparison exceeds the bounded novelty context limit."
                )
            # A failed precheck never sends a partial catalog to the reviewer.
            if problem is None:
                papers.append(view)
        except (TypeError, ValueError, AttributeError, UnicodeError):
            problem = problem or "An existing Library paper could not be read for novelty review."
    return NoveltySnapshot(
        fingerprint=digest.hexdigest(),
        candidate_fingerprint=hashlib.sha256(encoded.encode()).hexdigest(),
        count=count,
        candidate=candidate_view,
        papers=tuple(papers),
        problem=problem,
        duplicate_ids=tuple(duplicate_ids),
    )


def review_snapshot(snapshot, reviewer):
    """Construct all provenance on the host; the LLM supplies only a bounded decision."""
    source_ids = []
    if snapshot.duplicate_ids:
        verdict, method = "not_novel", "exact_duplicate"
        source_ids = list(snapshot.duplicate_ids)
        rationale = (
            "The submitted full text duplicates an existing Library paper after normalization."
        )
    elif snapshot.problem:
        verdict, method, rationale = "inconclusive", "precheck", snapshot.problem
    else:
        try:
            decision = SemanticNoveltyDecision.model_validate(reviewer(snapshot))
            expected = {paper["id"] for paper in snapshot.papers}
            if {str(value) for value in decision.reviewed_paper_ids} != expected:
                raise ValueError("Reviewer did not cover the complete snapshot")
            verdict, method, rationale = decision.verdict, "semantic", decision.rationale
            source_ids = decision.source_paper_ids
        except Exception:
            verdict, method = "inconclusive", "precheck"
            rationale = (
                "The independent novelty reviewer could not produce a complete valid review."
            )
    return NoveltyReview(
        verdict=verdict,
        rationale=rationale,
        source_paper_ids=source_ids,
        snapshot_fingerprint=snapshot.fingerprint,
        snapshot_count=snapshot.count,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        method=method,
    )
