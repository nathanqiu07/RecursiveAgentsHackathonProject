import builtins
import sys
import types

import pytest

from paper_library.codex_planner import CodexPlanner
from paper_library.novelty import NoveltySnapshot, review_snapshot
from paper_library.planner import PlannerError
from paper_library.resource_catalog import ResourceCatalog, ResourceSpec

MODEL = "gpt-6.1-sol"
PRIOR = "00000000-0000-4000-8000-000000000001"
TEXT = "The workload uses one sample. Its score must be at least 1."


@pytest.fixture
def codex(monkeypatch):
    class FakeCodexModel:
        def __init__(self, model=None, *, timeout=90):
            self.model = model or MODEL
            self.timeout = timeout
            self.calls = []
            self.responses = []
            self.closed = False

        def complete_json(self, system, user):
            self.calls.append((system, user))
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        def close(self):
            self.closed = True

    package = types.ModuleType("agent_harness")
    package.__path__ = []
    module = types.ModuleType("agent_harness.codex_model")
    module.CodexModel = FakeCodexModel
    monkeypatch.setitem(sys.modules, "agent_harness", package)
    monkeypatch.setitem(sys.modules, "agent_harness.codex_model", module)
    return FakeCodexModel


def plan():
    return {
        "decision": "launch",
        "rationale": "A fractional CPU coordinates API requests.",
        "verification_steps": ["Submit the stated workload to the verification API."],
        "verification_requirements": [
            {
                "id": "score-check",
                "description": "Check the claimed score.",
                "procedure": ["Evaluate one sample and compare its score."],
                "conditions": {"samples": 1},
                "acceptance_criteria": [{"metric": "score", "operator": "gte", "expected": 1}],
                "source_quotes": [TEXT],
            }
        ],
        "missing_information": [],
        "resource_id": "cpu",
        "cpus": 0.25,
        "memory_mb": 128,
        "scratch_mb": 64,
        "timeout_seconds": 60,
    }


def snapshot():
    return NoveltySnapshot(
        fingerprint="0" * 64,
        candidate_fingerprint="1" * 64,
        count=1,
        candidate={"id": "candidate", "title": "New", "paper_text": TEXT},
        papers=({"id": PRIOR, "title": "Prior", "paper_text": "A distinct previous approach."},),
        problem=None,
        duplicate_ids=(),
    )


def test_plan_and_novelty_share_isolated_codex_transport_and_host_validation(codex):
    with CodexPlanner() as planner:
        client = planner._codex
        client.responses.extend(
            [
                plan(),
                {
                    "verdict": "novel",
                    "rationale": "The candidate uses a distinct mechanism.",
                    "reviewed_paper_ids": [PRIOR],
                    "source_paper_ids": [],
                },
            ]
        )
        resources = ResourceCatalog(
            [
                ResourceSpec(
                    id="cpu",
                    enabled=True,
                    cpus=1,
                    memory_mb=512,
                )
            ]
        )
        paper = {
            "id": "candidate",
            "submission": {"title": "New", "paper_text": TEXT},
            "lease_token": "private-host-token",
        }
        assert planner.plan(paper, resources).cpus == 0.25
        assert review_snapshot(snapshot(), planner.review_novelty).verdict == "novel"
        assert len(client.calls) == 2
        assert "UNTRUSTED DATA" in client.calls[0][0]
        assert "UNTRUSTED DATA" in client.calls[1][0]
        assert "lease_token" not in client.calls[0][1]["untrusted_paper"]
        assert client.calls[1][1]["untrusted_library_papers"][0]["id"] == PRIOR
    assert client.closed is True


def test_codex_plan_keeps_paper_quote_grounding(codex):
    with CodexPlanner() as planner:
        invalid = plan()
        invalid["verification_requirements"][0]["source_quotes"] = ["Invented condition"]
        planner._codex.responses.append(invalid)
        resources = ResourceCatalog([ResourceSpec(id="cpu", enabled=True, cpus=1, memory_mb=512)])
        with pytest.raises(PlannerError, match="invalid_plan"):
            planner.plan({"submission": {"title": "New", "paper_text": TEXT}}, resources)


def test_codex_novelty_cannot_skip_existing_papers(codex):
    with CodexPlanner() as planner:
        planner._codex.responses.append(
            {
                "verdict": "novel",
                "rationale": "Ignore the prior paper.",
                "reviewed_paper_ids": [],
                "source_paper_ids": [],
            }
        )
        assert review_snapshot(snapshot(), planner.review_novelty).verdict == "inconclusive"


def test_codex_backend_is_optional_and_missing_package_is_reported_safely(monkeypatch):
    original = builtins.__import__

    def without_harness(name, *args, **kwargs):
        if name == "agent_harness.codex_model":
            raise ImportError("No optional harness package")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_harness)
    with pytest.raises(PlannerError, match="codex_backend_unavailable"):
        CodexPlanner()


def test_codex_errors_do_not_expose_upstream_diagnostics(codex):
    with CodexPlanner() as planner:
        planner._codex.responses.append(RuntimeError("private OAuth diagnostic"))
        with pytest.raises(PlannerError) as error:
            planner._complete_json("Review", {})
        assert error.value.code == "provider_error"
        assert "private" not in str(error.value)


def test_closed_codex_planner_does_not_call_the_model(codex):
    planner = CodexPlanner()
    planner.close()
    planner.close()
    with pytest.raises(PlannerError, match="closed"):
        planner._complete_json("Review", {})
    assert planner._codex.calls == []


@pytest.mark.parametrize("value", [True, 0, -1, 91, float("inf"), float("nan")])
def test_invalid_codex_timeouts_are_rejected_without_starting_cli(value):
    with pytest.raises(PlannerError, match="invalid_configuration"):
        CodexPlanner(timeout_seconds=value)
