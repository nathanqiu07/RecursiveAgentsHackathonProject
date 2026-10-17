"""Exercise the control plane with real temporary-workspace commands and fake APIs.

The fake compute manager below is exclusively a test fixture. Production commands
must pass through ResourceManager's isolated execution image.
"""

import copy
import json
from uuid import uuid4

import pytest

from agent_harness.clients import ServiceError
from agent_harness.engine import EndRun, Engine, GateError, run_once
from agent_harness.models import RunRequest
from agent_harness.sandbox import execute
from agent_harness.store import RunCancelled, RunStore

BENCHMARK_ID = "11111111-1111-4111-8111-111111111111"
PAPER_ID = "22222222-2222-4222-8222-222222222222"
OTHER_PAPER_ID = "33333333-3333-4333-8333-333333333333"
CHECKPOINT = "https://github.com/example/base/commit/" + "a" * 40
IDEA = "Use a deterministic triangular-number recurrence and compare it with summation."
ONLINE_PAPER = "https://arxiv.org/abs/2401.12345"
PROPOSAL = {
    "hypothesis": IDEA,
    "novelty_difference": (
        "Use the retrieved baseline to isolate recurrence correctness from throughput."
    ),
    "source_urls": [ONLINE_PAPER],
    "experiment_plan": [
        {
            "baseline": "Direct Python summation",
            "experiment": "Compare recurrence with direct summation at n=10",
            "metric": "Integer result equality",
            "success_criterion": "Both implementations produce 55 with an assertion",
        }
    ],
}
PAPER_TEXT = (
    "Hypothesis: triangular recurrence agrees with direct summation. Related work: the Library "
    "baseline uses direct summation. Method: compare both implementations at n=10 using Python. "
    "Observed result: both return 55. Reproduction: execute the included Python command. "
    "Limitation: this is a small CPU correctness experiment, not a performance claim. "
    "Retrieved related work: " + ONLINE_PAPER
)
EXPERIMENT = (
    "python3 -c 'n=10; baseline=sum(range(n+1)); candidate=n*(n+1)//2; "
    'assert baseline==candidate; print("observed", baseline, candidate)\''
)
ATTEMPT_TEST = (
    "python3 -c 'answer=sum(range(11)); assert answer==55; "
    'print("candidate correctness", answer)\''
)


def paper_record(paper_id=PAPER_ID, text="Direct summation is the prior baseline.", **changes):
    return {
        "id": paper_id,
        "status": "completed",
        "submission": {"title": "Summation baseline", "paper_text": text},
        "report": {"verdict": "verified", "summary": "Baseline reproduced."},
        **changes,
    }


def request(role="researcher", **changes):
    value = {
        "role": role,
        "objective": "Investigate deterministic summation and its correctness.",
        "allocation": {
            "resource_id": "test-cpu",
            "node_id": "test-node",
            "cpus": 1.0,
            "memory_mb": 128,
            "scratch_mb": 16,
            "timeout_seconds": 600,
        },
        "max_steps": 30,
    }
    if role == "attempt":
        value["benchmark_id"] = BENCHMARK_ID
    value.update(changes)
    return RunRequest.model_validate(value)


def paper_args(**changes):
    return {
        "title": "A tested triangular recurrence",
        "paper_text": PAPER_TEXT,
        "authors": ["Research agent"],
        "evidence_ids": ["execute-2"],
        **changes,
    }


def attempt_args(**changes):
    return {
        "title": "A tested summation candidate",
        "base_checkpoint_url": CHECKPOINT,
        "payload": {"answer": 55},
        "evidence_ids": ["execute-4"],
        **changes,
    }


def action(name, **arguments):
    return name, arguments


class FakeModel:
    def __init__(self, actions=(), novelty=(), supported=True):
        self.actions = list(actions)
        self.novelty = list(novelty)
        self.supported = supported
        self.model_calls = []
        self.novelty_calls = []
        self.evidence_calls = []

    def complete(self, messages, tools):
        self.model_calls.append(copy.deepcopy({"messages": messages, "tools": tools}))
        if not self.actions:
            return {"role": "assistant", "content": "No further supported result."}
        name, arguments = self.actions.pop(0)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{len(self.model_calls)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }

    def complete_json(self, system, value):
        if "Library novelty reviewer" in system:
            self.novelty_calls.append(copy.deepcopy(value))
            decision = self.novelty.pop(0) if self.novelty else True
            if isinstance(decision, dict):
                return decision
            return {
                "novel": decision,
                "rationale": "Compared substantive contributions against every supplied paper.",
                "related_paper_ids": [item["id"] for item in value["library_papers"]],
            }
        self.evidence_calls.append(copy.deepcopy(value))
        return {"supported": self.supported, "rationale": "Reviewed recorded measurements."}


class FakeWeb:
    def __init__(self):
        self.searches, self.reads = [], []

    def research_capabilities(self):
        return {
            "provider": "you_mcp",
            "profile": "free",
            "tools": ["you-search"],
            "contents_available": False,
        }

    def search_papers(self, query):
        self.searches.append(query)
        return {
            "provider": "you_mcp",
            "papers": [{"url": ONLINE_PAPER, "title": "Recorded test source"}],
        }

    def read_paper(self, url):
        self.reads.append(url)
        return {
            "url": url,
            "read_url": url.replace("/abs/", "/html/"),
            "text": (
                "An explicit test fixture of a retrieved primary paper "
                "describing a recurrence and baseline. "
            )
            * 8,
            "retrieved_at": 1000,
            "content_kind": "full_text",
            "truncated": False,
        }

    def search(self, query):
        self.searches.append(query)
        return {"results": [{"title": "Triangular numbers", "url": "https://example.org/math"}]}

    def read(self, url):
        self.reads.append(url)
        return {"url": url, "text": "Triangular numbers sum consecutive integers."}


class FakeServices:
    def __init__(self, root):
        self.root = root
        self.library = [paper_record()]
        self.jobs, self.allocations, self.sent = {}, {}, {}
        self.published_papers, self.published_attempts = [], []
        self.cancelled_jobs = []
        self.cleanup_pending = False
        self.publication_error = None
        self.before_publish = None
        self.benchmark_value = {
            "id": BENCHMARK_ID,
            "title": "Deterministic summation",
            "prompt": "Return the sum of the integers zero through ten.",
            "submission_schema": {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            "score_direction": "higher",
        }

    def resources(self):
        return [
            {
                "id": "test-cpu",
                "node_id": "test-node",
                "cpus": 1.0,
                "memory_mb": 128,
                "scratch_mb": 16,
                "max_timeout_seconds": 600,
            }
        ]

    def papers(self):
        return copy.deepcopy(self.library)

    def paper(self, paper_id):
        return copy.deepcopy(next(item for item in self.library if item["id"] == paper_id))

    def benchmark(self, benchmark_id):
        assert benchmark_id == BENCHMARK_ID
        return copy.deepcopy(self.benchmark_value)

    def allocate(self, allocation, key):
        if key not in self.allocations:
            job_id = str(uuid4())
            directory = self.root / job_id
            directory.mkdir()
            self.jobs[job_id] = {
                "id": job_id,
                "status": "running",
                "sandbox_id": "sandbox-" + job_id,
                "cleanup_required": False,
                "directory": directory,
                "lines": [],
            }
            self.allocations[key] = job_id
        return self.job(self.allocations[key])

    def job(self, job_id):
        return {
            key: value
            for key, value in self.jobs[job_id].items()
            if key not in {"directory", "lines"}
        }

    def send(self, job_id, message_id, line):
        # This direct local execution exists only in this fixture. The production
        # service sends the same line through ResourceManager's sandbox channel.
        key = (job_id, message_id)
        if key in self.sent:
            assert self.sent[key] == line
            return {}
        self.sent[key] = line
        message = json.loads(line)
        output = execute(
            message["command"], message["timeout_seconds"], self.jobs[job_id]["directory"]
        )
        result = {"type": "execution_result", "id": message["id"], **output}
        records = self.jobs[job_id]["lines"]
        records.append({"sequence": len(records) + 1, "line": json.dumps(result)})
        return {}

    def output(self, job_id, after):
        records = self.jobs[job_id]["lines"]
        return {"lines": copy.deepcopy(records[after:]), "next_cursor": len(records)}

    def cancel(self, job_id):
        self.cancelled_jobs.append(job_id)
        self.jobs[job_id].update(status="cancelled", cleanup_required=self.cleanup_pending)
        return self.job(job_id)

    def submit_paper(self, body):
        if self.before_publish:
            self.before_publish()
        self.published_papers.append(copy.deepcopy(body))
        if self.publication_error:
            raise self.publication_error
        return {"id": str(uuid4()), "status": "pending_verification"}

    def submit_attempt(self, benchmark_id, body):
        assert benchmark_id == BENCHMARK_ID
        self.published_attempts.append(copy.deepcopy(body))
        if self.publication_error:
            raise self.publication_error
        return {"id": str(uuid4()), "benchmark_id": benchmark_id, "status": "queued"}


@pytest.fixture
def harness(tmp_path):
    clock = [1000.0]
    store = RunStore(tmp_path / "runs.sqlite3", now=lambda: clock[0])
    services = FakeServices(tmp_path)
    yield store, services, FakeWeb(), clock
    store.close()


def owned_engine(harness, role="researcher", model=None, allocate=True, online=True):
    store, services, web, _ = harness
    store.create(request(role))
    run, token, _ = store.claim()
    engine = Engine(store, run, token, services, model or FakeModel(), web)
    if allocate:
        engine.allocate()
    engine.state.update(steps=0, experiments=[])
    if role == "researcher" and online:
        engine.tool("search_research_papers", {"query": "fixture recurrence paper"})
        engine.tool("read_research_paper", {"url": ONLINE_PAPER})
    engine.save()
    return engine


def tool(engine, step, name, **arguments):
    engine.state["steps"] = step
    if name == "review_novelty" and engine.role == "researcher":
        return engine.tool("propose_research_idea", {**PROPOSAL, "hypothesis": arguments["idea"]})[
            "novelty"
        ]
    return engine.tool(name, arguments)


def researcher_actions():
    return [
        action("search_library", query="summation"),
        action("read_paper", paper_id=PAPER_ID),
        action("search_research_papers", query="triangular recurrence"),
        action("read_research_paper", url=ONLINE_PAPER),
        action("propose_research_idea", **PROPOSAL),
        action("execute", command=EXPERIMENT, purpose="experiment", timeout_seconds=2),
        action("submit_paper", **paper_args(evidence_ids=["execute-6"])),
    ]


def attempt_actions():
    return [
        action("read_benchmark"),
        action("search_library", query="summation"),
        action("read_paper", paper_id=PAPER_ID),
        action("execute", command=ATTEMPT_TEST, purpose="test", timeout_seconds=2),
        action("submit_attempt", **attempt_args()),
    ]


def test_researcher_submits_paper_to_library_with_real_recorded_experiments(harness):
    store, services, web, _ = harness
    services.library.append(
        paper_record(
            OTHER_PAPER_ID, "An unverified alternative.", status="pending_verification", report=None
        )
    )
    run = store.create(request())
    model = FakeModel(researcher_actions())

    assert run_once(store, services, model, web)
    saved = store.get(run["id"])
    assert saved["status"] == "submitted"
    assert saved["result"]["kind"] == "paper"
    assert saved["result"]["verification_status"] == "pending_verification"
    assert saved["cleanup_required"] is False
    assert len(services.published_papers) == 1
    assert services.published_attempts == []
    paper = services.published_papers[0]
    assert PAPER_TEXT in paper["paper_text"]
    assert "observed 55 55" in paper["paper_text"]
    assert "Harness experiment records" in paper["paper_text"]
    assert paper["submitted_by"] == "researcher:" + run["id"]
    assert len(model.novelty_calls) == 2
    assert all(
        {paper["id"] for paper in call["library_papers"]} == {PAPER_ID, OTHER_PAPER_ID}
        for call in model.novelty_calls
    )
    assert model.novelty_calls[0]["final_paper"] is False
    assert model.novelty_calls[1]["candidate"] == PAPER_TEXT
    assert model.novelty_calls[1]["final_paper"] is True
    assert model.evidence_calls[0]["recorded_executions"][0]["output"] == "observed 55 55\n"
    assert web.searches == ["triangular recurrence"]
    assert web.reads == [ONLINE_PAPER]
    kinds = [event["kind"] for event in store.events(run["id"])]
    assert kinds.index("novelty_review") < kinds.index("experiment_started")
    assert kinds.index("experiment_result") < kinds.index("publication_started")
    assert kinds[-1] == "sandbox_cleanup"


def test_attempt_reads_existing_paper_tests_candidate_and_submits_to_benchmark(harness):
    store, services, web, _ = harness
    run = store.create(request("attempt"))
    model = FakeModel(attempt_actions())
    assert run_once(store, services, model, web)
    saved = store.get(run["id"])
    assert saved["status"] == "submitted"
    assert saved["result"]["kind"] == "attempt"
    assert saved["result"]["verification_status"] == "queued"
    assert saved["state"]["read_paper_ids"] == [PAPER_ID]
    assert saved["state"]["experiments"][0]["output"] == "candidate correctness 55\n"
    assert len(services.published_attempts) == 1
    assert services.published_attempts[0] == {
        "title": "A tested summation candidate",
        "base_checkpoint_url": CHECKPOINT,
        "payload": {"answer": 55},
        "submitted_by": "attempt:" + run["id"],
    }
    assert not services.published_papers
    assert not model.novelty_calls


def test_each_run_gets_a_different_workspace_and_compute_job(harness):
    store, services, web, _ = harness
    first = store.create(request("attempt"))
    first_model = FakeModel(
        [
            action("read_benchmark"),
            action(
                "execute",
                command="printf private > private.txt",
                purpose="setup",
                timeout_seconds=1,
            ),
            action("finish", summary="First workspace prepared."),
        ]
    )
    assert run_once(store, services, first_model, web)
    second = store.create(request("attempt"))
    second_model = FakeModel(
        [
            action("read_benchmark"),
            action(
                "execute",
                command="test ! -f private.txt && printf isolated",
                purpose="test",
                timeout_seconds=1,
            ),
            action("finish", summary="Second workspace isolated."),
        ]
    )
    assert run_once(store, services, second_model, web)
    first_saved, second_saved = store.get(first["id"]), store.get(second["id"])
    assert first_saved["compute_job_id"] != second_saved["compute_job_id"]
    assert second_saved["state"]["experiments"][0]["output"] == "isolated"
    assert len(services.allocations) == 2
    assert all(key.startswith("harness-") for key in services.allocations)


@pytest.mark.parametrize(
    "role,name,args",
    [
        ("researcher", "submit_attempt", attempt_args()),
        ("attempt", "submit_paper", paper_args()),
        ("attempt", "review_novelty", {"idea": IDEA}),
    ],
)
def test_agent_role_cannot_invoke_other_roles_publication_or_review(harness, role, name, args):
    engine = owned_engine(harness, role)
    with pytest.raises(GateError, match="not available"):
        engine.tool(name, args)


@pytest.mark.parametrize(
    "role,reason",
    [
        ("researcher", "novelty"),
        ("attempt", "benchmark"),
    ],
)
def test_experiments_require_novelty_or_benchmark_read_first(harness, role, reason):
    engine = owned_engine(harness, role)
    with pytest.raises(GateError, match=reason):
        tool(
            engine,
            1,
            "execute",
            command="printf premature",
            purpose="experiment",
            timeout_seconds=1,
        )
    assert harness[1].sent == {}


def test_paper_cannot_be_published_before_recorded_experiment(harness):
    engine = owned_engine(harness)
    tool(engine, 1, "review_novelty", idea=IDEA)
    with pytest.raises(GateError, match="recorded sandbox executions"):
        tool(engine, 2, "submit_paper", **paper_args())
    assert not harness[1].published_papers


def test_attempt_cannot_be_published_before_a_recorded_test(harness):
    engine = owned_engine(harness, "attempt")
    tool(engine, 1, "read_benchmark")
    tool(engine, 2, "search_library", query="summation")
    tool(engine, 3, "read_paper", paper_id=PAPER_ID)
    with pytest.raises(GateError, match="recorded sandbox executions"):
        tool(engine, 4, "submit_attempt", **attempt_args())
    assert not harness[1].published_attempts


def test_rejected_idea_blocks_experiments_and_paper_submission(harness):
    model = FakeModel(novelty=[False])
    engine = owned_engine(harness, model=model)
    assessment = tool(engine, 1, "review_novelty", idea=IDEA)
    assert assessment["approved"] is False
    with pytest.raises(GateError, match="novelty"):
        tool(engine, 2, "execute", command=EXPERIMENT, purpose="experiment", timeout_seconds=2)
    with pytest.raises(GateError, match="novelty"):
        tool(engine, 3, "submit_paper", **paper_args())
    assert not harness[1].sent
    assert not harness[1].published_papers


def test_changed_final_paper_receives_fresh_novelty_review_and_can_be_rejected(harness):
    model = FakeModel(novelty=[True, False])
    engine = owned_engine(harness, model=model)
    tool(engine, 1, "review_novelty", idea=IDEA)
    tool(engine, 2, "execute", command=EXPERIMENT, purpose="experiment", timeout_seconds=2)
    harness[1].library.append(paper_record(OTHER_PAPER_ID, "Newly submitted related work."))
    changed = PAPER_TEXT + " The final claim has changed substantially."
    with pytest.raises(GateError, match="Final paper novelty review failed"):
        tool(engine, 3, "submit_paper", **paper_args(paper_text=changed))
    assert model.novelty_calls[1]["candidate"] == changed
    assert {item["id"] for item in model.novelty_calls[1]["library_papers"]} == {
        PAPER_ID,
        OTHER_PAPER_ID,
    }
    assert not model.evidence_calls
    assert not harness[1].published_papers


def test_changing_hypothesis_requires_new_experiment_evidence(harness):
    engine = owned_engine(harness)
    tool(engine, 1, "review_novelty", idea=IDEA)
    tool(engine, 2, "execute", command=EXPERIMENT, purpose="experiment", timeout_seconds=2)
    tool(engine, 3, "review_novelty", idea=IDEA + " Also test a different recurrence.")
    with pytest.raises(GateError, match="successful, nontruncated experiment"):
        tool(engine, 4, "submit_paper", **paper_args())
    assert not harness[1].published_papers


def test_unreadable_paper_prevents_incomplete_novelty_approval(harness):
    harness[1].library.append(paper_record(OTHER_PAPER_ID, None))
    model = FakeModel()
    engine = owned_engine(harness, model=model)
    with pytest.raises(GateError, match="no readable text"):
        tool(engine, 1, "review_novelty", idea=IDEA)
    assert not model.novelty_calls


@pytest.mark.parametrize(
    "change",
    [
        {"exit_code": 1},
        {"truncated": True},
        {"output": "   "},
        {"purpose": "setup"},
    ],
)
def test_failed_truncated_empty_or_setup_records_cannot_authorize_publication(harness, change):
    engine = owned_engine(harness, "attempt")
    engine.state["experiments"] = [
        {
            "id": "execute-4",
            "purpose": "test",
            "exit_code": 0,
            "output": "candidate correctness 55",
            "truncated": False,
            "step": 4,
            **change,
        }
    ]
    with pytest.raises(GateError, match="successful, nontruncated test"):
        engine.evidence(["execute-4"], "test")


def test_evidence_reviewer_can_reject_unsubstantiated_paper_even_with_successful_command(harness):
    model = FakeModel(supported=False)
    engine = owned_engine(harness, model=model)
    tool(engine, 1, "review_novelty", idea=IDEA)
    tool(
        engine,
        2,
        "execute",
        command="printf made-up-output",
        purpose="experiment",
        timeout_seconds=1,
    )
    with pytest.raises(GateError, match="Evidence reviewer did not approve"):
        tool(engine, 3, "submit_paper", **paper_args())
    assert not harness[1].published_papers


def test_invalid_benchmark_payload_and_unpinned_checkpoint_do_not_publish(harness):
    engine = owned_engine(harness, "attempt")
    tool(engine, 1, "read_benchmark")
    tool(engine, 2, "search_library", query="summation")
    tool(engine, 3, "read_paper", paper_id=PAPER_ID)
    tool(engine, 4, "execute", command=ATTEMPT_TEST, purpose="test", timeout_seconds=2)
    with pytest.raises(GateError, match="submission_schema"):
        tool(engine, 5, "submit_attempt", **attempt_args(payload={"answer": "55"}))
    with pytest.raises(GateError, match="full commit SHA"):
        tool(
            engine,
            6,
            "submit_attempt",
            **attempt_args(base_checkpoint_url="https://github.com/example/base/tree/main"),
        )
    assert not harness[1].published_attempts


def test_publication_intent_is_persisted_before_library_post(harness):
    store, services, web, _ = harness
    run = store.create(request())

    def check_intent():
        saved = store.get(run["id"])
        assert saved["state"]["publication"]["kind"] == "paper"
        assert saved["state"]["publication"]["status"] == "dispatching"
        assert saved["state"]["publication"]["body_sha256"]

    services.before_publish = check_intent
    assert run_once(store, services, FakeModel(researcher_actions()), web)
    assert store.get(run["id"])["status"] == "submitted"


def test_uncertain_publication_is_never_automatically_replayed(harness):
    store, services, web, clock = harness
    services.publication_error = ServiceError("unavailable", uncertain=True)
    run = store.create(request())
    model = FakeModel(researcher_actions())
    assert run_once(store, services, model, web)
    saved = store.get(run["id"])
    assert saved["status"] == "needs_attention"
    assert saved["error"] == "publication_outcome_unknown_do_not_resubmit"
    assert saved["state"]["publication"]["status"] == "dispatching"
    clock[0] += 1000
    assert run_once(store, services, model, web) is False
    assert len(services.published_papers) == 1


@pytest.mark.parametrize("publishing", [False, True])
def test_interrupted_run_reconciles_sandbox_without_restarting_agent(harness, publishing):
    store, services, web, clock = harness
    engine = owned_engine(harness)
    if publishing:
        engine.state["publication"] = {"kind": "paper", "status": "dispatching"}
        engine.save()
    clock[0] += 121
    model = FakeModel(researcher_actions())
    # A fresh store connection models worker restart against persisted run state.
    restarted = RunStore(store.path, now=lambda: clock[0])
    try:
        assert run_once(restarted, services, model, web)
    finally:
        restarted.close()
    saved = store.get(engine.id)
    assert saved["status"] == "needs_attention"
    assert saved["error"] == (
        "publication_outcome_unknown_do_not_resubmit"
        if publishing
        else "interrupted_run_not_replayed"
    )
    assert services.cancelled_jobs == [engine.job_id]
    assert saved["cleanup_required"] is False
    assert not model.model_calls
    assert not services.published_papers


def test_cancellation_during_agent_loop_cleans_up_without_publication(harness):
    store, services, web, _ = harness
    run = store.create(request("attempt"))

    class CancellingModel(FakeModel):
        def complete(self, messages, tools):
            store.cancel(run["id"])
            return super().complete(messages, tools)

    model = CancellingModel(attempt_actions())
    assert run_once(store, services, model, web)
    saved = store.get(run["id"])
    assert saved["status"] == "cancelled"
    assert saved["cleanup_required"] is False
    assert services.cancelled_jobs == [saved["compute_job_id"]]
    assert not services.published_attempts


def test_unconfirmed_cleanup_is_retained_and_only_cleanup_retries(harness):
    store, services, web, clock = harness
    services.cleanup_pending = True
    run = store.create(request("attempt"))
    model = FakeModel([action("finish", summary="No candidate supported.")])
    assert run_once(store, services, model, web)
    saved = store.get(run["id"])
    assert saved["status"] == "completed"
    assert saved["cleanup_required"] is True
    calls = len(model.model_calls)
    clock[0] += 6
    services.cleanup_pending = False
    assert run_once(store, services, model, web)
    assert store.get(run["id"])["cleanup_required"] is False
    assert len(model.model_calls) == calls
    assert len(services.allocations) == 1
    assert len(services.cancelled_jobs) == 2


def test_queued_cancellation_never_allocates_a_sandbox(harness):
    store, services, web, _ = harness
    run = store.create(request())
    store.cancel(run["id"])
    assert run_once(store, services, FakeModel(), web) is False
    assert store.get(run["id"])["status"] == "cancelled"
    assert not services.allocations


def test_second_publication_attempt_cannot_replay_persisted_intent(harness):
    engine = owned_engine(harness)
    engine.state["publication"] = {"kind": "paper", "status": "dispatching"}
    engine.save()
    with pytest.raises(EndRun) as raised:
        engine.publish("paper", {"title": "Do not dispatch again"})
    assert raised.value.status == "needs_attention"
    assert not harness[1].published_papers


def test_attempt_must_read_a_relevant_paper_when_search_has_matches(harness):
    engine = owned_engine(harness, "attempt")
    tool(engine, 1, "read_benchmark")
    tool(engine, 2, "search_library", query="summation")
    tool(engine, 4, "execute", command=ATTEMPT_TEST, purpose="test", timeout_seconds=2)
    with pytest.raises(GateError, match="read|Read"):
        tool(engine, 5, "submit_attempt", **attempt_args())
    assert not harness[1].published_attempts


@pytest.mark.parametrize(
    "payload,schema",
    [
        (55, {"type": "integer"}),
        ("55", {"type": "string"}),
        ([5, 5], {"type": "array", "items": {"type": "integer"}}),
        (None, {"type": "null"}),
    ],
)
def test_benchmark_schema_can_define_payloads_other_than_objects(harness, payload, schema):
    harness[1].benchmark_value["submission_schema"] = schema
    engine = owned_engine(harness, "attempt")
    tool(engine, 1, "read_benchmark")
    tool(engine, 2, "search_library", query="summation")
    tool(engine, 3, "read_paper", paper_id=PAPER_ID)
    tool(engine, 4, "execute", command=ATTEMPT_TEST, purpose="test", timeout_seconds=2)
    with pytest.raises(EndRun) as raised:
        tool(engine, 5, "submit_attempt", **attempt_args(payload=payload))
    assert raised.value.status == "submitted"
    assert harness[1].published_attempts[0]["payload"] == payload


def test_empty_library_does_not_require_reading_a_nonexistent_paper(harness):
    harness[1].library.clear()
    engine = owned_engine(harness, "attempt")
    tool(engine, 1, "read_benchmark")
    result = tool(engine, 2, "search_library", query="summation")
    assert result["library_total"] == 0
    tool(engine, 4, "execute", command=ATTEMPT_TEST, purpose="test", timeout_seconds=2)
    with pytest.raises(EndRun) as raised:
        tool(engine, 5, "submit_attempt", **attempt_args())
    assert raised.value.status == "submitted"


def test_cancellation_during_publication_preserves_uncertain_intent(harness):
    store, services, web, _ = harness
    run = store.create(request())
    services.before_publish = lambda: store.cancel(run["id"])
    assert run_once(store, services, FakeModel(researcher_actions()), web)
    saved = store.get(run["id"])
    assert len(services.published_papers) == 1
    assert saved["status"] == "needs_attention"
    assert saved["error"] == "publication_outcome_unknown_do_not_resubmit"
    assert saved["state"]["publication"]["status"] == "dispatching"
    assert saved["cleanup_required"] is False


def test_uncertain_compute_allocation_recovers_same_job_only_to_cancel(harness):
    store, services, web, _ = harness
    run = store.create(request())
    allocate = services.allocate
    calls = []

    def uncertain_allocate(allocation, key):
        job = allocate(allocation, key)
        calls.append(key)
        if len(calls) == 1:
            raise ServiceError("unavailable", uncertain=True)
        return job

    services.allocate = uncertain_allocate
    model = FakeModel(researcher_actions())
    assert run_once(store, services, model, web)
    saved = store.get(run["id"])
    assert saved["status"] == "failed"
    assert saved["error"] == "unavailable"
    assert saved["cleanup_required"] is False
    assert len(calls) == 2 and calls[0] == calls[1]
    assert len(services.jobs) == 1
    assert services.cancelled_jobs == [saved["compute_job_id"]]
    assert model.model_calls  # Online idea generation precedes the lazy allocation.


def test_library_change_during_final_evidence_review_prevents_publication(harness):
    engine = owned_engine(harness)
    _, services, _, _ = harness
    tool(engine, 1, "review_novelty", idea=IDEA)
    tool(engine, 2, "execute", command=EXPERIMENT, purpose="experiment", timeout_seconds=2)
    review = engine.model.complete_json

    def changed_library(system, value):
        result = review(system, value)
        if "experiment/test records" in system:
            services.library.append(paper_record(OTHER_PAPER_ID, text="A concurrent new paper."))
        return result

    engine.model.complete_json = changed_library
    with pytest.raises(GateError, match="Library changed"):
        tool(engine, 3, "submit_paper", **paper_args())
    assert services.published_papers == []


def test_cancellation_before_allocation_does_not_create_a_compute_job(harness):
    engine = owned_engine(harness, allocate=False)
    store, services, _, _ = harness
    store.cancel(engine.id)
    with pytest.raises(RunCancelled):
        engine.allocate()
    assert services.jobs == {}
    assert not store.get(engine.id)["allocation_started"]


def test_url_only_paper_does_not_count_as_read_for_attempt_publication(harness):
    engine = owned_engine(harness, role="attempt")
    _, services, _, _ = harness
    services.library[0]["submission"]["paper_text"] = None
    with pytest.raises(GateError, match="no readable"):
        tool(engine, 1, "read_paper", paper_id=PAPER_ID)
    assert engine.state.get("read_paper_ids", []) == []
