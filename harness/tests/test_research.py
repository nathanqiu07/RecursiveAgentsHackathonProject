"""Real MCP protocol roundtrips via MockTransport and sourced/lazy experiment gates."""

import hashlib
import json

import httpx
import pytest
from test_engine import (
    IDEA,
    ONLINE_PAPER,
    PROPOSAL,
    FakeModel,
    action,
    owned_engine,
    request,
)
from test_engine import harness as harness

from agent_harness.clients import PAPER_DOMAINS, ServiceError, WebResearch, paper_source_url
from agent_harness.engine import GateError, run_once
from agent_harness.you_mcp import FREE_URL, KEYED_URL, YouMCP


def mcp_server(seen, key=None, *, failed=False, returned=None):
    def handler(http_request):
        seen.append(http_request)
        assert str(http_request.url) == (KEYED_URL if key else FREE_URL)
        assert http_request.headers.get("authorization") == ("Bearer " + key if key else None)
        assert "cookie" not in http_request.headers
        if http_request.method == "GET":
            return httpx.Response(405)
        if http_request.method == "DELETE":
            return httpx.Response(204)
        body = json.loads(http_request.content)
        method = body["method"]
        if method.startswith("notifications/"):
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "You.com protocol fixture", "version": "1"},
            }
        elif method == "tools/list":
            names = ["you-search", "you-contents"] if key else ["you-search", "you-discover"]
            result = {
                "tools": [
                    {"name": name, "inputSchema": {"type": "object", "additionalProperties": True}}
                    for name in names
                ]
            }
        else:
            assert method == "tools/call"
            assert body["params"]["name"] == "you-search"
            assert body["params"]["_meta"]["com.you.com/search"]["include_domains"] == list(
                PAPER_DOMAINS
            )
            value = returned or {
                "results": {"web": [{"url": ONLINE_PAPER, "title": "Actual mock paper"}]}
            }
            result = {
                "isError": failed,
                "content": [{"type": "text", "text": json.dumps(value)}],
                "structuredContent": value,
            }
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("key", [None, "dummy-you-mcp-key"])
def test_mcp_initialize_list_call_and_scope_are_actual_protocol(key):
    seen = []
    client = YouMCP(key, transport=mcp_server(seen, key))
    result = client.search("quantized inference papers", PAPER_DOMAINS)
    assert result["results"]["web"][0]["url"] == ONLINE_PAPER
    methods = [json.loads(item.content)["method"] for item in seen if item.method == "POST"]
    assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    assert all(
        (key or "not-a-real-key") not in item.content.decode() for item in seen if item.content
    )


def test_free_profile_discovery_does_not_claim_contents_and_disallows_extra_tools():
    seen = []
    client = YouMCP(transport=mcp_server(seen))
    assert client.capabilities() == {
        "transport": "streamable_http",
        "provider": "you_mcp",
        "profile": "free",
        "tools": ["you-search"],
        "contents_available": False,
    }
    with pytest.raises(ServiceError, match="you_mcp_tool_unavailable"):
        client.contents(ONLINE_PAPER)
    assert not any(
        json.loads(item.content).get("method") == "tools/call" for item in seen if item.content
    )


def test_mcp_errors_and_echoes_do_not_expose_keys():
    key, seen = "private-test-you-key", []
    client = YouMCP(
        key,
        transport=mcp_server(
            seen, key, returned={"results": {"web": [{"url": ONLINE_PAPER, "title": key}]}}
        ),
    )
    assert key not in json.dumps(client.search("inference", PAPER_DOMAINS))
    client = YouMCP(key, transport=mcp_server([], key, failed=True))
    with pytest.raises(ServiceError, match="you_mcp_tool_failed"):
        client.search("inference", PAPER_DOMAINS)


@pytest.mark.parametrize(
    "url",
    ["https://arxiv.org/pdf/2401.12345.pdf", "https://arxiv.org/html/2401.12345", ONLINE_PAPER],
)
def test_arxiv_paper_urls_have_one_identity(url):
    assert paper_source_url(url) == ONLINE_PAPER


@pytest.mark.parametrize(
    "url",
    [
        "https://blog.example/paper",
        "http://arxiv.org/abs/2401.12345",
        "https://arxiv.org/search",
        "https://user:password@arxiv.org/abs/2401.12345",
    ],
)
def test_research_rejects_nonprimary_or_unsafe_sources(url):
    with pytest.raises(ServiceError):
        paper_source_url(url)


def test_paper_search_uses_mcp_and_filters_blog_or_fabricated_urls(monkeypatch):
    def search(_self, query, domains):
        assert domains == PAPER_DOMAINS
        return {
            "results": {"web": [{"url": ONLINE_PAPER}, {"url": "https://blog.example/not-primary"}]}
        }

    monkeypatch.setattr(YouMCP, "search", search)
    result = WebResearch().search_papers("online papers")
    assert result["provider"] == "you_mcp"
    assert [paper["url"] for paper in result["papers"]] == [ONLINE_PAPER]


def test_keyless_paper_read_prefers_html_and_explicitly_labels_abstract(monkeypatch):
    calls = []

    def read(_self, url):
        calls.append(url)
        if "/html/" in url:
            raise ServiceError("upstream_rejected")
        return {
            "url": url,
            "text": "Explicitly retrieved abstract fixture. " * 20,
            "truncated": False,
        }

    monkeypatch.setattr(WebResearch, "read", read)
    result = WebResearch().read_paper(ONLINE_PAPER)
    assert calls == [ONLINE_PAPER.replace("/abs/", "/html/"), ONLINE_PAPER]
    assert result["content_kind"] == "abstract_page"
    assert result["text_sha256"] == hashlib.sha256(result["text"].encode()).hexdigest()


def test_idea_generation_needs_no_gpu_and_keeps_sources_and_plan(harness):
    store, services, web, _ = harness
    run = store.create(request(allocation=None))
    model = FakeModel(
        [
            action("search_research_papers", query="real paper discovery"),
            action("read_research_paper", url=ONLINE_PAPER),
            action("propose_research_idea", **PROPOSAL),
            action(
                "finish",
                summary=(
                    "A sourced hypothesis and controlled experiment plan, not measured results."
                ),
            ),
        ]
    )
    assert run_once(store, services, model, web)
    saved = store.get(run["id"])
    assert saved["status"] == "completed"
    assert services.jobs == {}
    assert saved["result"]["research_ideas"][0]["hypothesis"] == IDEA
    assert saved["result"]["sources"][0]["url"] == ONLINE_PAPER
    assert saved["result"]["published"] is False


def test_resource_unavailable_preserves_sourced_idea_without_invented_experiments(harness):
    store, services, web, _ = harness
    services.resources = lambda: []
    run = store.create(request(allocation=None))
    model = FakeModel(
        [
            action("search_research_papers", query="papers"),
            action("read_research_paper", url=ONLINE_PAPER),
            action("propose_research_idea", **PROPOSAL),
            action(
                "execute",
                command="printf should-never-run",
                purpose="experiment",
                timeout_seconds=2,
            ),
        ]
    )
    assert run_once(store, services, model, web)
    saved = store.get(run["id"])
    assert saved["status"] == "needs_attention"
    assert saved["error"] == "research_gpu_capacity_unavailable"
    assert saved["result"]["research_ideas"]
    assert not saved["state"]["experiments"]
    assert not services.jobs


def test_gpu_allocation_is_deferred_until_first_real_experiment(harness):
    store, services, web, _ = harness
    run = store.create(request(allocation=None))
    model = FakeModel(
        [
            action("search_research_papers", query="papers"),
            action("read_research_paper", url=ONLINE_PAPER),
            action("propose_research_idea", **PROPOSAL),
            action(
                "execute",
                command="python3 -c 'assert sum(range(11)) == 55; print(55)'",
                purpose="experiment",
                timeout_seconds=2,
            ),
            action("finish", summary="Measured correctness; no GPU throughput conclusion."),
        ]
    )
    assert run_once(store, services, model, web)
    saved = store.get(run["id"])
    kinds = [event["kind"] for event in store.events(run["id"])]
    assert (
        kinds.index("research_idea")
        < kinds.index("sandbox_allocated")
        < kinds.index("experiment_result")
    )
    assert saved["state"]["selected_allocation"]["resource_id"] == "test-cpu"
    assert len(saved["state"]["experiments"]) == 1


def test_snippets_and_unread_sources_cannot_authorize_a_proposal(harness):
    engine = owned_engine(harness, allocate=False, online=False)
    engine.tool("search_research_papers", {"query": "papers"})
    with pytest.raises(GateError, match="actual online paper text"):
        engine.tool("propose_research_idea", PROPOSAL)
    with pytest.raises(GateError, match="Search and read online papers"):
        engine.tool("review_novelty", {"idea": IDEA})
    assert engine.job_id is None


def test_unknown_paper_citations_are_rejected_even_with_read_real_source(harness):
    engine = owned_engine(harness, allocate=False)
    engine.tool("propose_research_idea", PROPOSAL)
    with pytest.raises(GateError, match="actually read"):
        engine.validate_citations(
            "Citations: " + ONLINE_PAPER + " https://arxiv.org/abs/2501.99999"
        )
    with pytest.raises(GateError, match="Cite the recorded"):
        engine.validate_citations("An uncited unsupported paper.")


def test_assigned_benchmark_must_be_read_before_a_sourced_proposal(harness):
    from test_engine import BENCHMARK_ID

    from agent_harness.engine import Engine

    store, services, web, _ = harness
    store.create(request(benchmark_id=BENCHMARK_ID, allocation=None))
    run, token, _ = store.claim()
    engine = Engine(store, run, token, services, FakeModel(), web)
    engine.state.update(steps=0, experiments=[])
    engine.tool("search_research_papers", {"query": "papers"})
    engine.tool("read_research_paper", {"url": ONLINE_PAPER})
    with pytest.raises(GateError, match="Read the assigned benchmark"):
        engine.tool("propose_research_idea", PROPOSAL)
    engine.tool("read_benchmark", {})
    assert engine.tool("propose_research_idea", PROPOSAL)["proposal"]["hypothesis"] == IDEA
    assert not services.jobs
