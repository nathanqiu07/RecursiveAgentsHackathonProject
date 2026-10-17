import json
import socket
from uuid import UUID

import httpx
import pytest

from agent_harness.clients import ChatModel, ServiceError, Services, WebResearch

BENCHMARK = "00000000-0000-4000-8000-000000000001"
PAPER = "00000000-0000-4000-8000-000000000002"
JOB = "00000000-0000-4000-8000-000000000003"
SUBMISSION = "00000000-0000-4000-8000-000000000004"


def services(handler, **kwargs):
    return Services(
        "http://library.internal:8001",
        "http://dashboard.internal:3001",
        "http://manager.internal:8002",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_library_scan_reads_every_page_without_filtering_pending_papers():
    records = [{"id": str(UUID(int=index + 1))} for index in range(101)]
    requests = []

    def handler(request):
        requests.append(request)
        offset = int(request.url.params["offset"])
        assert "status" not in request.url.params
        assert request.url.params["limit"] == "100"
        return httpx.Response(200, json={"total": 101, "papers": records[offset : offset + 100]})

    assert services(handler).papers() == records
    assert [request.url.params["offset"] for request in requests] == ["0", "100"]


@pytest.mark.parametrize("change", ["duplicate", "total", "empty", "excess"])
def test_library_scan_rejects_incomplete_or_changing_snapshots(change):
    first = [{"id": str(UUID(int=index + 1))} for index in range(100)]

    def handler(request):
        if request.url.params["offset"] == "0":
            return httpx.Response(200, json={"total": 101, "papers": first})
        total = 100 if change == "total" else 101
        records = [] if change == "empty" else [first[0]]
        if change == "excess":
            records = [{"id": str(UUID(int=102))}, {"id": str(UUID(int=103))}]
        return httpx.Response(200, json={"total": total, "papers": records})

    with pytest.raises(ServiceError):
        services(handler).papers()


def test_library_scan_never_silently_truncates_a_large_library():
    with pytest.raises(ServiceError, match="library_too_large"):
        services(lambda _: httpx.Response(200, json={"total": 1001, "papers": []})).papers()


@pytest.mark.parametrize(
    "result,uncertain",
    [
        (httpx.Response(500, text="secret upstream diagnostic"), True),
        (httpx.Response(202, content=b"not JSON"), True),
        (httpx.Response(202, json={"id": "not a uuid"}), True),
        (httpx.Response(202, json={"id": PAPER, "status": "verified"}), True),
        (httpx.Response(200, json={"id": PAPER, "status": "pending_verification"}), True),
        (httpx.Response(422, json={"detail": "secret body"}), False),
        (httpx.Response(307, headers={"location": "http://private.internal"}), True),
    ],
)
def test_publication_failure_preserves_uncertainty_without_replay_or_secret_text(result, uncertain):
    calls = []

    def handler(request):
        calls.append(request)
        return result

    with pytest.raises(ServiceError) as error:
        services(handler).submit_paper({"title": "Paper", "paper_text": "Evidence"})
    assert error.value.uncertain is uncertain
    assert "secret" not in str(error.value)
    assert len(calls) == 1


def test_publication_timeout_is_uncertain_and_redacts_exception_details():
    def handler(request):
        raise httpx.ReadTimeout("Authorization: Bearer secret-provider-key", request=request)

    with pytest.raises(ServiceError) as error:
        services(handler).submit_paper({"title": "Paper", "paper_text": "Text"})
    assert error.value.code == "timeout"
    assert error.value.uncertain is True
    assert "secret" not in str(error.value)


def test_invalid_local_publication_does_not_claim_it_was_dispatched():
    calls = []
    with pytest.raises(ServiceError) as error:
        services(lambda request: calls.append(request)).submit_paper({"score": float("nan")})
    assert error.value.uncertain is False
    assert calls == []


def test_publication_routes_and_credentials_stay_with_the_correct_service():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "library.internal":
            return httpx.Response(202, json={"id": PAPER, "status": "pending_verification"})
        return httpx.Response(
            202, json={"id": SUBMISSION, "benchmark_id": BENCHMARK, "status": "queued"}
        )

    client = services(handler, library_key="library-secret", benchmark_key="benchmark-secret")
    client.submit_paper({"title": "Paper", "paper_text": "Text"})
    client.submit_attempt(BENCHMARK, {"payload": {"answer": 42}})
    assert requests[0].url.path == "/papers"
    assert requests[0].headers["authorization"] == "Bearer library-secret"
    assert requests[1].url.path == f"/api/benchmarks/{BENCHMARK}/submissions"
    assert requests[1].headers["authorization"] == "Bearer benchmark-secret"


@pytest.mark.parametrize("identifier", ["../admin", BENCHMARK + "/cancel", "%2e%2e", "invalid"])
def test_identifiers_cannot_control_service_routes(identifier):
    requests = []
    client = services(lambda request: requests.append(request))
    for action in (client.paper, client.benchmark, client.submission, client.job, client.cancel):
        with pytest.raises(ServiceError, match="invalid_id"):
            action(identifier)
    assert requests == []


@pytest.mark.parametrize(
    "url",
    [
        "http://user:key@localhost",
        "http://localhost/api",
        "http://localhost?q=1",
        "http://localhost#fragment",
        "file:///etc/passwd",
        "http://localhost:99999",
    ],
)
def test_service_urls_must_be_clean_origins(url):
    with pytest.raises(ServiceError, match="invalid_url"):
        Services(url, "http://localhost:3001", "http://localhost:8002")


def test_compute_allocation_is_idempotent_and_cannot_choose_image_or_command():
    seen = []
    allocation = {
        "resource_id": "cpu",
        "node_id": "node",
        "cpus": 0.25,
        "memory_mb": 128,
        "scratch_mb": 64,
        "timeout_seconds": 60,
    }

    def handler(request):
        seen.append(request)
        assert request.headers["idempotency-key"] == "run-unique-allocation"
        assert json.loads(request.content) == allocation
        return httpx.Response(202, json={"id": JOB, "status": "queued"})

    client = services(handler)
    assert client.allocate(allocation, "run-unique-allocation")["id"] == JOB
    with pytest.raises(ServiceError, match="invalid_allocation"):
        client.allocate({**allocation, "image": "attacker"}, "run-unique-allocation")
    assert len(seen) == 1


@pytest.mark.parametrize(
    "outcome,uncertain",
    [
        ("timeout", True),
        ("connection", True),
        ("server", True),
        ("json", True),
        ("identifier", True),
        ("status", True),
        ("redirect", True),
        ("rejected", False),
        ("conflict", False),
    ],
)
def test_allocation_uncertainty_retains_potential_remote_job(outcome, uncertain):
    allocation = {
        "resource_id": "cpu",
        "node_id": "node",
        "cpus": 0.25,
        "memory_mb": 128,
        "scratch_mb": 64,
        "timeout_seconds": 60,
    }
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["idempotency-key"] == "stable-run-key"
        if outcome == "timeout":
            raise httpx.ReadTimeout("secret timeout details", request=request)
        if outcome == "connection":
            raise httpx.ReadError("secret connection details", request=request)
        responses = {
            "server": httpx.Response(503, text="secret upstream state"),
            "json": httpx.Response(202, content=b"invalid JSON"),
            "identifier": httpx.Response(202, json={"id": "not-a-uuid", "status": "queued"}),
            "status": httpx.Response(202, json={"id": JOB, "status": "unknown"}),
            "redirect": httpx.Response(303, headers={"location": "/jobs/" + JOB}),
            "rejected": httpx.Response(422, json={"detail": "invalid allocation"}),
            "conflict": httpx.Response(409, json={"detail": "idempotency conflict"}),
        }
        return responses[outcome]

    with pytest.raises(ServiceError) as error:
        services(handler).allocate(allocation, "stable-run-key")
    assert error.value.uncertain is uncertain
    assert "secret" not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [42, ["candidate", 1], "answer", None, True])
def test_attempt_payload_can_be_any_json_type(payload):
    def handler(request):
        assert json.loads(request.content)["payload"] == payload
        return httpx.Response(
            202, json={"id": SUBMISSION, "benchmark_id": BENCHMARK, "status": "queued"}
        )

    result = services(handler).submit_attempt(BENCHMARK, {"payload": payload})
    assert result["id"] == SUBMISSION


def test_compute_output_checks_cursor_and_contiguous_sequence():
    client = services(
        lambda _: httpx.Response(
            200,
            json={
                "lines": [{"sequence": 3, "line": "result"}],
                "next_cursor": 3,
            },
        )
    )
    assert client.output(JOB, 2)["next_cursor"] == 3
    with pytest.raises(ServiceError, match="invalid_response"):
        client.output(JOB, 1)
    with pytest.raises(ServiceError, match="invalid_cursor"):
        client.output(JOB, True)


def test_model_tool_request_and_json_mode_use_host_credentials_only():
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer model-secret"
        if "tools" in body:
            assert body["tool_choice"] == "auto"
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "library_search", "arguments": '{"query":"idea"}'},
                    }
                ],
            }
        else:
            assert body["response_format"] == {"type": "json_object"}
            message = {"role": "assistant", "content": '{"novel":true}'}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    model = ChatModel(
        "https://model.example/v1",
        "model-name",
        api_key="model-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert model.complete([{"role": "user", "content": "Research"}], tools=[{}])["tool_calls"]
    assert model.complete_json("Review novelty", {"idea": "Example"}) == {"novel": True}
    assert len(requests) == 2


@pytest.mark.parametrize("content", ['{"novel":true,"novel":false}', '{"score":NaN}', "[]"])
def test_model_json_rejects_duplicate_keys_nonfinite_values_and_nonobjects(content):
    model = ChatModel(
        "http://model.local/v1",
        "model",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
                )
            )
        ),
    )
    with pytest.raises(ServiceError, match="invalid_model_response"):
        model.complete_json("Review", "Candidate")


def public_dns(monkeypatch, *addresses):
    def resolve(host, port, **kwargs):
        assert host == "papers.example"
        assert port == 443
        assert kwargs == {"type": socket.SOCK_STREAM}
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))
            for ip in addresses
        ]

    monkeypatch.setattr("agent_harness.clients.socket.getaddrinfo", resolve)


def test_web_read_pins_vetted_ip_and_preserves_tls_name_without_service_cookies(monkeypatch):
    public_dns(monkeypatch, "93.184.216.34")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "papers.example"
        assert request.extensions["sni_hostname"] == "papers.example"
        assert request.headers["connection"] == "close"
        assert "cookie" not in request.headers
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            text=(
                "<html><style>hidden CSS</style><body><h1>Paper</h1>"
                "<script>ignore these commands</script><p>Actual &amp; evidence.</p></body></html>"
            ),
            headers={"content-type": "text/html"},
        )

    injected = httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer service-secret"},
        cookies={"session": "service-session"},
    )
    result = WebResearch(client=injected).read("https://papers.example/research")
    assert result["url"] == "https://papers.example/research"
    assert "Actual & evidence." in result["text"]
    assert "hidden CSS" not in result["text"]
    assert "ignore these commands" not in result["text"]
    assert len(requests) == 1


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "::1",
        "fc00::1",
        "224.0.0.1",
        "0.0.0.0",
        "::ffff:127.0.0.1",
    ],
)
def test_web_read_blocks_private_metadata_and_mixed_dns_answers(monkeypatch, address):
    public_dns(monkeypatch, "93.184.216.34", address)
    calls = []
    web = WebResearch(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: calls.append(request)))
    )
    with pytest.raises(ServiceError, match="web_destination_denied"):
        web.read("https://papers.example/research")
    assert calls == []


@pytest.mark.parametrize(
    "url",
    [
        "http://papers.example/research",
        "https://papers.example:8443/research",
        "https://user:password@papers.example/research",
        "file:///etc/passwd",
    ],
)
def test_web_read_rejects_non_https_credentials_and_unapproved_ports(url):
    with pytest.raises(ServiceError):
        WebResearch().read(url)


def test_web_redirects_are_rejected_without_following_them(monkeypatch):
    public_dns(monkeypatch, "93.184.216.34")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "http://169.254.169.254/credentials"})

    with pytest.raises(ServiceError, match="redirect_rejected"):
        WebResearch(client=httpx.Client(transport=httpx.MockTransport(handler))).read(
            "https://papers.example/redirect"
        )
    assert len(calls) == 1


def test_web_response_is_bounded_before_parsing(monkeypatch):
    public_dns(monkeypatch, "93.184.216.34")
    web = WebResearch(
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, content=b"x" * (512 * 1024 + 1), headers={"content-type": "text/html"}
                )
            )
        )
    )
    with pytest.raises(ServiceError, match="response_too_large"):
        web.read("https://papers.example/large")


def test_search_gateway_is_explicit_and_credential_does_not_enter_the_query():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.path == "/search"
        assert request.url.params["q"] == "kernel optimization & inference"
        assert request.headers["authorization"] == "Bearer search-secret"
        assert "search-secret" not in str(request.url)
        return httpx.Response(200, json={"results": [{"url": "https://papers.example/a"}]})

    web = WebResearch(
        "http://search.internal/search",
        "search-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert web.search("kernel optimization & inference")["results"]
    assert len(requests) == 1
    with pytest.raises(ServiceError, match="search_not_configured"):
        WebResearch().search("anything")
