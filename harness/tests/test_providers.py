import json
import socket

import httpx
import pytest

from agent_harness import providers
from agent_harness.clients import ServiceError, WebResearch
from agent_harness.providers import CRUSOE_BASE_URL, check_provider, provider_models

MODEL = "nvidia/Example-Model"
KEY = "dummy-provider-secret"


def client(handler):
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer unrelated-secret", "Cookie": "unrelated-cookie"},
    )


def catalog(parameters=None):
    return {
        "data": [{"id": MODEL, "supported_parameters": parameters or ["tools", "response_format"]}]
    }


def tool_message(arguments='{"ok":true}'):
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "connection_check",
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            }
        ],
    }


def json_message(content='{"ok":true}'):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_native_you_search_uses_post_header_and_exposes_only_search_documents():
    seen = []

    def handler(request):
        seen.append(request)
        assert str(request.url) == "https://ydc-index.io/v1/search"
        assert request.method == "POST"
        assert request.headers["x-api-key"] == KEY
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert json.loads(request.content) == {"query": "new benchmark ideas", "count": 5}
        return httpx.Response(
            200,
            json={
                "request_id": KEY,
                "results": {
                    "web": [
                        {
                            "url": "https://papers.example/paper",
                            "title": "A result",
                            "description": "Unexpected echo: " + KEY,
                            "snippets": ["Evidence", KEY],
                            "raw_provider_metadata": KEY,
                        }
                    ]
                },
            },
        )

    result = WebResearch(search_key=KEY, provider="you", client=client(handler)).search(
        "new benchmark ideas"
    )
    assert result == {
        "results": {
            "web": [
                {
                    "url": "https://papers.example/paper",
                    "title": "A result",
                    "description": "Unexpected echo: [redacted]",
                    "snippets": ["Evidence", "[redacted]"],
                }
            ],
            "news": [],
        }
    }
    assert KEY not in json.dumps(result)
    assert len(seen) == 1


@pytest.mark.parametrize("url", ["http://localhost", "https://other.example/search"])
def test_you_credentials_cannot_be_routed_to_another_search_endpoint(url):
    with pytest.raises(ServiceError, match="invalid_configuration"):
        WebResearch(search_url=url, search_key=KEY, provider="you")


def test_you_search_requires_credentials():
    with pytest.raises(ServiceError, match="search_not_configured"):
        WebResearch(provider="you").search("paper")


def test_you_key_is_not_attached_to_public_paper_reads(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )

    def handler(request):
        assert "x-api-key" not in request.headers
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        return httpx.Response(200, text="public paper", headers={"Content-Type": "text/plain"})

    result = WebResearch(search_key=KEY, provider="you", client=client(handler)).read(
        "https://papers.example/paper"
    )
    assert result["text"] == "public paper"


@pytest.mark.parametrize(
    "response", [{}, {"results": {}}, {"results": []}, {"results": {"web": [{}]}}]
)
def test_malformed_you_results_fail_safely(response):
    result = check_provider("you", KEY, client=client(lambda _: httpx.Response(200, json=response)))
    assert result["ok"] is False
    assert KEY not in json.dumps(result)


def test_crusoe_catalog_is_fixed_origin_and_filters_provider_metadata():
    def handler(request):
        assert str(request.url) == CRUSOE_BASE_URL + "/models"
        assert request.headers["authorization"] == "Bearer " + KEY
        assert "cookie" not in request.headers
        return httpx.Response(
            200,
            json={
                "secret": KEY,
                "data": [
                    {
                        "id": MODEL,
                        "supported_parameters": ["tools", "response_format"],
                        "description": KEY,
                        "pricing": {"key": KEY},
                    }
                ],
            },
        )

    assert provider_models(KEY, client=client(handler)) == [
        {
            "id": MODEL,
            "supported_parameters": ["response_format", "tools"],
        }
    ]


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"id": KEY}],
        [{"id": "invalid model"}],
        [{"id": MODEL, "supported_parameters": [KEY]}],
        [{"id": MODEL}, {"id": MODEL}],
    ],
)
def test_catalog_does_not_expose_unvalidated_or_secret_metadata(data):
    with pytest.raises(ServiceError):
        provider_models(KEY, client=client(lambda _: httpx.Response(200, json={"data": data})))


def test_crusoe_connection_runs_both_bounded_compatibility_checks():
    seen = []

    def handler(request):
        seen.append(request)
        assert request.extensions["timeout"]["read"] == 8
        assert request.headers["authorization"] == "Bearer " + KEY
        assert KEY.encode() not in request.content
        if request.method == "GET":
            return httpx.Response(200, json=catalog())
        assert str(request.url) == CRUSOE_BASE_URL + "/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == MODEL
        assert body["max_tokens"] == 256
        if "tools" in body:
            return httpx.Response(200, json=tool_message())
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json=json_message())

    assert check_provider("crusoe", KEY, MODEL, client=client(handler))["ok"] is True
    assert len(seen) == 3


@pytest.mark.parametrize("condition", ["missing_model", "unsupported", "missing_selection"])
def test_invalid_model_selection_never_generates_paid_completions(condition):
    seen = []

    def handler(request):
        seen.append(request)
        assert request.method == "GET"
        return httpx.Response(
            200, json=catalog(["temperature"] if condition == "unsupported" else None)
        )

    selected = None if condition == "missing_selection" else MODEL
    if condition == "missing_model":
        selected = "absent/model"
    result = check_provider("crusoe", KEY, selected, client=client(handler))
    assert result["ok"] is False
    assert len(seen) == (0 if condition == "missing_selection" else 1)


@pytest.mark.parametrize(
    "mode,value",
    [
        ("tool", '{"ok":1}'),
        ("tool", '{"ok":true,"extra":1}'),
        ("tool", '{"ok":true,"ok":false}'),
        ("json", '```json\n{"ok":true}\n```'),
        ("json", '<think>reasoning</think>{"ok":true}'),
        ("json", '{"ok":1}'),
    ],
)
def test_compatibility_checks_require_actual_tool_calls_and_strict_json(mode, value):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=catalog())
        body = json.loads(request.content)
        if "tools" in body:
            return httpx.Response(
                200, json=tool_message(value if mode == "tool" else '{"ok":true}')
            )
        return httpx.Response(200, json=json_message(value))

    result = check_provider("crusoe", KEY, MODEL, client=client(handler))
    assert result["ok"] is False
    assert "check" in result["message"]


@pytest.mark.parametrize("provider", ["you", "crusoe"])
@pytest.mark.parametrize("failure", ["rejected", "redirect", "timeout", "body"])
def test_connection_failures_are_safe_and_never_follow_redirects(provider, failure):
    seen = []

    def handler(request):
        seen.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout(KEY, request=request)
        if failure == "redirect":
            return httpx.Response(302, headers={"Location": "https://other.example/" + KEY})
        if failure == "rejected":
            return httpx.Response(401, text=KEY)
        return httpx.Response(200, text=KEY)

    result = check_provider(provider, KEY, MODEL, client=client(handler))
    assert result["ok"] is False
    assert KEY not in json.dumps(result)
    assert len(seen) == 1


def test_you_connection_check_uses_an_explicit_small_search():
    seen = []

    def handler(request):
        seen.append(request)
        assert request.extensions["timeout"]["read"] == 8
        assert json.loads(request.content)["count"] == 5
        return httpx.Response(200, json={"results": {"web": [], "news": []}})

    result = check_provider("you", KEY, client=client(handler))
    assert result["ok"] is True
    assert len(seen) == 1


def test_shared_check_deadline_stops_before_another_paid_request(monkeypatch):
    now, seen = [100.0], []
    monkeypatch.setattr(providers.time, "monotonic", lambda: now[0])

    def handler(request):
        seen.append(request)
        now[0] += 26
        return httpx.Response(200, json=catalog())

    result = check_provider("crusoe", KEY, MODEL, client=client(handler))
    assert result["ok"] is False
    assert "timed out" in result["message"]
    assert len(seen) == 1
