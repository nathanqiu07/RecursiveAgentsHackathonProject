"""Credential-owning, bounded clients for the trusted harness control plane.

Agents receive tool results, never these clients or their credentials. Public web
reads use a numeric address vetted before connection and the original TLS name;
DNS cannot redirect a later connection to the service network.
"""

import hashlib
import ipaddress
import json
import re
import socket
import time
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import UUID

import httpx

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_WEB_BYTES = 512 * 1024
YOU_SEARCH_URL = "https://ydc-index.io/v1/search"
PAPER_DOMAINS = (
    "arxiv.org",
    "openreview.net",
    "proceedings.mlr.press",
    "aclanthology.org",
    "papers.nips.cc",
    "papers.neurips.cc",
    "proceedings.neurips.cc",
    "openaccess.thecvf.com",
    "dl.acm.org",
    "ieeexplore.ieee.org",
)
_ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\Z")


def paper_source_url(value):
    """Canonical primary-paper URLs; no model-selected arbitrary destinations."""
    parsed = urlsplit(str(_url(value, https=True)))
    if parsed.port not in {None, 443}:
        raise ServiceError("paper_source_not_primary")
    hostname = (parsed.hostname or "").lower()
    hostname = hostname.removeprefix("www.")
    if hostname == "export.arxiv.org":
        hostname = "arxiv.org"
    if hostname not in PAPER_DOMAINS or not parsed.path.strip("/"):
        raise ServiceError("paper_source_not_primary")
    if hostname == "arxiv.org":
        match = re.fullmatch(r"/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?", parsed.path)
        if not match or not _ARXIV_ID.fullmatch(match[1]):
            raise ServiceError("paper_source_not_primary")
        return "https://arxiv.org/abs/" + match[1]
    if hostname == "openreview.net":
        identifier = parse_qs(parsed.query).get("id", [])
        if (
            parsed.path not in {"/forum", "/pdf"}
            or len(identifier) != 1
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", identifier[0])
        ):
            raise ServiceError("paper_source_not_primary")
        return "https://openreview.net/forum?" + urlencode({"id": identifier[0]})
    # Strip tracking queries; preserve a concrete publication path.
    return "https://" + hostname + parsed.path


MAX_PAPERS = 1000
TIMEOUT = httpx.Timeout(30, connect=5, write=10, pool=5)
JOB_STATUSES = {"queued", "provisioning", "running", "completed", "failed", "cancelled"}


class ServiceError(Exception):
    """Safe error metadata, with uncertainty for non-idempotent publication."""

    def __init__(self, code: str, uncertain: bool = False):
        self.code = code
        self.uncertain = uncertain
        super().__init__(f"Service request failed ({code})")


def _url(value, *, origin=False, https=False):
    try:
        parsed = urlsplit(value)
        url = httpx.URL(value)
        _ = parsed.port  # Validate malformed and out-of-range explicit ports.
        valid = (
            isinstance(value, str)
            and len(value) <= 8192
            and parsed.scheme in ({"https"} if https else {"http", "https"})
            and bool(parsed.hostname)
            and "@" not in parsed.netloc
            and "\\" not in value
            and all(ord(character) > 32 and ord(character) != 127 for character in value)
            and not parsed.fragment
            and "#" not in value
            and (not origin or (parsed.path in {"", "/"} and "?" not in value))
        )
        if not valid:
            raise ValueError
        return url
    except (TypeError, ValueError, AttributeError, httpx.InvalidURL):
        raise ServiceError("invalid_url") from None


def _key(value):
    if value is not None and (
        not isinstance(value, str) or any(not 32 <= ord(char) <= 126 for char in value)
    ):
        raise ServiceError("invalid_configuration")
    return value


def _id(value):
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError
        return value
    except (ValueError, TypeError, AttributeError):
        raise ServiceError("invalid_id") from None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("Nonfinite JSON")


def _json(value):
    return json.loads(value, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _body(response, maximum, deadline):
    if response.headers.get("content-encoding", "identity").lower() != "identity":
        raise ServiceError("unsupported_encoding")
    if response.is_stream_consumed:
        chunks = [response.content]
    else:
        chunks = response.iter_raw(chunk_size=65536)
    content = bytearray()
    chunks = iter(chunks)
    while True:
        if time.monotonic() >= deadline:
            raise ServiceError("timeout")
        try:
            chunk = next(chunks)
        except StopIteration:
            break
        if time.monotonic() >= deadline:
            raise ServiceError("timeout")
        if len(content) + len(chunk) > maximum:
            raise ServiceError("response_too_large")
        content.extend(chunk)
    return bytes(content)


class _HTTP:
    def __init__(self, client=None, *, public=False):
        self.client = (
            client
            if client is not None
            else httpx.Client(
                trust_env=False,
                follow_redirects=False,
                timeout=TIMEOUT,
                # Pinned web requests must not reuse a TLS session established for a
                # different original hostname sharing the same numeric address.
                limits=httpx.Limits(max_keepalive_connections=0) if public else httpx.Limits(),
            )
        )
        self.owns_client = client is None

    def close(self):
        if self.owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def request(
        self,
        method,
        url,
        *,
        body=None,
        key=None,
        headers=None,
        publication=False,
        expected_status=None,
        maximum=MAX_JSON_BYTES,
        extensions=None,
        text=False,
        deadline=None,
    ):
        request_headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if key:
            request_headers["Authorization"] = f"Bearer {key}"
        if headers:
            request_headers.update(headers)
        content = None
        if body is not None:
            try:
                content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (TypeError, ValueError, UnicodeError, RecursionError):
                raise ServiceError("invalid_request") from None
            if len(content) > MAX_REQUEST_BYTES:
                raise ServiceError("request_too_large")
            request_headers["Content-Type"] = "application/json"
        explicit_deadline = deadline is not None
        deadline = (
            min(time.monotonic() + 30, deadline) if deadline is not None else time.monotonic() + 30
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ServiceError("timeout")
        request_extensions = {"timeout": TIMEOUT.as_dict(), **(extensions or {})}
        if explicit_deadline:
            # Keep connection, TLS, write, and pool waits inside a shared
            # operation budget. The final read may overrun by its socket timeout;
            # _body checks the deadline before beginning every further read.
            request_extensions["timeout"] = {
                name: min(value, remaining if name == "read" else remaining / 5)
                for name, value in request_extensions["timeout"].items()
            }
        request = httpx.Request(
            method, url, headers=request_headers, content=content, extensions=request_extensions
        )
        dispatched = False
        try:
            # Build Request explicitly: a caller-supplied client must not merge
            # its default cookies, authorization, or service headers into a tool.
            dispatched = True
            response = self.client.send(request, stream=True, auth=None, follow_redirects=False)
            try:
                if not 200 <= response.status_code < 300:
                    if 300 <= response.status_code < 400:
                        code = "redirect_rejected"
                    elif response.status_code == 409:
                        code = "conflict"
                    else:
                        code = "upstream_rejected"
                    # An unexpected redirect may follow a successful mutation;
                    # only explicit 4xx rejections establish non-acceptance.
                    raise ServiceError(code, publication and not 400 <= response.status_code < 500)
                if expected_status is not None and response.status_code != expected_status:
                    raise ServiceError("invalid_acknowledgement", publication)
                raw = _body(response, maximum, deadline)
                if text:
                    return raw, response.headers.get("content-type", "")
                result = _json(raw)
                if not isinstance(result, dict):
                    raise ValueError
                return result
            finally:
                response.close()
        except ServiceError as exc:
            if (
                publication
                and dispatched
                and exc.code in {"timeout", "response_too_large", "unsupported_encoding"}
            ):
                raise ServiceError(exc.code, True) from None
            raise
        except httpx.TimeoutException:
            raise ServiceError("timeout", publication and dispatched) from None
        except (httpx.HTTPError, OSError):
            raise ServiceError("unavailable", publication and dispatched) from None
        except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError):
            raise ServiceError("invalid_response", publication and dispatched) from None


class Services(_HTTP):
    """Only fixed Library, dashboard, and generic compute API routes."""

    def __init__(
        self,
        library_url,
        benchmark_url,
        manager_url,
        library_key=None,
        benchmark_key=None,
        manager_key=None,
        client=None,
    ):
        self.library_url = str(_url(library_url, origin=True)).rstrip("/")
        self.benchmark_url = str(_url(benchmark_url, origin=True)).rstrip("/")
        self.manager_url = str(_url(manager_url, origin=True)).rstrip("/")
        self.library_key, self.benchmark_key, self.manager_key = (
            _key(library_key),
            _key(benchmark_key),
            _key(manager_key),
        )
        super().__init__(client)

    def _library(self, method, path, **kwargs):
        return self.request(method, self.library_url + path, key=self.library_key, **kwargs)

    def _benchmark(self, method, path, **kwargs):
        return self.request(
            method, self.benchmark_url + "/api" + path, key=self.benchmark_key, **kwargs
        )

    def _manager(self, method, path, **kwargs):
        return self.request(method, self.manager_url + path, key=self.manager_key, **kwargs)

    @staticmethod
    def _record(value, expected_id=None, *, publication=False, job=False):
        try:
            record_id = _id(value.get("id"))
            if expected_id is not None and record_id != expected_id:
                raise ValueError
            if job and value.get("status") not in JOB_STATUSES:
                raise ValueError
            return value
        except (ServiceError, ValueError, TypeError, AttributeError):
            raise ServiceError(
                "invalid_acknowledgement" if publication else "invalid_response", publication
            ) from None

    def papers(self):
        papers, seen, expected_total = [], set(), None
        while True:
            page = self._library("GET", f"/papers?limit=100&offset={len(papers)}")
            total, records = page.get("total"), page.get("papers")
            if type(total) is not int or total < 0 or not isinstance(records, list):
                raise ServiceError("invalid_response")
            if total > MAX_PAPERS:
                raise ServiceError("library_too_large")
            if expected_total is not None and total != expected_total:
                raise ServiceError("library_changed")
            expected_total = total
            if len(records) > 100 or (not records and len(papers) != total):
                raise ServiceError("invalid_response")
            for record in records:
                record = self._record(record)
                if record["id"] in seen:
                    raise ServiceError("library_changed")
                seen.add(record["id"])
                papers.append(record)
            if len(papers) > total:
                raise ServiceError("invalid_response")
            if len(papers) == total:
                return papers

    def paper(self, paper_id):
        paper_id = _id(paper_id)
        return self._record(self._library("GET", "/papers/" + paper_id), paper_id)

    def benchmark(self, benchmark_id):
        benchmark_id = _id(benchmark_id)
        return self._record(self._benchmark("GET", "/benchmarks/" + benchmark_id), benchmark_id)

    def submit_paper(self, body):
        result = self._library("POST", "/papers", body=body, publication=True, expected_status=202)
        result = self._record(result, publication=True)
        if result.get("status") != "pending_verification":
            raise ServiceError("invalid_acknowledgement", True)
        return result

    def submit_attempt(self, benchmark_id, body):
        benchmark_id = _id(benchmark_id)
        result = self._benchmark(
            "POST",
            "/benchmarks/" + benchmark_id + "/submissions",
            body=body,
            publication=True,
            expected_status=202,
        )
        result = self._record(result, publication=True)
        if result.get("benchmark_id") != benchmark_id or result.get("status") not in {
            "queued",
            "running",
            "verified",
            "failed",
            "rejected",
        }:
            raise ServiceError("invalid_acknowledgement", True)
        return result

    def submission(self, submission_id):
        submission_id = _id(submission_id)
        return self._record(self._benchmark("GET", "/submissions/" + submission_id), submission_id)

    def resources(self):
        result = self._manager("GET", "/resources").get("resources")
        if (
            not isinstance(result, list)
            or len(result) > 100
            or any(not isinstance(item, dict) for item in result)
        ):
            raise ServiceError("invalid_response")
        return result

    def resource_status(self):
        result = self._manager("GET", "/resources/status")
        resources = result.get("resources")
        if (
            not isinstance(resources, list)
            or len(resources) > 100
            or any(not isinstance(item, dict) for item in resources)
        ):
            raise ServiceError("invalid_response")
        return result

    def allocate(self, allocation, key):
        if (
            not isinstance(key, str)
            or not key
            or len(key) > 200
            or any(not 33 <= ord(char) <= 126 for char in key)
        ):
            raise ServiceError("invalid_idempotency_key")
        fields = {"resource_id", "node_id", "cpus", "memory_mb", "scratch_mb", "timeout_seconds"}
        if not isinstance(allocation, dict) or set(allocation) != fields:
            raise ServiceError("invalid_allocation")
        result = self._manager(
            "POST",
            "/jobs",
            body=allocation,
            headers={"Idempotency-Key": key},
            expected_status=202,
            publication=True,
        )
        return self._record(result, job=True, publication=True)

    def job(self, job_id):
        job_id = _id(job_id)
        return self._record(self._manager("GET", "/jobs/" + job_id), job_id, job=True)

    def cancel(self, job_id):
        job_id = _id(job_id)
        return self._record(self._manager("POST", "/jobs/" + job_id + "/cancel"), job_id, job=True)

    def send(self, job_id, message_id, line):
        job_id = _id(job_id)
        if (
            not isinstance(message_id, str)
            or not message_id
            or len(message_id) > 200
            or not isinstance(line, str)
            or "\n" in line
            or "\r" in line
        ):
            raise ServiceError("invalid_input")
        try:
            if len(line.encode("utf-8")) > 2 * 1024 * 1024:
                raise ValueError
        except (UnicodeError, ValueError):
            raise ServiceError("invalid_input") from None
        return self._manager(
            "POST", "/jobs/" + job_id + "/stdin", body={"id": message_id, "line": line}
        )

    def output(self, job_id, after=0):
        job_id = _id(job_id)
        if type(after) is not int or after < 0:
            raise ServiceError("invalid_cursor")
        result = self._manager("GET", f"/jobs/{job_id}/stdout?after={after}")
        lines, cursor = result.get("lines"), result.get("next_cursor")
        if not isinstance(lines, list) or len(lines) > 64 or type(cursor) is not int:
            raise ServiceError("invalid_response")
        last = after
        for item in lines:
            if (
                not isinstance(item, dict)
                or type(item.get("sequence")) is not int
                or item["sequence"] != last + 1
                or not isinstance(item.get("line"), str)
            ):
                raise ServiceError("invalid_response")
            last = item["sequence"]
        if cursor != last:
            raise ServiceError("invalid_response")
        return result

    def close_input(self, job_id):
        return self._manager("POST", "/jobs/" + _id(job_id) + "/stdin/close")


class ChatModel(_HTTP):
    """A configurable tool-calling chat endpoint; no provider SDK in sandboxes."""

    def __init__(self, base_url, model, api_key=None, client=None):
        url = _url(base_url)
        if "?" in base_url or ".." in url.path.split("/"):
            raise ServiceError("invalid_url")
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise ServiceError("invalid_configuration")
        self.base_url, self.model, self.api_key = str(url).rstrip("/"), model, _key(api_key)
        super().__init__(client)

    def _complete(self, messages, *, tools=None, json_mode=False):
        if not isinstance(messages, list) or not messages or len(messages) > 300:
            raise ServiceError("invalid_messages")
        body = {"model": self.model, "messages": messages, "max_tokens": 4096}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        result = self.request(
            "POST",
            self.base_url + "/chat/completions",
            body=body,
            key=self.api_key,
            maximum=MAX_REQUEST_BYTES,
        )
        try:
            choices = result["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError
            message = choices[0]["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError
            content, calls = message.get("content"), message.get("tool_calls")
            if content is not None and not isinstance(content, str):
                raise ValueError
            if calls is not None and (not isinstance(calls, list) or len(calls) > 20):
                raise ValueError
            if not content and not calls:
                raise ValueError
            clean = {"role": "assistant", "content": content}
            if calls:
                for call in calls:
                    if (
                        not isinstance(call, dict)
                        or not isinstance(call.get("id"), str)
                        or not call["id"]
                        or len(call["id"]) > 200
                        or call.get("type") != "function"
                        or not isinstance(call.get("function"), dict)
                        or not isinstance(call["function"].get("name"), str)
                        or not isinstance(call["function"].get("arguments"), str)
                    ):
                        raise ValueError
                if len({call["id"] for call in calls}) != len(calls):
                    raise ValueError
                clean["tool_calls"] = calls
            return clean
        except (KeyError, IndexError, TypeError, ValueError):
            raise ServiceError("invalid_model_response") from None

    def complete(self, messages, tools=None):
        return self._complete(messages, tools=tools)

    def complete_json(self, system, user):
        if not isinstance(system, str) or not isinstance(user, (str, dict, list)):
            raise ServiceError("invalid_messages")
        try:
            content = user if isinstance(user, str) else json.dumps(user, allow_nan=False)
            message = self._complete(
                [{"role": "system", "content": system}, {"role": "user", "content": content}],
                json_mode=True,
            )
            result = _json(message["content"])
            if not isinstance(result, dict):
                raise ValueError
            return result
        except (TypeError, ValueError, UnicodeError, RecursionError):
            raise ServiceError("invalid_model_response") from None


class _ReadableHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden, self.text = 0, []

    def handle_starttag(self, tag, _attrs):
        if tag in {"script", "style", "noscript", "template"}:
            self.hidden += 1
        elif tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"} and not self.hidden:
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "template"} and self.hidden:
            self.hidden -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "tr"} and not self.hidden:
            self.text.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


class WebResearch(_HTTP):
    """Optional native You.com/gateway search and credential-free HTTPS reads.

    The operator-configured search gateway returns JSON for GET ?q=<query>.
    Search credentials are never applied to pages selected by the agent. A
    supplied HTTP client is intended for tests; production uses TLS validation,
    no proxies, no redirects, and no persistent public connections.
    """

    def __init__(self, search_url=None, search_key=None, client=None, *, provider="gateway"):
        if provider not in {"gateway", "you"}:
            raise ServiceError("invalid_configuration")
        self.provider = provider
        if provider == "you":
            if search_url is not None and search_url != YOU_SEARCH_URL:
                raise ServiceError("invalid_configuration")
            self.search_url = YOU_SEARCH_URL
        elif search_url:
            url = _url(search_url)
            if "?" in search_url:
                raise ServiceError("invalid_url")
            self.search_url = str(url)
        else:
            self.search_url = None
        self.search_key = _key(search_key)
        super().__init__(client, public=True)

    def search(self, query, *, deadline=None, timeout=None):
        if not self.search_url or (self.provider == "you" and not self.search_key):
            raise ServiceError("search_not_configured")
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ServiceError("invalid_query")
        if self.provider == "you":
            result = self.request(
                "POST",
                YOU_SEARCH_URL,
                body={"query": query, "count": 5},
                headers={"X-API-Key": self.search_key},
                maximum=MAX_WEB_BYTES,
                extensions={"timeout": timeout} if timeout else None,
                deadline=deadline,
            )
            return self._you_results(result)
        url = httpx.URL(self.search_url).copy_add_param("q", query)
        return self.request(
            "GET",
            url,
            key=self.search_key,
            maximum=MAX_WEB_BYTES,
            extensions={"timeout": timeout} if timeout else None,
            deadline=deadline,
        )

    def research_capabilities(self):
        from .you_mcp import YouMCP

        return YouMCP(self.search_key if self.provider == "you" else None).capabilities()

    def search_papers(self, query):
        from .you_mcp import YouMCP

        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ServiceError("invalid_query")
        response = YouMCP(self.search_key if self.provider == "you" else None).search(
            query, PAPER_DOMAINS
        )
        if (
            isinstance(response, dict)
            and "results" not in response
            and isinstance(response.get("data"), dict)
        ):
            response = response["data"]
        if not isinstance(response, dict):
            raise ServiceError("you_mcp_invalid_response")
        documents = self._you_results(response)["results"]["web"]
        papers, seen = [], set()
        for document in documents:
            try:
                canonical = paper_source_url(document["url"])
            except ServiceError:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            papers.append({**document, "url": canonical, "source_type": "primary_paper"})
        return {"provider": "you_mcp", "query": query, "papers": papers[:5]}

    def read_paper(self, url):
        from .you_mcp import YouMCP

        canonical = paper_source_url(url)
        arxiv = canonical.startswith("https://arxiv.org/abs/")
        preferred = canonical.replace("/abs/", "/html/", 1) if arxiv else canonical
        try:
            source = self.read(preferred)
            content_kind = "full_text" if arxiv else "publication_page"
            if len(source["text"].strip()) < 200:
                raise ServiceError("paper_text_unavailable")
        except ServiceError:
            # Contents extraction remains fixed-origin and receives only validated paper URLs.
            # It is useful for publication pages rendered by JavaScript; keys never reach them.
            read_url = canonical
            if self.provider == "you" and self.search_key:
                result = YouMCP(self.search_key).contents(read_url)
                if isinstance(result, dict):
                    result = result.get("results", result.get("data"))
            else:
                source = self.read(canonical)
                result = [{"url": canonical, "markdown": source["text"]}]

            if len(result) != 1 or not isinstance(result[0], dict):
                raise ServiceError("invalid_response")
            item = result[0]
            if paper_source_url(item.get("url", "")) != canonical:
                raise ServiceError("paper_source_identity_mismatch")
            text = item.get("markdown")
            if not isinstance(text, str) or len(text.strip()) < 200:
                raise ServiceError("paper_text_unavailable")
            source = {"url": read_url, "text": text[:100000], "truncated": len(text) > 100000}
            content_kind = "abstract_page" if arxiv else "publication_page"
        text = (
            source["text"].replace(self.search_key, "[redacted]")
            if self.search_key
            else source["text"]
        )
        return {
            "url": canonical,
            "read_url": source["url"],
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "truncated": source.get("truncated", False),
            "content_kind": content_kind,
            "retrieved_at": time.time(),
            "provider": "you_mcp_and_public_page",
        }

    def _you_results(self, result):
        """Only expose bounded search documents, never provider diagnostics."""
        source = result.get("results")
        if not isinstance(source, dict) or not any(name in source for name in ("web", "news")):
            raise ServiceError("invalid_response")
        cleaned = {}
        for category in ("web", "news"):
            items = source.get(category, [])
            if not isinstance(items, list):
                raise ServiceError("invalid_response")
            documents = []
            for item in items[:10]:
                if not isinstance(item, dict):
                    raise ServiceError("invalid_response")
                url = item.get("url")
                if not isinstance(url, str) or len(url) > 8192:
                    raise ServiceError("invalid_response")
                document = {"url": url}
                for field in ("title", "description"):
                    value = item.get(field)
                    if isinstance(value, str):
                        document[field] = value[:4000]
                snippets = item.get("snippets", [])
                if not isinstance(snippets, list) or any(
                    not isinstance(snippet, str) for snippet in snippets
                ):
                    raise ServiceError("invalid_response")
                document["snippets"] = [snippet[:4000] for snippet in snippets[:8]]
                # Even an unexpected upstream echo must not enter agent context.
                if self.search_key:
                    for field, value in document.items():
                        if isinstance(value, str):
                            document[field] = value.replace(self.search_key, "[redacted]")
                        else:
                            document[field] = [
                                snippet.replace(self.search_key, "[redacted]") for snippet in value
                            ]
                documents.append(document)
            cleaned[category] = documents
        return {"results": cleaned}

    @staticmethod
    def _pinned_url(value):
        url = _url(value, https=True)
        if url.port not in {None, 443} or "%" in url.host:
            raise ServiceError("web_destination_denied")
        try:
            addresses = socket.getaddrinfo(
                url.raw_host.decode("ascii"), 443, type=socket.SOCK_STREAM
            )
            if not addresses or len(addresses) > 32:
                raise ValueError
            vetted = []
            for address in addresses:
                ip = ipaddress.ip_address(address[4][0])
                if not ip.is_global or ip.is_multicast or ip.is_reserved:
                    raise ServiceError("web_destination_denied")
                vetted.append(str(ip))
            return url, url.copy_with(host=vetted[0])
        except ServiceError:
            raise
        except (OSError, ValueError, TypeError, IndexError):
            raise ServiceError("web_destination_unavailable") from None

    def read(self, url):
        original, pinned = self._pinned_url(url)
        hostname = original.raw_host.decode("ascii")
        raw, content_type = self.request(
            "GET",
            pinned,
            maximum=MAX_WEB_BYTES,
            text=True,
            headers={
                "Host": original.netloc.decode("ascii"),
                "Accept": "text/html,text/plain",
                "Connection": "close",
            },
            extensions={"sni_hostname": hostname},
        )
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
            raise ServiceError("unsupported_web_content")
        text = raw.decode("utf-8", errors="replace")
        if media_type != "text/plain":
            parser = _ReadableHTML()
            parser.feed(text)
            text = "\n".join(" ".join(line.split()) for line in "".join(parser.text).splitlines())
        return {"url": str(original), "text": text[:100_000], "truncated": len(text) > 100_000}
