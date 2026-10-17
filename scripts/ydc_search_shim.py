"""Translate the agent harness's search-gateway contract to You.com's search API.

The harness (harness/agent_harness/clients.py, WebResearch.search) sends
``GET <HARNESS_SEARCH_URL>?q=<query>`` and expects a JSON object back. You.com
takes ``POST https://ydc-index.io/v1/search`` with ``{"query", "count"}`` and an
``X-API-Key`` header. This shim bridges the two without changing SA's code.

The You.com key is read from this process's own environment (YDC_API_KEY) and is
never logged or returned. Bind to loopback only.

    YDC_API_KEY=... python scripts/ydc_search_shim.py   # serves 127.0.0.1:8011
"""

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

YDC_URL = "https://ydc-index.io/v1/search"
HOST, PORT = "127.0.0.1", int(os.environ.get("YDC_SHIM_PORT", "8011"))
RESULT_COUNT = 8
MAX_SNIPPET_CHARS = 1200


def _results(payload):
    """Flatten You.com's web/news sections into {title, url, snippet} sources."""
    sections = payload.get("results", payload) if isinstance(payload, dict) else {}
    flattened = []
    for section in ("web", "news"):
        for hit in sections.get(section) or []:
            if not isinstance(hit, dict) or not hit.get("url"):
                continue
            snippets = hit.get("snippets") or []
            text = " ".join(s for s in snippets if isinstance(s, str)) or hit.get("description") or ""
            flattened.append({
                "title": hit.get("title") or "",
                "url": hit["url"],
                "snippet": text[:MAX_SNIPPET_CHARS],
                "source": section,
            })
    return flattened


def search(query, key):
    body = json.dumps({"query": query, "count": RESULT_COUNT}).encode()
    request = urllib.request.Request(
        YDC_URL, data=body, method="POST",
        headers={"X-API-Key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.loads(response.read())


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, value):
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parts = urlsplit(self.path)
        if parts.path == "/health":
            return self._send(200, {"ok": True})
        query = (parse_qs(parts.query).get("q") or [""])[0].strip()
        if not query:
            return self._send(400, {"error": "missing q"})
        key = os.environ.get("YDC_API_KEY")
        if not key:
            return self._send(503, {"error": "search key not configured"})
        try:
            results = _results(search(query, key))
        except urllib.error.HTTPError as exc:
            return self._send(502, {"error": f"upstream status {exc.code}"})
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return self._send(502, {"error": "upstream unavailable"})
        self._send(200, {"query": query, "results": results})

    def log_message(self, fmt, *args):
        # Log the request line only; never headers (the harness may send a bearer token).
        sys.stderr.write("ydc-shim %s\n" % (fmt % args))


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
