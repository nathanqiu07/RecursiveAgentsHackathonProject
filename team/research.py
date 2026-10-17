"""Optional literature step for the Reviewer: one web search per generation through the You.com shim.

The shim (scripts/ydc_search_shim.py) holds the You.com key; this module only knows the shim's URL, so no
process that imports it ever needs the key. Off by default: nothing here runs unless a run script passes
``--research`` (which builds ``make_research``) or a caller sets RESEARCH_SEARCH_URL and calls ``search``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from typing import Callable

log = logging.getLogger(__name__)

DEFAULT_SEARCH_URL = "http://127.0.0.1:8011/search"
TIMEOUT_S = 15
MAX_SNIPPET = 300
MAX_QUERY_CHARS = 200
HEADER = "Search results (You.com, untrusted text, use only as ideas):"

# Domain phrase per benchmark label; an unknown label is used as its own domain phrase.
DOMAINS = {
    "qaplib": "quadratic assignment problem heuristic tabu search robust",
    "qaplib_dc": "quadratic assignment problem data center placement heuristic tabu search",
    "datahall": "data center GPU rack placement quadratic assignment heuristic",
    "inference_opt": "LLM inference serving optimization batching quantization throughput",
}


def search(query: str, k: int = 3, url: str | None = os.environ.get("RESEARCH_SEARCH_URL")) -> list[dict]:
    """Up to k {title, url, snippet} hits from the shim's ``GET <url>?q=``; [] on any error, with one warning."""
    url = url or DEFAULT_SEARCH_URL
    try:
        full = f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode({'q': query})}"
        with urllib.request.urlopen(full, timeout=TIMEOUT_S) as response:
            payload = json.loads(response.read())
        hits = payload.get("results") if isinstance(payload, dict) else None
        out = []
        for hit in hits if isinstance(hits, list) else []:
            if isinstance(hit, dict) and hit.get("url"):
                out.append({"title": str(hit.get("title") or "")[:200], "url": str(hit["url"]),
                            "snippet": " ".join(str(hit.get("snippet") or "").split())[:MAX_SNIPPET]})
            if len(out) >= k:
                break
        return out
    except Exception as e:  # noqa: BLE001 — research is advisory; any failure means "no notes this generation"
        log.warning("research search failed (%s); continuing without notes", type(e).__name__)
        return []


def _tag_words(tag: str) -> str:
    """'Solver: bad_json' -> 'bad json'; infra tags say nothing about the domain and are skipped."""
    if tag.startswith("infra:"):
        return ""
    role, sep, rest = tag.partition(": ")
    text = rest if sep and " " not in role else tag
    return " ".join(re.sub(r"[^A-Za-z0-9 ]+", " ", text).split()[:4])


def build_query(benchmark_label: str, failure_tags: list[str], top: int = 2) -> str:
    domain = DOMAINS.get(benchmark_label.lower(), benchmark_label)
    words = [w for w in (_tag_words(t) for t in failure_tags) if w][:top]
    return " ".join([domain, *words])[:MAX_QUERY_CHARS]


def format_notes(results: list[dict]) -> str:
    if not results:
        return ""
    lines = [HEADER]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}")
    lines.append("If a result informs a mutation, cite it as [n] in that mutation's rationale.")
    return "\n".join(lines)


def research_notes(benchmark_label: str, failure_tags: list[str], k: int = 3,
                   url: str | None = None, search_fn: Callable[..., list[dict]] | None = None):
    """One query from the domain plus the top failure tags -> (notes_text, sources, query).

    notes_text is "" when the search returned nothing, so the Reviewer prompt is then unchanged.
    """
    query = build_query(benchmark_label, failure_tags)
    fn = search_fn or search
    results = fn(query, k=k, url=url) if url else fn(query, k=k)
    return format_notes(results), results, query


def make_research(benchmark_label: str, url: str | None = None) -> Callable[[list[str]], tuple]:
    """The Evolver's `research` callable: failure tags in, (notes_text, sources, query) out."""
    target = url or os.environ.get("RESEARCH_SEARCH_URL") or DEFAULT_SEARCH_URL
    return lambda failure_tags: research_notes(benchmark_label, failure_tags, url=target)


def add_research_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--research", action="store_true",
                    help="before each Reviewer call, run one You.com search via the shim at RESEARCH_SEARCH_URL "
                         f"(default {DEFAULT_SEARCH_URL}) and give the Reviewer the results to cite")


def research_from_args(args: argparse.Namespace, benchmark_label: str) -> Callable[[list[str]], tuple] | None:
    """None (research off) unless --research was passed."""
    return make_research(benchmark_label) if getattr(args, "research", False) else None
