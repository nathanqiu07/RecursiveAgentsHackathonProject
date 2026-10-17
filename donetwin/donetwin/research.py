"""Live, cited code references from the You.com Search API for the Spec agent."""
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

YDC_URL = "https://ydc-index.io/v1/search"


def _search(query, key, count=3):
    req = urllib.request.Request(
        YDC_URL, data=json.dumps({"query": query, "count": count}).encode(),
        headers={"X-API-Key": key, "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    out = []
    for w in data.get("results", {}).get("web", [])[:count]:
        snippet = " ".join(w.get("snippets") or [])[:400] or (w.get("description") or "")[:400]
        out.append({"query": query, "title": w.get("title", ""), "url": w.get("url", ""), "snippet": snippet})
    return out


def code_references(scenario):
    """Search the scenario's research queries in parallel. Returns [] when no YDC_API_KEY is set."""
    key = os.getenv("YDC_API_KEY")
    queries = scenario.get("research_queries", [])
    if not key or not queries:
        return []
    refs = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(lambda q: _safe(q, key), queries):
            refs.extend(res)
    return refs


def _safe(q, key):
    try:
        return _search(q, key)
    except Exception:
        return []
