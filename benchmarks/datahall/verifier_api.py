"""HTTP verifier for generated GPU data halls (the evolve loop's held-out halls), for the team dashboard (stdlib only).

  .venv/bin/python -I benchmarks/datahall/verifier_api.py [--port 9003]

Rikin's donetwin/verifier_server.py (port 9000) is bound to the fixed Hall B. This one rebuilds any generated hall
from its (seed, tier) with donetwin's halls.generate and scores the layout with donetwin/donetwin/score.py, so the
hall itself can never be submitted, only decisions. No score is ever taken from the request.

POST /evaluations
  {"submission": {"seed": 10007, "tier": "medium", "layout": {"racks": [...]}}}
      -> 200 {"status": "completed", "score": 0-100, "feasible", "hall", "components", "violations", "feedback"}
  {"submission": {"halls": [{"seed", "tier", "layout"}, ...]}}
      -> 200 {"status": "completed", "score": mean over the halls, "results": [...], "feedback"}
  The body may also be the submission itself ({"seed", "tier", "layout"} at the top level).
  An infeasible layout scores 0 with its violations in the feedback. Unknown tier, bad seed, duplicate hall or
  malformed body -> 422 {"status": "rejected", "error"}
GET /health -> {"ok": true, "tiers": [...], "heldout_seeds": [first, last]}
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "donetwin"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from donetwin import halls  # noqa: E402
from donetwin import score as scoring  # noqa: E402

MAX_BODY = 8 * 1024 * 1024
MAX_HALLS = 64


class Rejected(ValueError):
    """The request cannot be scored at all (as opposed to a valid request with an invalid layout, which scores 0)."""


@lru_cache(maxsize=256)
def _hall_json(seed: int, tier: str) -> str:
    return json.dumps(halls.generate(seed, tier))


def hall_for(seed: Any, tier: Any) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Rejected(f"seed must be a non-negative integer, got {seed!r}")
    if not isinstance(tier, str) or tier not in halls.TIERS:
        raise Rejected(f"tier must be one of {sorted(halls.TIERS)}, got {tier!r}")
    return json.loads(_hall_json(seed, tier))


def violation_lines(res: dict[str, Any], limit: int = 25) -> list[str]:
    out = []
    for ch in res.get("checks", []):
        if ch.get("passed"):
            continue
        for v in ch.get("violations", []):
            out.append(f"[{ch['name']}] {v.get('detail', '')}" + (f" -> fix: {v['fix']}" if v.get("fix") else ""))
    return out[:limit]


def score_one(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise Rejected("each hall must be an object with 'seed', 'tier' and 'layout'")
    hall = hall_for(item.get("seed"), item.get("tier"))
    layout = item.get("layout")
    if not isinstance(layout, dict) or not isinstance(layout.get("racks"), list):
        raise Rejected("'layout' must be an object {\"racks\": [...]}")
    try:
        res = scoring.score(hall, layout)
    except Exception as exc:  # the scorer validates most structure; a crash on an odd shape is an invalid layout
        return {"hall": hall["name"], "seed": item["seed"], "tier": item["tier"], "score": 0.0, "feasible": False,
                "components": None, "violations": [f"[payload] malformed layout: {type(exc).__name__}: {exc}"[:300]]}
    return {"hall": hall["name"], "seed": item["seed"], "tier": item["tier"], "score": float(res["score"]),
            "feasible": bool(res["feasible"]), "it_kw": res.get("it_kw"), "upper_bound_kw": res.get("upper_bound_kw"),
            "components": res.get("components"), "violations": violation_lines(res)}


def line(r: dict[str, Any]) -> str:
    head = f"{r['hall']} (seed {r['seed']}, {r['tier']}): score {r['score']:.3f}, {'feasible' if r['feasible'] else 'INFEASIBLE'}"
    return "\n".join([head] + [f"  {v}" for v in r["violations"]])


def evaluate(req: Any) -> dict[str, Any]:
    if not isinstance(req, dict):
        raise Rejected("body must be a JSON object")
    sub = req.get("submission", req)
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except ValueError as exc:
            raise Rejected(f"submission is not valid JSON: {exc}") from None
    if not isinstance(sub, dict):
        raise Rejected("submission must be an object")
    if "halls" not in sub:
        r = score_one(sub)
        return {"status": "completed", **r, "feedback": line(r)}
    items = sub["halls"]
    if not isinstance(items, list) or not items or len(items) > MAX_HALLS:
        raise Rejected(f"'halls' must be a non-empty list of at most {MAX_HALLS}")
    seen = set()
    for it in items:
        key = (it.get("seed"), it.get("tier")) if isinstance(it, dict) else None
        if key in seen:
            raise Rejected(f"duplicate hall {key!r}")
        seen.add(key)
    results = [score_one(it) for it in items]
    mean = sum(r["score"] for r in results) / len(results)
    feasible = sum(r["feasible"] for r in results)
    head = f"mean hall score {mean:.3f} over {len(results)} halls; {feasible} feasible"
    return {"status": "completed", "score": mean, "n": len(results), "feasible": feasible, "results": results,
            "feedback": "\n".join([head] + [line(r) for r in results])}


class Handler(BaseHTTPRequestHandler):
    server_version = "DatahallVerifier/1.0"

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0].rstrip("/") == "/health":
            seeds = halls.heldout_seeds()
            return self._send(200, {"ok": True, "tiers": sorted(halls.TIERS), "heldout_seeds": [seeds[0], seeds[-1]]})
        self._send(404, {"status": "rejected", "error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?")[0].rstrip("/") != "/evaluations":
            return self._send(404, {"status": "rejected", "error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send(400, {"status": "rejected", "error": "bad Content-Length"})
        if n > MAX_BODY:
            return self._send(413, {"status": "rejected", "error": f"body exceeds {MAX_BODY} bytes"})
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8") or "null")
        except (ValueError, UnicodeDecodeError) as exc:
            return self._send(400, {"status": "rejected", "error": f"malformed JSON: {exc}"})
        try:
            self._send(200, evaluate(req))
        except Rejected as exc:
            self._send(422, {"status": "rejected", "error": str(exc)})
        except Exception as exc:  # a verifier bug must surface as an error, never as a score
            self._send(500, {"status": "failed", "error": f"could not score submission: {exc!r}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[datahall-verifier] " + (fmt % args) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9003)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"datahall verifier on http://127.0.0.1:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
