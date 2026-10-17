"""HTTP verifier for QAPLIB, for the team dashboard's benchmark contract (stdlib only).

  .venv/bin/python -I benchmarks/qaplib/verifier_api.py [--port 9001]

Wraps `benchmarks.qaplib.verify`, which recomputes the objective exactly from the instance's .dat file.
No score is ever taken from the request; it is always recomputed here.

POST /evaluations
  {"submission": {"instance": "nug22", "permutation": [...]}}
      -> 200 {"status": "completed", "score": BKS/cost, "cost", "bks", "gap_pct", "feedback"}
  {"submission": {"solutions": [{"instance", "permutation"}, ...]}, "split": "heldout"}
      -> 200 {"status": "completed", "score": mean BKS/cost over every instance of the split, ...}
         An instance of the split that is missing or has an invalid permutation scores 0, so a partial
         set cannot raise the mean. Without "split", the mean is over the submitted instances.
  Unknown instance, unknown split, duplicate instance or malformed body -> 400/422 {"status": "rejected", "error"}
GET /health -> {"ok": true, "instances": N}
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.qaplib.verify import INSTANCES, load_index, verify  # noqa: E402

MAX_BODY = 4 * 1024 * 1024


class Rejected(ValueError):
    """The request cannot be scored at all (as opposed to a valid request with an invalid answer, which scores 0)."""


def load_split(name: str) -> list[str]:
    splits = json.loads((ROOT / "benchmarks" / "qaplib" / "splits.json").read_text())
    if name not in ("train", "dev", "heldout"):
        raise Rejected(f"unknown split {name!r}")
    return list(splits[name])


def score_one(instance: Any, permutation: Any, index: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(instance, str) or instance not in index:
        raise Rejected(f"unknown instance {instance!r}")
    if isinstance(permutation, bool) or not isinstance(permutation, (list, str)):
        return {"instance": instance, "status": "invalid", "error": "permutation must be a list of integers",
                "score": 0.0, "cost": None, "bks": index[instance]["bks"], "gap_pct": None}
    if isinstance(permutation, list) and not all(isinstance(x, int) and not isinstance(x, bool) for x in permutation):
        return {"instance": instance, "status": "invalid", "error": "permutation must contain only integers",
                "score": 0.0, "cost": None, "bks": index[instance]["bks"], "gap_pct": None}
    v = verify(instance, permutation)
    v.pop("perm", None)
    return {"instance": instance, **v}


def line(r: dict[str, Any]) -> str:
    if r["status"] == "ok":
        return f"{r['instance']}: cost {r['cost']} vs BKS {r['bks']}, score {r['score']:.6f}, gap {r['gap_pct']:.3f}%"
    return f"{r['instance']}: {r['status']} ({r.get('error', '')}), score 0"


def evaluate(req: Any) -> dict[str, Any]:
    if not isinstance(req, dict) or not isinstance(req.get("submission"), (dict, str)):
        raise Rejected("body must be a JSON object with a 'submission' object")
    sub = req["submission"]
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except ValueError as exc:
            raise Rejected(f"submission is not valid JSON: {exc}") from None
        if not isinstance(sub, dict):
            raise Rejected("submission must be an object")
    index = load_index()
    if "solutions" not in sub:
        r = score_one(sub.get("instance"), sub.get("permutation"), index)
        return {"status": "completed", "score": float(r["score"]), "cost": r["cost"], "bks": r["bks"],
                "gap_pct": r["gap_pct"], "results": [r], "feedback": line(r)}
    sols = sub["solutions"]
    if not isinstance(sols, list) or not sols:
        raise Rejected("'solutions' must be a non-empty list")
    by_name: dict[str, Any] = {}
    for s in sols:
        if not isinstance(s, dict):
            raise Rejected("each solution must be an object with 'instance' and 'permutation'")
        name = s.get("instance")
        if name in by_name:
            raise Rejected(f"duplicate instance {name!r}")
        by_name[name] = s.get("permutation")
    split = req.get("split")
    names = load_split(split) if split else list(by_name)
    extra = [n for n in by_name if n not in names]
    if extra:
        raise Rejected(f"instances outside split {split!r}: {extra}")
    results = []
    for name in names:
        if name not in by_name:
            results.append({"instance": name, "status": "missing", "error": "no permutation submitted",
                             "score": 0.0, "cost": None, "bks": index[name]["bks"], "gap_pct": None})
        else:
            results.append(score_one(name, by_name[name], index))
    mean = sum(r["score"] for r in results) / len(results)
    ok = [r for r in results if r["status"] == "ok"]
    at_bks = sum(1 for r in ok if r["cost"] <= r["bks"])
    head = (f"mean BKS/cost {mean:.6f} over {len(results)} instances"
            f"{' of split ' + split if split else ''}; {len(ok)} valid, {at_bks} at or below BKS")
    return {"status": "completed", "score": mean, "n": len(results), "valid": len(ok), "at_bks": at_bks,
            "results": results, "feedback": "\n".join([head] + [line(r) for r in results])}


class Handler(BaseHTTPRequestHandler):
    server_version = "QaplibVerifier/1.0"

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0].rstrip("/") == "/health":
            return self._send(200, {"ok": True, "instances": len(load_index())})
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
        sys.stderr.write("[qaplib-verifier] " + (fmt % args) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9001)
    args = ap.parse_args()
    if not (INSTANCES / "index.json").is_file():
        sys.exit(f"missing {INSTANCES / 'index.json'}; run benchmarks/qaplib/fetch.py first")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"qaplib verifier on http://127.0.0.1:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
