"""HTTP verifier for ALE-Bench AHC039, for the team dashboard's benchmark contract (stdlib only).

  .venv/bin/python -I benchmarks/alebench/verifier_api.py [--port 9002]

Wraps `benchmarks.alebench.scorer`: inputs come from the contest's own generator and outputs are scored by the
contest's own visualizer (WASM via Node). No score is ever taken from the request.

POST /evaluations
  {"submission": {"program": "<python source>"}}
      -> runs the program on the 5 public Lite seeds (0-4) at the contest time limit, one evaluation at a time
         so concurrent submissions cannot slow each other into timeouts.
         200 {"status": "completed", "score": total official score over the seeds, "cases": [...], "feedback"}
  {"submission": {"seed": 3, "output": "<program output text>"}}
      -> 200 {"status": "completed", "score": official score of that output on that seed's input, "feedback"}
  "feedback" opens with two plain sentences comparing the result to the greedy reference program on the same seeds
  ("baseline": {"name", "score", "is_baseline"}), then "Details:" per seed.
  Malformed body -> 400/422 {"status": "rejected", "error"}
GET /health -> {"ok": true, "problem": "ahc039", "public_seeds": [...]}

Submitted programs run on this host in their own process group with a minimal environment and a hard timeout
(see scorer.run_case). That is not a security sandbox: bind to loopback and only accept trusted submitters.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.alebench import scorer  # noqa: E402

MAX_BODY = 2 * 1024 * 1024
MAX_SEED = 2**64 - 1
EVAL_LOCK = threading.Lock()


class Rejected(ValueError):
    """The request cannot be scored at all (a program that crashes or times out still scores, as 0 per case)."""


def public_seeds() -> list[int]:
    return [int(s) for s in scorer.metadata()["seeds"]["public_lite"]]


def score_program(code: str, seeds: list[int] | None = None, workers: int = 5) -> dict[str, Any]:
    seeds = public_seeds() if seeds is None else seeds
    with EVAL_LOCK:
        rows = scorer.evaluate(code, seeds, scorer.contest_time_limit(), workers)
    ref = scorer.reference_scores(seeds)
    total = sum(r["score"] for r in rows)
    for r in rows:
        r["ratio_vs_greedy"] = round(scorer.ratio(r["score"], ref[r["seed"]]), 4)
    ok = sum(1 for r in rows if r["status"] == "ok")
    mean_ratio = sum(r["ratio_vs_greedy"] for r in rows) / len(rows)
    lines = [f"total official score {total} over public seeds {seeds}; {ok}/{len(rows)} ok; "
             f"mean ratio vs reference greedy {mean_ratio:.4f}"]
    for r in rows:
        extra = f" ({r['error']})" if r.get("error") else ""
        lines.append(f"seed {r['seed']}: {r['status']}{extra}, score {r['score']}, "
                     f"ratio {r['ratio_vs_greedy']:.4f}, {r['wall_s']} s")
    greedy_total = sum(ref[s] for s in seeds)
    is_greedy = reference_name(code) == "greedy"
    return {"status": "completed", "score": float(total), "mean_ratio_vs_greedy": mean_ratio, "ok_cases": ok,
            "cases": rows, "baseline": {"name": "greedy reference program", "score": float(greedy_total),
                                        "is_baseline": is_greedy},
            "feedback": explain(code, rows, ref) + "\n\nDetails:\n" + "\n".join(lines)}


def reference_name(code: str) -> str | None:
    """'greedy' (the baseline) or 'naive' when the source is one of the reference programs, else None."""
    for name, path in (("greedy", scorer.GREEDY), ("naive", scorer.NAIVE)):
        try:
            if code.strip() == path.read_text().strip():
                return name
        except OSError:
            continue
    return None


def times(x: float, ref: float) -> str:
    r = x / ref if ref > 0 else 0.0
    if r >= 1:
        return f"{r:.1f}x"
    return f"{r * 100:.2f}% of" if r < 0.01 else f"{r * 100:.0f}% of"


def explain(code: str, rows: list[dict[str, Any]], ref: dict[int, int]) -> str:
    """Two plain sentences: the net catch against the greedy reference program on the same maps."""
    k = len(rows)
    total = sum(r["score"] for r in rows)
    gtotal = sum(ref[r["seed"]] for r in rows)
    maps = f"{k} test map{'s' if k != 1 else ''}"
    name = reference_name(code)
    if name == "greedy":
        return (f"This is the baseline: the greedy reference program catches {total:,} net fish over {maps}. "
                "A higher total means nets that hold more mackerel and fewer sardines.")
    first = f"Catches {total:,} net fish over {maps}, {times(total, gtotal)} the greedy reference program ({gtotal:,})."
    if name == "naive":
        first = "This is the naive reference program (a lower bound). " + first
    wins = sum(1 for r in rows if r["score"] > ref[r["seed"]])
    failed = sum(1 for r in rows if r["status"] != "ok")
    second = f"It beats greedy on {wins} of {k} maps" + (
        f"; {failed} run{'s' if failed != 1 else ''} crashed or ran out of time and scored 0" if failed else "")
    return first if name == "naive" else f"{first} {second}."


def score_output(seed: int, output: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="alebench_out_") as tmp:
        out = Path(tmp) / "out.txt"
        out.write_text(output)
        row = scorer.score_files([(scorer.input_path(seed), out)])[0]
    score = 0 if row["err"] else int(row["score"])
    feedback = f"seed {seed}: " + (f"wrong_answer ({row['err'][:200]}), score 0" if row["err"] else f"ok, score {score}")
    try:
        greedy = int(scorer.reference_scores([seed])[seed])
    except Exception:  # no reference for this seed: the score stands without a baseline comparison
        greedy = None
    if greedy is None:
        summary = f"Catches {score:,} net fish on test map {seed}."
    else:
        summary = (f"Catches {score:,} net fish on test map {seed}; the greedy reference program catches {greedy:,}."
                   + (" The output breaks a rule of the contest, so it scores 0." if row["err"] else ""))
    return {"status": "completed", "score": float(score), "seed": seed, "error": row["err"] or None,
            "baseline": None if greedy is None else {"name": "greedy reference program", "score": float(greedy),
                                                     "is_baseline": False},
            "feedback": summary + "\n\nDetails:\n" + feedback}


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
    if "program" in sub:
        if not isinstance(sub["program"], str) or not sub["program"].strip():
            raise Rejected("'program' must be non-empty Python source")
        return score_program(sub["program"])
    if "output" in sub:
        seed = sub.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise Rejected("'seed' must be a non-negative integer")
        if not isinstance(sub["output"], str):
            raise Rejected("'output' must be text")
        return score_output(seed, sub["output"])
    raise Rejected("submission needs 'program', or 'seed' and 'output'")


class Handler(BaseHTTPRequestHandler):
    server_version = "AleBenchVerifier/1.0"

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0].rstrip("/") == "/health":
            return self._send(200, {"ok": True, "problem": scorer.PROBLEM, "public_seeds": public_seeds()})
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
        except Exception as exc:  # a verifier fault must surface as an error, never as a score
            self._send(500, {"status": "failed", "error": f"could not score submission: {exc!r}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[alebench-verifier] " + (fmt % args) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9002)
    args = ap.parse_args()
    scorer.require_data()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"alebench verifier on http://127.0.0.1:{args.port}  problem={scorer.PROBLEM}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
