"""HTTP verifier for the inference-serving case study, for the team dashboard's benchmark contract (stdlib only).

  .venv/bin/python -I benchmarks/inference_opt/verifier_api.py [--port 9004]

Regenerates each task's questions from its seed and tier (benchmarks/inference_opt/workload.py, deterministic) and
grades the submitted answers exactly in code. No score is ever taken from the request.

POST /evaluations
  {"submission": {"seed": 10000, "tier": "small", "answers": {"q1": 87, ...}}}
      -> 200 {"status": "completed", "score": accuracy, "correct", "n", "baseline", "feedback"}
  {"submission": {"tasks": [{"seed", "tier", "answers"}, ...]}, "split": "heldout"}
      -> 200 accuracy over every question of every task in the split; a missing task scores 0 on all its questions.
  "feedback" opens with one plain sentence comparing the answers to v0 (Super, reasoning on, one call per question)
  on the same tasks, from the live run's saved v0 solutions, then "Details:".
  Unknown seed/tier, task outside the split, duplicate task or malformed body -> 422 {"status": "rejected", "error"}
GET /health -> {"ok": true, "heldout_tasks": N}
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

from benchmarks.inference_opt.workload import TIERS, grade, load_tasks, make_questions  # noqa: E402

MAX_BODY = 1024 * 1024
V0_DIRS = [ROOT / "reports" / "inference_opt_2" / "solutions" / "v0-empty-brain-cold",
           ROOT / "runs" / "inference-opt-2" / "solutions" / "v0-empty-brain-cold"]
V0_NAME = "v0 (Super, reasoning on, one call per question)"


class Rejected(ValueError):
    """The request cannot be scored at all (as opposed to wrong answers, which score 0)."""


def split_keys(name: str) -> list[tuple[int, str]]:
    splits = dict(zip(("train", "dev", "heldout"), load_tasks()))
    if name not in splits:
        raise Rejected(f"unknown split {name!r}")
    return [(t.payload["seed"], t.payload["tier"]) for t in splits[name]]


def v0_answers() -> dict[tuple[int, str], dict[str, Any]]:
    """v0's held-out answers from the live run (first directory that has them); empty when no run saved them."""
    for d in V0_DIRS:
        out = {}
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                s = json.loads(f.read_text())["solution"]
                out[(int(s["seed"]), str(s["tier"]))] = s.get("answers") or {}
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if out:
            return out
    return {}


def score_task(seed: Any, tier: Any, answers: Any) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or tier not in TIERS:
        raise Rejected(f"each task needs an int seed and a tier in {list(TIERS)}, got seed={seed!r} tier={tier!r}")
    if not isinstance(answers, dict):
        raise Rejected("answers must be an object {question id: number}")
    g = grade(make_questions(seed, tier), answers)
    return {"seed": seed, "tier": tier, "correct": g["correct"], "n": g["n"]}


def explain(results: list[dict[str, Any]], base: dict[str, Any] | None) -> str:
    c, n = sum(r["correct"] for r in results), sum(r["n"] for r in results)
    head = f"Answers {c} of {n} questions correctly ({c / n:.1%})"
    if base is None:
        return head + "; no v0 answers are saved for these tasks yet, so there is nothing to compare against."
    bc = base["correct"]
    if bc == c:
        return f"{head}, the same accuracy as {V0_NAME}, so any cost it saved came free."
    return f"{head}, {abs(c - bc)} {'more' if c > bc else 'fewer'} than {V0_NAME}, which got {bc} of {n}."


def baseline_info(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    v0 = v0_answers()
    if not all((r["seed"], r["tier"]) in v0 for r in results):
        return None
    graded = [score_task(r["seed"], r["tier"], v0[(r["seed"], r["tier"])]) for r in results]
    c, n = sum(r["correct"] for r in graded), sum(r["n"] for r in graded)
    return {"name": "v0", "score": c / n if n else 0.0, "correct": c}


def evaluate(req: Any) -> dict[str, Any]:
    if not isinstance(req, dict) or not isinstance(req.get("submission"), dict):
        raise Rejected("body must be a JSON object with a 'submission' object")
    sub = req["submission"]
    if "tasks" not in sub:
        results = [score_task(sub.get("seed"), sub.get("tier"), sub.get("answers"))]
    else:
        tasks = sub["tasks"]
        if not isinstance(tasks, list) or not tasks:
            raise Rejected("'tasks' must be a non-empty list")
        by_key: dict[tuple[Any, Any], Any] = {}
        for t in tasks:
            if not isinstance(t, dict):
                raise Rejected("each task must be an object with seed, tier and answers")
            key = (t.get("seed"), t.get("tier"))
            if key in by_key:
                raise Rejected(f"duplicate task {key!r}")
            by_key[key] = t.get("answers")
        split = req.get("split")
        keys = split_keys(split) if split else list(by_key)
        extra = [k for k in by_key if k not in keys]
        if extra:
            raise Rejected(f"tasks outside split {split!r}: {extra}")
        results = [score_task(s, t, by_key.get((s, t), {})) for s, t in keys]
    c, n = sum(r["correct"] for r in results), sum(r["n"] for r in results)
    base = baseline_info(results)
    details = "\n".join(f"{r['tier']}-{r['seed']}: {r['correct']}/{r['n']}" for r in results)
    return {"status": "completed", "score": c / n if n else 0.0, "correct": c, "n": n, "results": results,
            "baseline": base, "feedback": explain(results, base) + "\n\nDetails:\n" + details}


class Handler(BaseHTTPRequestHandler):
    server_version = "InferenceOptVerifier/1.0"

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0].rstrip("/") == "/health":
            return self._send(200, {"ok": True, "heldout_tasks": len(split_keys("heldout"))})
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
        sys.stderr.write("[inference-opt-verifier] " + (fmt % args) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9004)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"inference_opt verifier on http://{args.host}:{args.port}/evaluations", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
