"""HTTP verifier for KernelBench mini's held-out kernels, for the team dashboard (stdlib only).

  .venv/bin/python -I benchmarks/kernelbench_mini/verifier_api.py [--port 9005]

Wraps `remote.evaluate_candidate`, which runs evaluate.py on a CUDA host (KB_REMOTE=user@host, or a local GPU):
static checks, correctness on fresh inputs, then CUDA-event timing against PyTorch eager on the same inputs.
No score is ever taken from the request.

POST /evaluations
  {"submission": {"solutions": [{"task": "kb_relu", "model_new": "<python source>"}, ...]}}
      -> 200 {"status": "completed", "score": mean capped speedup over the 5 held-out kernels, "results", "baseline",
              "feedback"}. A held-out kernel that is missing, wrong or fails to compile scores 0.
  With no GPU attached -> 200 {"status": "failed", "feedback": "Needs a GPU VM, not yet attached ..."}: an error,
  never a score.
  The one exception is the baseline itself: a ModelNew that is exactly PyTorch eager (eager_candidate(task)) is
  speedup 1.0 by definition, so it scores 1.0 without a GPU and its feedback says it was not timed (with no GPU
  attached, the feedback opens with NOT_RUN, which the dashboard shows as the board's summary).
  Unknown task, duplicate task or malformed body -> 422 {"status": "rejected", "error"}
GET /health -> {"ok": true, "gpu": "remote" | "local" | null, "heldout": [...]}
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.kernelbench_mini import remote  # noqa: E402
from benchmarks.kernelbench_mini.splits import PROBLEMS, PROBLEMS_DIR  # noqa: E402

MAX_BODY = 4 * 1024 * 1024
NO_GPU = "Needs a GPU VM, not yet attached"
# Shown as the baseline's feedback (and so as the board's summary) while no GPU is attached.
NOT_RUN = "Not run: our Crusoe project has no GPU quota, so no kernels were timed. Baseline only."


class Rejected(ValueError):
    """The request cannot be scored at all (a kernel that is wrong or does not compile still scores, as 0)."""


def heldout() -> list[str]:
    return list(json.loads((HERE / "splits.json").read_text())["heldout"])


def eager_candidate(task: str) -> str:
    """The baseline ModelNew: the problem's own PyTorch eager Model, unchanged."""
    return (PROBLEMS_DIR / PROBLEMS[task][0]).read_text().rstrip() + "\n\n\nclass ModelNew(Model):\n    pass\n"


def explain(results: list[dict[str, Any]]) -> str:
    n = len(results)
    mean = sum(r["score"] for r in results) / n
    if all(r.get("is_baseline") for r in results):
        return (f"This is the baseline: PyTorch eager runs each of the {n} kernels at speedup 1.0 by definition (not "
                "timed). A higher score means faster kernels that still give the same answers.")
    correct = sum(1 for r in results if r.get("correct"))
    first = f"Runs {mean:.2f}x as fast as PyTorch eager on average over {n} kernels; the eager baseline is 1.00x."
    wrong = n - correct
    second = f"{correct} of {n} kernels are correct" + (f"; {wrong} fail{'s' if wrong == 1 else ''} and score 0" if wrong else "")
    return f"{first} {second}."


def evaluate(req: Any) -> dict[str, Any]:
    if not isinstance(req, dict) or not isinstance(req.get("submission"), (dict, str)):
        raise Rejected("body must be a JSON object with a 'submission' object")
    sub = req["submission"]
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except ValueError as exc:
            raise Rejected(f"submission is not valid JSON: {exc}") from None
    sols = sub.get("solutions") if isinstance(sub, dict) else None
    if not isinstance(sols, list) or not sols:
        raise Rejected("'solutions' must be a non-empty list of {\"task\", \"model_new\"}")
    names = heldout()
    by_task: dict[str, str] = {}
    for s in sols:
        task = s.get("task") if isinstance(s, dict) else None
        if task not in names:
            raise Rejected(f"task must be one of the held-out kernels {names}, got {task!r}")
        if task in by_task:
            raise Rejected(f"duplicate task {task!r}")
        if not isinstance(s.get("model_new"), str) or not s["model_new"].strip():
            raise Rejected(f"{task}: 'model_new' must be non-empty Python source")
        by_task[task] = s["model_new"]
    if all(by_task.get(t) == eager_candidate(t) for t in names):
        results = [{"task": t, "score": 1.0, "speedup": 1.0, "correct": True, "is_baseline": True} for t in names]
        return {"status": "completed", "score": 1.0, "results": results,
                "baseline": {"name": "PyTorch eager", "score": 1.0, "is_baseline": True},
                "feedback": (NOT_RUN if remote.gpu_mode() is None else explain(results))
                            + "\n\nDetails:\nThis is the baseline, PyTorch eager, at speedup 1.0 by definition. Not "
                              "timed: ModelNew is the reference Model for every held-out kernel."}
    if remote.gpu_mode() is None:
        return {"status": "failed", "score": None,
                "feedback": f"{NO_GPU}. Set KB_REMOTE=user@host (see setup_gpu_vm.sh) and restart this verifier."}
    results = []
    for t in names:
        if t not in by_task:
            results.append({"task": t, "score": 0.0, "correct": False, "error": "missing"})
            continue
        r = remote.evaluate_candidate(PROBLEMS[t][0], by_task[t])
        if r.get("error"):
            results.append({"task": t, "score": 0.0, "correct": False, "error": f"{r['error']}: {r.get('detail', '')}"[:300]})
        else:
            results.append({"task": t, "score": float(r.get("score") or 0.0), "speedup": r.get("speedup"),
                            "correct": bool(r.get("correct")), "suspicious": r.get("suspicious"),
                            "error": "; ".join(r.get("reasons") or [])[:300] or None})
    mean = sum(r["score"] for r in results) / len(results)
    lines = [f"{r['task']}: score {r['score']:.3f}" + (f", speedup {r['speedup']:.3f}x" if r.get("speedup") else "")
             + ("" if r["correct"] else f", not correct ({r.get('error') or 'failed'})") for r in results]
    return {"status": "completed", "score": mean, "results": results,
            "baseline": {"name": "PyTorch eager", "score": 1.0, "is_baseline": False},
            "feedback": explain(results) + "\n\nDetails:\n" + "\n".join(lines)}


class Handler(BaseHTTPRequestHandler):
    server_version = "KernelBenchVerifier/1.0"

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0].rstrip("/") == "/health":
            return self._send(200, {"ok": True, "gpu": remote.gpu_mode(), "heldout": heldout()})
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
        sys.stderr.write("[kernelbench-verifier] " + (fmt % args) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9005)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"kernelbench verifier on http://127.0.0.1:{args.port}  gpu={remote.gpu_mode()}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
