"""HTTP verifier for the team dashboard (stdlib only).

  python verifier_server.py [--port 9000] [--hall scenarios/data_hall_2n.json]

POST /evaluations  {"submission": <payload>, "base_checkpoint_url", "submission_id", "benchmark_id"}
  -> 200 {"status":"completed","score":float,"feedback":str,"feasible":bool}
  -> 400 {"status":"failed","error":str} on malformed JSON / missing submission
GET /health -> {"ok":true,"hall":name}
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from donetwin import score as scoring  # noqa: E402

MAX_BODY = 2 * 1024 * 1024
HALL = {}


def _feedback_text(fb):
    if isinstance(fb, str):
        return fb
    if isinstance(fb, (list, tuple)):
        return "\n".join(_feedback_text(x) for x in fb)
    return json.dumps(fb, default=str)


class Handler(BaseHTTPRequestHandler):
    server_version = "HallVerifier/1.0"

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") == "/health":
            return self._send(200, {"ok": True, "hall": HALL.get("name")})
        self._send(404, {"status": "failed", "error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0].rstrip("/") != "/evaluations":
            return self._send(404, {"status": "failed", "error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send(400, {"status": "failed", "error": "bad Content-Length"})
        if n > MAX_BODY:
            return self._send(413, {"status": "failed", "error": f"body exceeds {MAX_BODY} bytes"})
        try:
            req = json.loads(self.rfile.read(n).decode("utf-8") or "null")
        except (ValueError, UnicodeDecodeError) as e:
            return self._send(400, {"status": "failed", "error": f"malformed JSON: {e}"})
        if not isinstance(req, dict) or "submission" not in req or req["submission"] is None:
            return self._send(400, {"status": "failed", "error": "missing 'submission'"})
        sub = req["submission"]
        if isinstance(sub, str):  # tolerate a JSON-encoded string submission
            try:
                sub = json.loads(sub)
            except ValueError as e:
                return self._send(400, {"status": "failed", "error": f"submission is not valid JSON: {e}"})
        try:
            res = scoring.score(HALL, sub)
        except Exception as e:
            return self._send(400, {"status": "failed", "error": f"could not score submission: {e!r}"})
        self._send(200, {"status": "completed", "score": float(res.get("score", 0.0)),
                         "feedback": _feedback_text(res.get("feedback", "")),
                         "feasible": bool(res.get("feasible"))})

    def log_message(self, fmt, *a):
        sys.stderr.write("[verifier] " + (fmt % a) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--hall", default="scenarios/data_hall_2n.json")
    args = ap.parse_args()
    p = Path(args.hall)
    p = p if p.is_absolute() else ROOT / p
    HALL.update(json.loads(p.read_text(encoding="utf-8")))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"verifier on http://127.0.0.1:{args.port}  hall={HALL.get('name')}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
