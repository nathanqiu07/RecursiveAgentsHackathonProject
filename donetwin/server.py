"""DoneTwin server: serves the 3D twin UI and runs agent loops in background threads.

    python server.py            -> http://localhost:8000
"""
import json
import os
import re
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from donetwin import halls
from donetwin.hall_run import execute_hall, is_2n_hall
from donetwin.llm import LLM
from donetwin.loop import RUNS_DIR, Run, execute
from donetwin.mock import MockLLM
from donetwin.sandbox import docker_available, openshell_available

ROOT = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = os.path.join(ROOT, "scenarios")
RUNS = {}
SANDBOX_MODE = os.getenv("SANDBOX_MODE") or (
    "openshell" if openshell_available() else "docker" if docker_available() else "subprocess")
# Held-out generated halls offered in the UI next to the scenario files (never in the hill-climb library).
GENERATED = [f"{t}-{halls.heldout_seeds()[0]}" for t in ("small", "medium", "large")]


def load_scenario(name):
    m = re.fullmatch(r"(small|medium|large)-(\d+)", os.path.basename(name))
    if m:
        return halls.generate(int(m.group(2)), m.group(1))
    with open(os.path.join(SCENARIOS, f"{os.path.basename(name)}.json"), encoding="utf-8") as f:
        return json.load(f)


def saved_events(run_id):
    path = os.path.join(RUNS_DIR, os.path.basename(run_id), "events.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        if "/api/events" not in (str(args[0]) if args else ""):
            sys.stderr.write("[http] " + fmt % args + "\n")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            self.path = "/ui/index.html"
            return super().do_GET()
        if u.path == "/sample_events.json":
            self.path = "/ui/sample_events.json"
            return super().do_GET()
        if u.path == "/api/scenario":
            try:
                return self._json(load_scenario(q.get("name", "sorting_cell")))
            except FileNotFoundError:
                return self._json({"error": "unknown scenario"}, 404)
        if u.path == "/api/scenarios":
            return self._json(sorted(f[:-5] for f in os.listdir(SCENARIOS) if f.endswith(".json")) + GENERATED)
        if u.path == "/api/runs":
            rows = []
            if os.path.isdir(RUNS_DIR):
                for rid in os.listdir(RUNS_DIR):
                    ev = saved_events(rid) or []
                    if not ev:
                        continue
                    done = next((e for e in reversed(ev) if e["type"] == "done"), None)
                    rows.append({"run_id": rid, "goal": ev[0].get("goal"),
                                 "status": done["status"] if done else "running",
                                 "started": os.path.getmtime(os.path.join(RUNS_DIR, rid))})
            return self._json(sorted(rows, key=lambda r: -r["started"]))
        if u.path == "/api/events":
            rid, since = q.get("run", ""), int(q.get("since", 0))
            run = RUNS.get(rid)
            if run:
                with run.lock:
                    evs = run.events[since:]
                return self._json({"events": evs, "finished": run.finished})
            evs = saved_events(rid)
            if evs is None:
                return self._json({"error": "unknown run"}, 404)
            return self._json({"events": evs[since:], "finished": True})
        if u.path.startswith("/ui/") or u.path.startswith("/scenarios/") or u.path.startswith("/runs/"):
            return super().do_GET()
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/run":
            return self.send_error(404)
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        scenario_name = body.get("scenario", "sorting_cell")
        base = load_scenario(scenario_name)
        goal = (body.get("goal") or base.get("goal", "")).strip()
        rid = time.strftime("%H%M%S-") + uuid.uuid4().hex[:4]
        run = Run(rid, goal, scenario_name)
        RUNS[rid] = run
        if is_2n_hall(base):  # 2N halls: Nemotron hill-climbs the verified hall score
            try:
                llm = None if body.get("mock") else LLM()
            except RuntimeError as e:
                return self._json({"error": str(e)}, 400)
            library = None if scenario_name in GENERATED else "library"  # held-out halls never train it
            threading.Thread(target=execute_hall, args=(run, llm, base),
                             kwargs={"iters": int(body.get("max_iters", 6)), "library_dir": library},
                             daemon=True).start()
            return self._json({"run_id": rid})
        try:
            llm = MockLLM() if body.get("mock") else LLM()
        except RuntimeError as e:
            return self._json({"error": str(e)}, 400)
        threading.Thread(target=execute, args=(run, llm, base),
                         kwargs={"max_iters": int(body.get("max_iters", 6)), "sandbox_mode": SANDBOX_MODE,
                                 "vision_mode": body.get("vision", "final")}, daemon=True).start()
        return self._json({"run_id": rid})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"DoneTwin on http://localhost:{port}  (sandbox: {SANDBOX_MODE})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
