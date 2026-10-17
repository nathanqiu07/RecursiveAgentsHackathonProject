"""Publish every case study's solutions to SA's dashboard as verified leaderboard entries.

  .venv/bin/python -I scripts/publish_results_to_dashboard.py [--dry-run] [--retry-failed] [--no-wait]

Idempotent and re-runnable as live runs finish:
- Benchmarks are found by exact title before any is created (definitions are immutable on the dashboard).
- Each submission title carries a short hash of its payload; a title already on the benchmark is skipped.
- Scores are never sent. The dashboard forwards each payload to the case study's verifier service, which
  recomputes the score from the solution itself.

Verifiers it expects (started separately; see each file's docstring):
  data hall  http://127.0.0.1:9000/evaluations  donetwin/verifier_server.py, bound to Hall B (data_hall_2n)
  QAPLIB     http://127.0.0.1:9001/evaluations  benchmarks/qaplib/verifier_api.py
  AHC039     http://127.0.0.1:9002/evaluations  benchmarks/alebench/verifier_api.py

Solutions it finds:
  data hall  the greedy layout for Hall B (recomputed; deterministic)
  QAPLIB     the random and 2-swap baselines on the held-out split (2-swap cached in runs/dashboard/)
  AHC039     the greedy and naive reference programs, plus every runs/alebench-*/logs/*.py the team runner saved
  all        any run solution file runs/<run>/solutions/**/<name>.json of this shape (the runners do not write
             these yet; this is the contract for that fix):
               {"case_study": "datahall" | "qaplib" | "alebench", "task_id": str, "config": "v0 + final brain",
                "hall": "<hall name, data hall only>",
                "solution": {"racks": [...]} | {"instance": str, "permutation": [int]} | {"program": str}}
             A data-hall layout is submitted only for Hall B (the hall the 9000 verifier is bound to); a QAPLIB
             config only once it covers every held-out instance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "donetwin"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

API = os.environ.get("DASHBOARD_API", "http://127.0.0.1:3001")
RUNS = ROOT / "runs"
CACHE = RUNS / "dashboard"
QAP_TWO_SWAP_CACHE = CACHE / "qaplib_two_swap_heldout.json"
QAP_TIME_LIMIT_S = 10.0
HALL_B = ROOT / "donetwin" / "scenarios" / "data_hall_2n.json"
FAILED_STATES = {"failed", "error", "rejected", "cancelled"}
DONE_STATES = {"verified", "completed", "succeeded"} | FAILED_STATES


# ---------------------------------------------------------------- dashboard API

def api(method: str, path: str, body: Any = None, timeout: float = 60) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(API + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {exc.read()[:600].decode('utf-8', 'replace')}") from None


def checkpoint_url(ref: str) -> str:
    """The public GitHub commit the solutions were produced from: origin's URL pinned to a full SHA."""
    remotes = subprocess.run(["git", "-C", str(ROOT), "remote", "-v"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"github\.com[:/]([^/\s]+)/([^\s]+?)(?:\.git)?\s", remotes)
    if not m:
        raise SystemExit(f"no GitHub remote found in:\n{remotes}")
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", ref], capture_output=True, text=True, check=True).stdout.strip()
    return f"https://github.com/{m.group(1)}/{m.group(2)}/commit/{sha}"


def find_or_create(definition: dict[str, Any], dry_run: bool) -> str | None:
    for b in api("GET", "/api/benchmarks")["benchmarks"]:
        if b["title"] == definition["title"]:
            return b["id"]
    if dry_run:
        print(f"  [dry-run] would create benchmark {definition['title']!r}")
        return None
    created = api("POST", "/api/benchmarks", definition)
    bid = created.get("id") or created.get("benchmark", {}).get("id")
    print(f"  created benchmark {bid}: {definition['title']}")
    return bid


def existing_submissions(bid: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        page = api("GET", f"/api/benchmarks/{bid}/submissions?limit=500&offset={offset}")
        rows = page.get("submissions", page if isinstance(page, list) else [])
        for r in rows:
            out[r["title"]] = r
        if len(rows) < 500:
            return out
        offset += 500


def digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]


def submit_all(bid: str | None, entries: list[dict[str, Any]], base: str, args: argparse.Namespace) -> list[str]:
    """entries: {"title", "submitted_by", "payload"}; returns the ids of new submissions."""
    have = existing_submissions(bid) if bid else {}
    new_ids = []
    for e in entries:
        title = f"{e['title']} [{digest(e['payload'])}]"[:200]
        old = have.get(title)
        if old and not (args.retry_failed and str(old.get("status", "")).lower() in FAILED_STATES):
            print(f"  skip (already submitted, {old.get('status')}): {title}")
            continue
        if args.dry_run or bid is None:
            print(f"  [dry-run] would submit: {title}")
            continue
        sub = api("POST", f"/api/benchmarks/{bid}/submissions",
                  {"title": title, "submitted_by": e["submitted_by"], "base_checkpoint_url": base, "payload": e["payload"]})
        sid = sub.get("id") or sub.get("submission", {}).get("id")
        print(f"  submitted {sid}: {title}")
        new_ids.append(sid)
    return new_ids


def wait_for(ids: list[str], timeout_s: float = 600) -> None:
    deadline = time.monotonic() + timeout_s
    pending = set(ids)
    while pending and time.monotonic() < deadline:
        for sid in list(pending):
            s = api("GET", f"/api/submissions/{sid}")
            s = s.get("submission", s)
            if str(s.get("status", "")).lower() in DONE_STATES:
                pending.discard(sid)
        if pending:
            time.sleep(2)
    if pending:
        print(f"  still verifying after {timeout_s:.0f} s: {sorted(pending)}")


def print_leaderboard(bid: str, title: str) -> None:
    board = api("GET", f"/api/benchmarks/{bid}/leaderboard")
    rows = board.get("entries", board.get("leaderboard", []))
    print(f"\n== {title}  ({bid}; {board.get('total', len(rows))} verified)")
    for r in rows:
        print(f"  #{r.get('rank')}  {r.get('score')}  {r.get('title')}")
    subs = existing_submissions(bid)
    unranked = [s for s in subs.values() if s["title"] not in {r.get("title") for r in rows}]
    for s in unranked:
        print(f"  (not ranked: {s.get('status')})  {s['title']}")


# ---------------------------------------------------------------- data hall

def datahall_definition() -> dict[str, Any]:
    rack = {"type": "object", "properties": {
        "id": {"type": "string"}, "model": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"},
        "rot": {"type": "number"}, "cdu": {"type": ["string", "null"]}, "feed_a": {"type": "string"},
        "feed_b": {"type": "string"}}, "required": ["id", "model", "x", "y"]}
    return {
        "title": "GPU data hall layout: Hall B (2N power, N+1 cooling)",
        "prompt": ("Place GPU racks in the fixed 2N AI data hall 'Hall B' (donetwin/scenarios/data_hall_2n.json) and "
                   "assign each rack a cooling unit and two power feeds. Maximize deployed IT load while surviving "
                   "single CDU failures. Hard rules (any violation scores 0): no collisions, in bounds, walkways and "
                   "keepouts clear, 1.2 m aisles, liquid racks on a CDU and air racks off it, CDU capacity, hose "
                   "reach, two feeds per rack within tap reach, 2N power, room air-cooling capacity, hot/cold aisles. "
                   "Score = 70 x density + 20 x resilience + 10 x hose, from 0 to 100, computed by "
                   "donetwin/donetwin/score.py with no LLM in the path."),
        "submission_instructions": ("Submit only decisions: {\"racks\": [{\"id\", \"model\", \"x\", \"y\", \"rot\", "
                                    "\"cdu\", \"feed_a\", \"feed_b\"}]}. The verifier rebuilds the hall from the "
                                    "scenario and the rack catalog, so physical quantities cannot be submitted."),
        "submission_schema": {"type": "object", "properties": {"racks": {"type": "array", "maxItems": 2000,
                                                                          "items": rack}},
                              "required": ["racks"], "additionalProperties": False},
        "score_name": "Hall score (0-100)",
        "score_direction": "higher",
        "score_min": 0,
        "score_max": 100,
        "verifier": {"url": "http://127.0.0.1:9000/evaluations", "method": "POST", "mode": "synchronous",
                     "score_path": "score", "feedback_path": "feedback", "timeout_seconds": 300},
    }


def datahall_entries() -> list[dict[str, Any]]:
    from donetwin import greedy

    hall = json.loads(HALL_B.read_text())
    entries = [{"title": "greedy (no LLM)", "submitted_by": "donetwin greedy baseline", "payload": greedy.solve(hall)}]
    for f, d in run_solutions("datahall"):
        if d.get("hall") != hall["name"]:
            print(f"  no verifier for hall {d.get('hall')!r} (9000 is bound to {hall['name']}): {f}")
            continue
        entries.append({"title": f"{d.get('config', 'run')}, {run_name(f)}", "submitted_by": run_name(f),
                        "payload": d["solution"]})
    return entries


# ---------------------------------------------------------------- QAPLIB

def qaplib_heldout() -> list[str]:
    return json.loads((ROOT / "benchmarks" / "qaplib" / "splits.json").read_text())["heldout"]


def qaplib_definition() -> dict[str, Any]:
    names = qaplib_heldout()
    sol = {"type": "object", "properties": {"instance": {"type": "string", "enum": names},
                                            "permutation": {"type": "array", "items": {"type": "integer"},
                                                            "minItems": 1, "maxItems": 256}},
           "required": ["instance", "permutation"], "additionalProperties": False}
    return {
        "title": f"QAPLIB held-out set ({len(names)} instances): mean BKS / cost",
        "prompt": ("Quadratic Assignment Problem on the frozen QAPLIB held-out split (benchmarks/qaplib/splits.json): "
                   + ", ".join(names) + ". For each instance, assign facility i to location p[i] to minimize "
                   "sum_ij A[i][j] * B[p[i]][p[j]]. The verifier recomputes every cost exactly from the instance "
                   "file. Score = mean over all held-out instances of BKS / cost (1.0 = best known solution, "
                   "above 1.0 = a new best); a missing or invalid permutation scores 0 for that instance."),
        "submission_instructions": ("Submit {\"solutions\": [{\"instance\": name, \"permutation\": [...]}]} with "
                                    "one permutation per held-out instance, 0-based (p[i] = location of facility i; "
                                    "1-based is also accepted)."),
        "submission_schema": {"type": "object", "properties": {"solutions": {"type": "array", "items": sol,
                                                                              "minItems": len(names),
                                                                              "maxItems": len(names)}},
                              "required": ["solutions"], "additionalProperties": False},
        "score_name": "Mean BKS / cost",
        "score_direction": "higher",
        "score_min": 0,
        "verifier": {"url": "http://127.0.0.1:9001/evaluations", "method": "POST", "mode": "synchronous",
                     "request_template": {"submission": "{{submission.payload}}", "split": "heldout",
                                          "submission_id": "{{submission.id}}", "benchmark_id": "{{benchmark.id}}"},
                     "score_path": "score", "feedback_path": "feedback", "timeout_seconds": 300},
    }


def _two_swap(name: str) -> tuple[str, list[int]]:
    import numpy as np

    from benchmarks.qaplib.baseline import two_swap
    from benchmarks.qaplib.verify import INSTANCES, parse_dat

    _, a, b = parse_dat(INSTANCES / f"{name}.dat")
    perm, _ = two_swap(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64), QAP_TIME_LIMIT_S, 0)
    return name, perm


def qaplib_entries() -> list[dict[str, Any]]:
    from benchmarks.qaplib.verify import INSTANCES, parse_dat

    names = qaplib_heldout()
    rand = []
    for name in names:  # exactly baseline.evaluate's random baseline
        n = parse_dat(INSTANCES / f"{name}.dat")[0]
        perm = list(range(n))
        random.Random(0).shuffle(perm)
        rand.append({"instance": name, "permutation": perm})
    cached = json.loads(QAP_TWO_SWAP_CACHE.read_text()) if QAP_TWO_SWAP_CACHE.is_file() else {}
    missing = [n for n in names if n not in cached]
    if missing:
        print(f"  running the 2-swap baseline on {len(missing)} held-out instances ({QAP_TIME_LIMIT_S:g} s each)...")
        with ProcessPoolExecutor(max_workers=min(8, len(missing))) as pool:
            cached.update(dict(pool.map(_two_swap, missing)))
        CACHE.mkdir(parents=True, exist_ok=True)
        QAP_TWO_SWAP_CACHE.write_text(json.dumps(cached))
    two = [{"instance": n, "permutation": cached[n]} for n in names]
    entries = [
        {"title": "random permutation (no LLM, seed 0)", "submitted_by": "qaplib baseline.py",
         "payload": {"solutions": rand}},
        {"title": f"2-swap baseline (no LLM, {QAP_TIME_LIMIT_S:g} s/instance, seed 0)",
         "submitted_by": "qaplib baseline.py", "payload": {"solutions": two}},
    ]
    by_config: dict[tuple[str, str], dict[str, list[int]]] = {}
    for f, d in run_solutions("qaplib"):
        s = d.get("solution") or {}
        if s.get("instance") in names:
            by_config.setdefault((run_name(f), d.get("config", "run")), {})[s["instance"]] = s.get("permutation")
    for (run, config), perms in sorted(by_config.items()):
        if len(perms) < len(names):
            print(f"  {run} {config}: only {len(perms)}/{len(names)} held-out permutations saved; not submitted")
            continue
        entries.append({"title": f"{config}, {run}", "submitted_by": run,
                        "payload": {"solutions": [{"instance": n, "permutation": perms[n]} for n in names]}})
    return entries


# ---------------------------------------------------------------- ALE-Bench AHC039

def alebench_definition() -> dict[str, Any]:
    return {
        "title": "ALE-Bench AHC039 Purse Seine Fishing: 5 public seeds",
        "prompt": ("AtCoder Heuristic Contest 039 (Purse Seine Fishing), from SakanaAI ALE-Bench. Write one complete "
                   "Python 3.10 program (standard library and numpy) that reads an instance on stdin and prints a "
                   "polygon net on stdout within the 2.0 s contest time limit. The verifier runs the program on the "
                   "5 public Lite seeds (0-4) and scores each output with the contest's official visualizer; the "
                   "score is the total over the 5 seeds. Seeds 0-4 are also the evolve loop's train0 and feedback "
                   "seeds, so this leaderboard is in-sample for evolved programs; the held-out (150 private seeds) "
                   "result is in each run's report."),
        "submission_instructions": "Submit {\"program\": \"<complete Python source>\"}.",
        "submission_schema": {"type": "object", "properties": {"program": {"type": "string", "minLength": 1,
                                                                           "maxLength": 1000000}},
                              "required": ["program"], "additionalProperties": False},
        "score_name": "Total official score (5 public seeds)",
        "score_direction": "higher",
        "score_min": 0,
        "verifier": {"url": "http://127.0.0.1:9002/evaluations", "method": "POST", "mode": "synchronous",
                     "score_path": "score", "feedback_path": "feedback", "timeout_seconds": 600},
    }


def alebench_entries() -> list[dict[str, Any]]:
    progs = ROOT / "benchmarks" / "alebench" / "programs"
    entries = [{"title": "greedy reference program (no LLM)", "submitted_by": "alebench reference",
                "payload": {"program": (progs / "greedy_ahc039.py").read_text()}},
               {"title": "naive reference program (no LLM)", "submitted_by": "alebench reference",
                "payload": {"program": (progs / "naive_ahc039.py").read_text()}}]
    for f in sorted(RUNS.glob("alebench-*/logs/*.py")):
        m = re.match(r"(?P<task>.+)-v(?P<ver>\d+)$", f.stem)
        label = f"{m['task']} best program, v{m['ver']}" if m else f.stem
        entries.append({"title": f"{label}, {run_name(f)}", "submitted_by": run_name(f),
                        "payload": {"program": f.read_text()}})
    for f, d in run_solutions("alebench"):
        program = (d.get("solution") or {}).get("program")
        if isinstance(program, str) and program.strip():
            entries.append({"title": f"{d.get('config', 'run')}, {d.get('task_id', f.stem)}, {run_name(f)}",
                            "submitted_by": run_name(f), "payload": {"program": program}})
    return entries


# ---------------------------------------------------------------- shared

def run_name(f: Path) -> str:
    return f.relative_to(RUNS).parts[0]


def run_solutions(case_study: str) -> list[tuple[Path, dict[str, Any]]]:
    """Solution files a runner saved under runs/<run>/solutions/; none exist until the runners persist them."""
    out = []
    for f in sorted(RUNS.glob("*/solutions/**/*.json")):
        try:
            d = json.loads(f.read_text())
        except ValueError:
            print(f"  unreadable solution file: {f}")
            continue
        if d.get("case_study") == case_study and "solution" in d:
            out.append((f, d))
    return out


CASE_STUDIES = {
    "datahall": (datahall_definition, datahall_entries, "http://127.0.0.1:9000/health"),
    "qaplib": (qaplib_definition, qaplib_entries, "http://127.0.0.1:9001/health"),
    "alebench": (alebench_definition, alebench_entries, "http://127.0.0.1:9002/health"),
}


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(CASE_STUDIES), action="append", help="limit to these case studies")
    ap.add_argument("--checkpoint-ref", default="origin/main", help="git ref for base_checkpoint_url (must be pushed)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true", help="resubmit titles whose earlier verification failed")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for verification before printing boards")
    args = ap.parse_args()
    base = checkpoint_url(args.checkpoint_ref)
    print(f"base_checkpoint_url: {base}")
    print("KernelBench mini: skipped; benchmarks/kernelbench_mini/evaluate.py needs a CUDA host with torch.")
    boards = []
    for name in args.only or list(CASE_STUDIES):
        definition_fn, entries_fn, health = CASE_STUDIES[name]
        definition = definition_fn()
        print(f"\n[{name}] {definition['title']}")
        if not healthy(health):
            print(f"  verifier not reachable at {health}; skipped (start it, then rerun)")
            continue
        bid = find_or_create(definition, args.dry_run)
        new_ids = submit_all(bid, entries_fn(), base, args)
        if new_ids and not args.no_wait:
            wait_for(new_ids)
        if bid:
            boards.append((bid, definition["title"]))
    for bid, title in boards:
        print_leaderboard(bid, title)


if __name__ == "__main__":
    main()
