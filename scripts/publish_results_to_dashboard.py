"""Publish every case study's solutions to SA's dashboard as verified leaderboard entries.

  .venv/bin/python -I scripts/publish_results_to_dashboard.py [--dry-run] [--retry-failed] [--no-wait]
      [--only NAME] [--resubmit-outdated]

Idempotent and re-runnable as live runs finish:
- Benchmarks are found by exact title before any is created (definitions are immutable on the dashboard).
- Each submission title carries a short hash of its payload; a payload already on the benchmark under the same
  name is skipped.
- Titles say how each entry compares to the board's baseline ("v1 + brain, run: +1.4 vs greedy"; the baseline
  itself is titled "Baseline: ..."). The delta comes from asking the same verifier before submitting; the
  dashboard then verifies the entry again and stores the verifier's plain-English feedback.
- Boards are the "(vs baseline)" versions, whose descriptions open with one sentence naming the baseline and
  what a higher score means. The earlier boards without that sentence are left as they were.
- Scores are never sent. The dashboard forwards each payload to the case study's verifier service, which
  recomputes the score from the solution itself.

Verifiers it expects (started separately; see each file's docstring):
  data hall  http://127.0.0.1:9000/evaluations  donetwin/verifier_server.py, bound to Hall B (data_hall_2n)
  QAPLIB     http://127.0.0.1:9001/evaluations  benchmarks/qaplib/verifier_api.py
  AHC039     http://127.0.0.1:9002/evaluations  benchmarks/alebench/verifier_api.py
  held-out   http://127.0.0.1:9003/evaluations  benchmarks/datahall/verifier_api.py (any generated hall by seed, tier)
  KernelBench http://127.0.0.1:9005/evaluations benchmarks/kernelbench_mini/verifier_api.py (needs KB_REMOTE or a GPU)

Solutions it finds:
  data hall  the greedy layout for Hall B (recomputed; deterministic)
  QAPLIB     the random and 2-swap baselines on the held-out split (2-swap cached in runs/dashboard/)
  AHC039     the greedy and naive reference programs, plus every runs/alebench-*/logs/*.py the team runner saved
  held-out   per run, the greedy layouts of its held-out halls (or of load_tasks()'s held-out split when no
             run has saved one), then each config's held-out layouts, as one {"halls": [...]} entry per (run, config)
  all        any run solution file runs/<run>/solutions/<config slug>/<task id>.json, written by team.evolve from
             RunResult.solution:
               {"case_study": "datahall" | "qaplib" | "alebench" | "kernelbench", "task_id": str,
                "config": "v0 + final brain", "version": int, "split": "train" | "dev" | "heldout", "score": float,
                "hall", "seed", "tier": <data hall only>,
                "solution": {"racks": [...]} | {"instance": str, "permutation": [int]} | {"program": str}
                            | {"model_new": str}}
             A Hall B layout goes to the 9000 board and a generated hall's held-out layout to the 9003 board; a
             QAPLIB config only once it covers every held-out instance; AHC039 only held-out configs (the board's
             seeds are public); a KernelBench config once it covers every held-out kernel (until a GPU VM is
             attached its verifier answers "Needs a GPU VM, not yet attached" and the entry shows as failed;
             rerun with --retry-failed once it is attached). The KernelBench baseline is PyTorch eager itself,
             speedup 1.0 by definition.
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
        for r in rows:  # newest first: keep the latest row when a title was resubmitted
            out.setdefault(r["title"], r)
        if len(rows) < 500:
            return out
        offset += 500


def digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]


BASELINE_PREFIX = "Baseline: "


def verifier_body(definition: dict[str, Any], payload: Any) -> Any:
    """The body the dashboard would send: the request template with the payload in, other placeholders left out."""
    def fill(v: Any) -> Any:
        if v == "{{submission.payload}}":
            return payload
        if isinstance(v, dict):
            return {k: fill(x) for k, x in v.items() if not (isinstance(x, str) and x.startswith("{{"))
                    or x == "{{submission.payload}}"}
        return v
    return fill(definition["verifier"].get("request_template") or {"submission": "{{submission.payload}}"})


def prescore(definition: dict[str, Any], payload: Any, cache: dict[str, Any]) -> dict[str, Any]:
    key = digest(payload)
    if key not in cache:
        req = urllib.request.Request(definition["verifier"]["url"], method="POST",
                                     data=json.dumps(verifier_body(definition, payload)).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=definition["verifier"].get("timeout_seconds", 600)) as resp:
            cache[key] = json.load(resp)
    return cache[key]


def entry_title(e: dict[str, Any], entries: list[dict[str, Any]], definition: dict[str, Any],
                baseline_label: tuple[str, str] | None, cache: dict[str, Any]) -> str:
    """'Baseline: <title>' for the baseline, else '<title>: +1.4 vs greedy' from the verifier's own score."""
    if e.get("baseline"):
        return BASELINE_PREFIX + e["title"]
    if e.get("plain") or not baseline_label:  # a reframed row, or a case study without a BASELINE_LABELS entry
        return e["title"]
    label, fmt = baseline_label
    try:
        res = prescore(definition, e["payload"], cache)
        base = res.get("baseline") or {}
        bscore = base.get("score")
        if bscore is None:  # a verifier that does not report a baseline (Rikin's Hall B): score the baseline entry
            b = next((x for x in entries if x.get("baseline") and x.get("group") == e.get("group")), None)
            bscore = prescore(definition, b["payload"], cache).get("score") if b else None
        if bscore is None or res.get("score") is None:
            return e["title"]
        return f"{e['title']}: {format(float(res['score']) - float(bscore), fmt)} vs {label}"
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        print(f"  could not pre-score {e['title']!r} for its title ({exc}); submitted without the delta")
        return e["title"]


def outdated(old: dict[str, Any], e: dict[str, Any], definition: dict[str, Any], cache: dict[str, Any]) -> bool:
    """True when the verifier now explains this payload differently from the feedback stored on the board."""
    try:
        return prescore(definition, e["payload"], cache).get("feedback") != old.get("feedback")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"  could not re-check {old['title']!r} ({exc}); left as is")
        return False


def submit_all(bid: str | None, entries: list[dict[str, Any]], base: str, args: argparse.Namespace,
               definition: dict[str, Any] | None = None, baseline_label: tuple[str, str] | None = None) -> list[str]:
    """entries: {"title", "submitted_by", "payload", "baseline"?, "group"?}; returns the ids of new submissions."""
    have = existing_submissions(bid) if bid else {}
    cache: dict[str, Any] = {}
    new_ids = []
    for e in entries:
        dg = digest(e["payload"])
        old = next((r for t, r in have.items() if t.endswith(f"[{dg}]")
                    and (t.startswith(e["title"]) or t.startswith(BASELINE_PREFIX + e["title"]))), None)
        if old and not (args.retry_failed and str(old.get("status", "")).lower() in FAILED_STATES):
            if not (getattr(args, "resubmit_outdated", False) and definition and outdated(old, e, definition, cache)):
                print(f"  skip (already submitted, {old.get('status')}): {old['title']}")
                continue
            print(f"  resubmit (verifier feedback changed since): {old['title']}")
        name = entry_title(e, entries, definition, baseline_label, cache) if definition else e["title"]
        title = f"{name[:189]} [{dg}]"
        if args.dry_run or bid is None:
            print(f"  [dry-run] would submit: {title}")
            continue
        # Re-read the board right before posting: another publisher may have posted this exact title since.
        latest = existing_submissions(bid).get(title)
        if latest and not (args.retry_failed and str(latest.get("status", "")).lower() in FAILED_STATES):
            print(f"  skip (title already on the board, {latest.get('status')}): {title}")
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
        "title": "GPU data hall layout: Hall B (2N power, N+1 cooling) (vs baseline)",
        "prompt": ("Baseline: donetwin's greedy layout (no LLM), titled 'Baseline: greedy'; a higher score means more "
                   "GPU power deployed in the hall while it keeps running through a cooling-unit failure.\n\n"
                   "Place GPU racks in the fixed 2N AI data hall 'Hall B' (donetwin/scenarios/data_hall_2n.json) and "
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
    entries = [{"title": "greedy (no LLM)", "submitted_by": "donetwin greedy baseline", "payload": greedy.solve(hall),
                "baseline": True}]
    for f, d in run_solutions("datahall"):
        if d.get("hall") != hall["name"]:
            continue  # a generated hall: scored by the 9003 verifier on the held-out board
        entries.append({"title": f"{d.get('config', 'run')}, {run_name(f)}", "submitted_by": run_name(f),
                        "payload": d["solution"]})
    return entries


# ---------------------------------------------------------------- data hall, held-out generated halls

def datahall_heldout_definition() -> dict[str, Any]:
    from donetwin import halls

    seeds = halls.heldout_seeds()
    layout = datahall_definition()["submission_schema"]["properties"]["racks"]
    item = {"type": "object", "properties": {
        "seed": {"type": "integer", "minimum": seeds[0], "maximum": seeds[-1]},
        "tier": {"type": "string", "enum": sorted(halls.TIERS)},
        "layout": {"type": "object", "properties": {"racks": layout}, "required": ["racks"],
                   "additionalProperties": False}},
        "required": ["seed", "tier", "layout"], "additionalProperties": False}
    return {
        "title": "GPU data hall, held-out (vs baseline)",
        "prompt": ("Baseline: donetwin's greedy layout (no LLM) on the same halls, one 'Baseline: greedy' row per run; "
                   "a higher score means more GPU power placed safely across the halls.\n\n"
                   f"Place GPU racks in generated 2N AI data halls from donetwin's held-out seeds ({seeds[0]}-{seeds[-1]}; "
                   "never used for training or Gate decisions). Each hall is rebuilt by the verifier from (seed, tier) "
                   "with donetwin/donetwin/halls.py generate(seed, tier) and scored by donetwin/donetwin/score.py "
                   "(same hard rules as Hall B; any violation scores 0; score = 70 x density + 20 x resilience + "
                   "10 x hose, 0-100). Score = mean hall score over the submitted halls. Entries are comparable "
                   "within one run (same held-out halls): each run's greedy baseline is listed beside its configs."),
        "submission_instructions": ("Submit {\"halls\": [{\"seed\": int, \"tier\": \"small\" | \"medium\" | \"large\", "
                                    "\"layout\": {\"racks\": [...]}}]}, one entry per hall, no duplicates."),
        "submission_schema": {"type": "object", "properties": {"halls": {"type": "array", "items": item, "minItems": 1,
                                                                          "maxItems": 64}},
                              "required": ["halls"], "additionalProperties": False},
        "score_name": "Mean hall score (0-100)",
        "score_direction": "higher",
        "score_min": 0,
        "score_max": 100,
        "verifier": {"url": "http://127.0.0.1:9003/evaluations", "method": "POST", "mode": "synchronous",
                     "score_path": "score", "feedback_path": "feedback", "timeout_seconds": 600},
    }


def _greedy_halls(keys: list[tuple[int, str]]) -> list[dict[str, Any]]:
    from donetwin import greedy, halls

    return [{"seed": seed, "tier": tier, "layout": greedy.solve(halls.generate(seed, tier))} for seed, tier in keys]


def datahall_heldout_entries() -> list[dict[str, Any]]:
    by_config: dict[tuple[str, str], dict[tuple[int, str], Any]] = {}
    for f, d in run_solutions("datahall"):
        if d.get("split") == "heldout" and isinstance(d.get("seed"), int) and d.get("tier"):
            by_config.setdefault((run_name(f), d.get("config", "run")), {})[(d["seed"], d["tier"])] = d["solution"]
    runs: dict[str, list[tuple[int, str]]] = {}
    for (run, _), layouts in by_config.items():
        runs[run] = sorted(set(runs.get(run, [])) | set(layouts))
    entries = []
    if not runs:
        from team.datahall import load_tasks

        held = load_tasks()[2]
        entries.append({"title": f"greedy (no LLM), load_tasks() held-out split ({len(held)} halls)",
                        "submitted_by": "donetwin greedy baseline", "baseline": True,
                        "payload": {"halls": [{"seed": t.payload["seed"], "tier": t.payload["tier"],
                                               "layout": t.payload["greedy_layout"]} for t in held]}})
    for run, keys in sorted(runs.items()):
        entries.append({"title": f"greedy (no LLM), {run} held-out halls ({len(keys)})", "group": run,
                        "submitted_by": "donetwin greedy baseline", "baseline": True,
                        "payload": {"halls": _greedy_halls(keys)}})
    for (run, config), layouts in sorted(by_config.items()):
        if len(layouts) < len(runs[run]):
            print(f"  {run} {config}: only {len(layouts)}/{len(runs[run])} held-out halls saved; not submitted")
            continue
        entries.append({"title": f"{config}, {run}", "submitted_by": run, "group": run,
                        "payload": {"halls": [{"seed": k[0], "tier": k[1], "layout": layouts[k]} for k in runs[run]]}})
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
        "title": f"QAPLIB held-out set ({len(names)} instances): mean BKS / cost (vs baseline)",
        "prompt": (f"Baseline: 2-swap local search (no LLM, {QAP_TIME_LIMIT_S:g} s per problem); a higher score means "
                   "cheaper assignments, and 1.0 means every answer matches the best known cost.\n\n"
                   "Quadratic Assignment Problem on the frozen QAPLIB held-out split (benchmarks/qaplib/splits.json): "
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
        {"title": f"2-swap local search (no LLM, {QAP_TIME_LIMIT_S:g} s/instance, seed 0)",
         "submitted_by": "qaplib baseline.py", "payload": {"solutions": two}, "baseline": True},
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


# ---------------------------------------------------------------- GPU rack placement (data-center QAP)
# benchmarks/qaplib/dc_instances.py: QAP instances built from generated DoneTwin halls, scored by the QAPLIB verifier's
# /dc/evaluations route against stored reference costs. The baseline is each instance's stored 2-swap permutation.

def dc_qap_definition() -> dict[str, Any]:
    from benchmarks.qaplib import dc_instances as dc

    names = dc.load_splits()["heldout"]
    index = dc.load_index()
    sol = {"type": "object", "properties": {"instance": {"type": "string", "enum": names},
                                            "permutation": {"type": "array", "items": {"type": "integer"},
                                                            "minItems": 1, "maxItems": 64}},
           "required": ["instance", "permutation"], "additionalProperties": False}
    halls = ", ".join(f"{n} ({index[n]['n']} racks)" for n in names)
    return {
        "title": f"GPU rack placement (data-center QAP): {len(names)} held-out halls, mean reference / cost (vs baseline)",
        "prompt": (f"Baseline: 2-swap local search (no LLM, {dc.TWO_SWAP_TIME_S:g} s per hall); a higher score means "
                   "less training traffic crossing long cables, and 1.0 means every placement matches the reference "
                   "cost.\n\n"
                   "Place the n GPU racks of one AI training job into the n rack slots of a generated data hall "
                   "(DoneTwin halls, greedy layout slots) to minimize sum_ij traffic[i][j] x cable_distance[slot(i)]"
                   "[slot(j)]: a Koopmans-Beckmann QAP in QAPLIB's .dat format (benchmarks/qaplib/dc_instances/). "
                   "Traffic (GB/s) follows tensor-parallel rack pairs (heavy), pipeline stages (medium) and "
                   "data-parallel all-reduce rings (light); distance is Manhattan cable-tray decimetres. Held-out "
                   f"halls: {halls}. These are fresh instances, so no answer has been published. The verifier "
                   "recomputes every cost exactly; score = mean over the held-out halls of reference / cost, where the "
                   "reference is the best of a time-boxed robust tabu search and 2-swap (above 1.0 beats it); a "
                   "missing or invalid permutation scores 0 for that hall."),
        "submission_instructions": ("Submit {\"solutions\": [{\"instance\": name, \"permutation\": [...]}]} with "
                                    "one permutation per held-out hall, 0-based (p[i] = slot of rack i; 1-based is "
                                    "also accepted)."),
        "submission_schema": {"type": "object", "properties": {"solutions": {"type": "array", "items": sol,
                                                                              "minItems": len(names),
                                                                              "maxItems": len(names)}},
                              "required": ["solutions"], "additionalProperties": False},
        "score_name": "Mean reference / cost",
        "score_direction": "higher",
        "score_min": 0,
        "verifier": {"url": "http://127.0.0.1:9001/dc/evaluations", "method": "POST", "mode": "synchronous",
                     "request_template": {"submission": "{{submission.payload}}", "split": "heldout",
                                          "submission_id": "{{submission.id}}", "benchmark_id": "{{benchmark.id}}"},
                     "score_path": "score", "feedback_path": "feedback", "timeout_seconds": 300},
    }


def dc_qap_entries() -> list[dict[str, Any]]:
    from benchmarks.qaplib import dc_instances as dc

    names = dc.load_splits()["heldout"]
    index = dc.load_index()
    rand = []
    for name in names:  # the same seeded random baseline as QAPLIB's
        perm = list(range(index[name]["n"]))
        random.Random(0).shuffle(perm)
        rand.append({"instance": name, "permutation": perm})
    two = [{"instance": n, "permutation": index[n]["two_swap"]["perm"]} for n in names]
    entries = [
        {"title": "random placement (no LLM, seed 0)", "submitted_by": "dc_instances.py", "payload": {"solutions": rand}},
        {"title": f"2-swap local search (no LLM, {dc.TWO_SWAP_TIME_S:g} s/hall, seed 0)", "submitted_by": "dc_instances.py",
         "payload": {"solutions": two}, "baseline": True},
    ]
    by_config: dict[tuple[str, str], dict[str, list[int]]] = {}
    for f, d in run_solutions("qaplib"):
        s = d.get("solution") or {}
        if s.get("instance") in names:
            by_config.setdefault((run_name(f), d.get("config", "run")), {})[s["instance"]] = s.get("permutation")
    for (run, config), perms in sorted(by_config.items()):
        if len(perms) < len(names):
            print(f"  {run} {config}: only {len(perms)}/{len(names)} held-out placements saved; not submitted")
            continue
        entries.append({"title": f"{config}, {run}", "submitted_by": run,
                        "payload": {"solutions": [{"instance": n, "permutation": perms[n]} for n in names]}})
    return entries


# ---------------------------------------------------------------- QAPLIB, the live runs' 6 unseen instances

QAP_UNSEEN_RUN = "qaplib-evolve-3"  # the split is read from this run's saved held-out solutions, never typed in


def qaplib_unseen_names() -> list[str]:
    return sorted({d["solution"]["instance"] for f, d in run_solutions("qaplib")
                   if run_name(f) == QAP_UNSEEN_RUN and d.get("split") == "heldout"
                   and isinstance((d.get("solution") or {}).get("instance"), str)})


def qaplib_unseen_definition() -> dict[str, Any] | None:
    names = qaplib_unseen_names()
    if not names:
        print(f"  runs/{QAP_UNSEEN_RUN} has no saved held-out solutions here; board skipped")
        return None
    sol = {"type": "object", "properties": {"instance": {"type": "string", "enum": names},
                                            "permutation": {"type": "array", "items": {"type": "integer"},
                                                            "minItems": 1, "maxItems": 256}},
           "required": ["instance", "permutation"], "additionalProperties": False}
    return {
        "title": f"QAPLIB held-out ({len(names)} unseen instances)",
        "prompt": (f"Baseline: 2-swap local search (no LLM, {QAP_TIME_LIMIT_S:g} s per problem) on the same "
                   f"{len(names)} problems; a higher score means cheaper assignments, and 1.0 means every answer "
                   "matches the best known cost.\n\n"
                   "Quadratic Assignment Problem on exactly the held-out instances the team's live runs were scored "
                   f"on ({QAP_UNSEEN_RUN} onward): " + ", ".join(names) + ". They come from the frozen held-out split "
                   "(benchmarks/qaplib/splits.json) and were never used for training or dev. For each instance, "
                   "assign facility i to location p[i] to minimize sum_ij A[i][j] * B[p[i]][p[j]]; the verifier "
                   "recomputes every cost exactly from the instance file. Score = mean over these instances of "
                   "BKS / cost (1.0 = best known solution); an invalid permutation scores 0 for that instance."),
        "submission_instructions": (f"Submit {{\"solutions\": [{{\"instance\": name, \"permutation\": [...]}}]}} with "
                                    f"exactly one permutation for each of the {len(names)} instances, 0-based "
                                    "(p[i] = location of facility i; 1-based is also accepted)."),
        "submission_schema": {"type": "object", "properties": {"solutions": {"type": "array", "items": sol,
                                                                              "minItems": len(names),
                                                                              "maxItems": len(names)}},
                              "required": ["solutions"], "additionalProperties": False},
        "score_name": "Mean BKS / cost",
        "score_direction": "higher",
        "score_min": 0,
        # No "split": the mean is over the submitted instances, and the schema plus the verifier's duplicate
        # check make those exactly the instances above.
        "verifier": {"url": "http://127.0.0.1:9001/evaluations", "method": "POST", "mode": "synchronous",
                     "request_template": {"submission": "{{submission.payload}}",
                                          "submission_id": "{{submission.id}}", "benchmark_id": "{{benchmark.id}}"},
                     "score_path": "score", "feedback_path": "feedback", "timeout_seconds": 300},
    }


QAP_SIX_BOARD = "721ba51c-b868-405a-8c34-b9ed4e8beb39"  # created when qaplib-evolve-3 had 6 held-out instances


def qaplib_six_definition() -> dict[str, Any] | None:
    """The existing 6-instance board, read back from the dashboard (its runs have since filled 18 instances)."""
    try:
        return api("GET", f"/api/benchmarks/{QAP_SIX_BOARD}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"  could not read board {QAP_SIX_BOARD} ({exc}); skipped")
        return None


def qaplib_six_entries() -> list[dict[str, Any]]:
    schema = qaplib_six_definition()["submission_schema"]
    return qaplib_unseen_entries(schema["properties"]["solutions"]["items"]["properties"]["instance"]["enum"])


def qaplib_unseen_entries(names: list[str] | None = None) -> list[dict[str, Any]]:
    names = names or qaplib_unseen_names()
    cached = json.loads(QAP_TWO_SWAP_CACHE.read_text()) if QAP_TWO_SWAP_CACHE.is_file() else {}
    if not all(n in cached for n in names):
        qaplib_entries()  # computes and caches the 2-swap baseline for the whole held-out split
        cached = json.loads(QAP_TWO_SWAP_CACHE.read_text())
    entries = [{"title": f"2-swap local search (no LLM, {QAP_TIME_LIMIT_S:g} s/instance, seed 0)",
                "submitted_by": "qaplib baseline.py", "baseline": True,
                "payload": {"solutions": [{"instance": n, "permutation": cached[n]} for n in names]}}]
    by_config: dict[tuple[str, str], dict[str, list[int]]] = {}
    for f, d in run_solutions("qaplib"):
        s = d.get("solution") or {}
        if d.get("split") == "heldout" and isinstance(s.get("instance"), str):
            by_config.setdefault((run_name(f), d.get("config", "run")), {})[s["instance"]] = s.get("permutation")
    for (run, config), perms in sorted(by_config.items()):
        if not set(names) <= set(perms):
            continue  # another held-out set (the data-center instances); a full-split run covers these 6 too
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
                "payload": {"program": (progs / "greedy_ahc039.py").read_text()}, "baseline": True},
               {"title": "naive reference program (no LLM)", "submitted_by": "alebench reference",
                "payload": {"program": (progs / "naive_ahc039.py").read_text()}}]
    for f in sorted(RUNS.glob("alebench-*/logs/*.py")):
        m = re.match(r"(?P<task>.+)-v(?P<ver>\d+)$", f.stem)
        label = f"{m['task']} best program, v{m['ver']}" if m else f.stem
        entries.append({"title": f"{label}, {run_name(f)}", "submitted_by": run_name(f),
                        "payload": {"program": f.read_text()}})
    seen = {digest(e["payload"]) for e in entries}
    for f, d in run_solutions("alebench"):
        program = (d.get("solution") or {}).get("program")
        if d.get("split", "heldout") != "heldout" or not isinstance(program, str) or not program.strip():
            continue
        if digest({"program": program}) in seen:
            continue
        seen.add(digest({"program": program}))
        entries.append({"title": f"{d.get('config', 'run')}, {d.get('task_id', f.stem)}, {run_name(f)}",
                        "submitted_by": run_name(f), "payload": {"program": program}})
    return entries


# ---------------------------------------------------------------- KernelBench mini

def kernelbench_definition() -> dict[str, Any]:
    from benchmarks.kernelbench_mini import verifier_api as kb

    names = kb.heldout()
    sol = {"type": "object", "properties": {"task": {"type": "string", "enum": names},
                                            "model_new": {"type": "string", "minLength": 1, "maxLength": 200000}},
           "required": ["task", "model_new"], "additionalProperties": False}
    return {
        "title": f"KernelBench mini held-out ({len(names)} GPU kernels): mean speedup vs PyTorch eager (vs baseline)",
        "prompt": ("Baseline: PyTorch eager, speedup 1.0 by definition; a higher score means GPU kernels that run "
                   "faster than PyTorch while giving the same answers.\n\n"
                   "Write a faster ModelNew for each held-out KernelBench mini problem (benchmarks/kernelbench_mini/"
                   "splits.json: " + ", ".join(names) + "). The verifier runs evaluate.py on a CUDA host: static "
                   "checks, correctness on 5 fresh signed input sets (atol=rtol=1e-4), then CUDA-event timing against "
                   "PyTorch eager on the same inputs. Score = mean over the held-out kernels of the speedup, capped at "
                   "10; a missing, wrong or non-compiling kernel scores 0."),
        "submission_instructions": ("Submit {\"solutions\": [{\"task\": \"kb_relu\", \"model_new\": \"<python "
                                    "source defining class ModelNew>\"}]}, one per held-out kernel."),
        "submission_schema": {"type": "object", "properties": {"solutions": {"type": "array", "items": sol,
                                                                              "minItems": 1, "maxItems": len(names)}},
                              "required": ["solutions"], "additionalProperties": False},
        "score_name": "Mean speedup vs PyTorch eager",
        "score_direction": "higher",
        "score_min": 0,
        "score_max": 10,
        "verifier": {"url": "http://127.0.0.1:9005/evaluations", "method": "POST", "mode": "synchronous",
                     "score_path": "score", "feedback_path": "feedback", "timeout_seconds": 3600},
    }


def kernelbench_entries() -> list[dict[str, Any]]:
    from benchmarks.kernelbench_mini import verifier_api as kb

    names = kb.heldout()
    entries = [{"title": "PyTorch eager (no LLM)", "submitted_by": "kernelbench reference", "baseline": True,
                "payload": {"solutions": [{"task": t, "model_new": kb.eager_candidate(t)} for t in names]}}]
    try:
        with urllib.request.urlopen("http://127.0.0.1:9005/health", timeout=5) as resp:
            gpu = json.load(resp).get("gpu")
    except (urllib.error.URLError, OSError, ValueError):
        gpu = None
    if gpu is None:
        n = len(run_solutions("kernelbench"))
        print(f"  no GPU attached: only the PyTorch eager baseline is published ({n} saved run solutions held back)")
        return entries
    by_config: dict[tuple[str, str], dict[str, str]] = {}
    for f, d in run_solutions("kernelbench"):
        src = (d.get("solution") or {}).get("model_new")
        if d.get("task_id") in names and isinstance(src, str) and src.strip():
            by_config.setdefault((run_name(f), d.get("config", "run")), {})[d["task_id"]] = src
    for (run, config), sols in sorted(by_config.items()):
        if len(sols) < len(names):
            print(f"  {run} {config}: only {len(sols)}/{len(names)} held-out kernels saved; not submitted")
            continue
        entries.append({"title": f"{config}, {run}", "submitted_by": run,
                        "payload": {"solutions": [{"task": t, "model_new": sols[t]} for t in names]}})
    return entries


# ---------------------------------------------------------------- inference serving cost

def inference_opt_definition() -> dict[str, Any]:
    from benchmarks.inference_opt.workload import load_tasks

    held = [(t.payload["seed"], t.payload["tier"]) for t in load_tasks()[2]]
    task = {"type": "object", "properties": {"seed": {"type": "integer"}, "tier": {"type": "string"},
                                             "answers": {"type": "object"}},
            "required": ["seed", "tier", "answers"], "additionalProperties": False}
    return {
        "title": f"Inference serving cost: held-out data-center ops Q&A ({len(held)} halls, 48 questions): accuracy (vs baseline)",
        "prompt": ("Baseline: v0, Nemotron Super with reasoning on and one call per question; a higher score means more "
                   "questions answered exactly, and the run's goal is the same accuracy at a lower Crusoe bill.\n\n"
                   "Exact-answer questions about DoneTwin data halls (2N busway budget left, racks that fit a row, CDU "
                   "headroom, hose slack), regenerated from each held-out hall's seed and tier "
                   "(benchmarks/inference_opt/workload.py): " + ", ".join(f"{t}-{s}" for s, t in held) + ". "
                   "Score = fraction of all held-out questions within the stated tolerance; a missing task scores 0."),
        "submission_instructions": ("Submit {\"tasks\": [{\"seed\": int, \"tier\": str, \"answers\": {\"q1\": number, "
                                    "...}}]} with one entry per held-out hall."),
        "submission_schema": {"type": "object", "properties": {"tasks": {"type": "array", "items": task,
                                                                          "minItems": len(held), "maxItems": len(held)}},
                              "required": ["tasks"], "additionalProperties": False},
        "score_name": "Accuracy",
        "score_direction": "higher",
        "score_min": 0,
        "verifier": {"url": "http://127.0.0.1:9004/evaluations", "method": "POST", "mode": "synchronous",
                     "request_template": {"submission": "{{submission.payload}}", "split": "heldout",
                                          "submission_id": "{{submission.id}}", "benchmark_id": "{{benchmark.id}}"},
                     "score_path": "score", "feedback_path": "feedback", "timeout_seconds": 60},
    }


def inference_opt_entries() -> list[dict[str, Any]]:
    """One entry per (run, config) that saved answers for every held-out hall; v0 cold is the baseline."""
    from benchmarks.inference_opt.workload import load_tasks

    held = [(t.payload["seed"], t.payload["tier"]) for t in load_tasks()[2]]
    by_config: dict[tuple[str, str], dict[tuple[int, str], dict[str, Any]]] = {}
    for f, d in run_solutions("inference_opt"):
        s = d.get("solution") or {}
        key = (s.get("seed"), s.get("tier"))
        if key in held:
            by_config.setdefault((run_name(f), d.get("config", "run")), {})[key] = s.get("answers") or {}
    entries = []
    for (run, config), answers in sorted(by_config.items()):
        if len(answers) < len(held):
            print(f"  {run} {config}: only {len(answers)}/{len(held)} held-out halls saved; not submitted")
            continue
        entries.append({"title": f"{config}, {run}", "submitted_by": run, "baseline": config.startswith("v0 + empty"),
                        "payload": {"tasks": [{"seed": s, "tier": t, "answers": answers[(s, t)]} for s, t in held]}})
    return sorted(entries, key=lambda e: not e["baseline"])


# ---------------------------------------------------------------- shared

def run_name(f: Path) -> str:
    return f.relative_to(RUNS).parts[0]


RUN_PREFIXES = {"datahall": "datahall", "qaplib": "qaplib", "alebench": "alebench", "kernelbench": "kernelbench"}


def run_solutions(case_study: str) -> list[tuple[Path, dict[str, Any]]]:
    """Solution files team.evolve saved under runs/<run>/solutions/ (case study from the file, else the run name)."""
    out = []
    for f in sorted(RUNS.glob("*/solutions/**/*.json")):
        try:
            d = json.loads(f.read_text())
        except ValueError:
            print(f"  unreadable solution file: {f}")
            continue
        cs = d.get("case_study") or next((c for pre, c in RUN_PREFIXES.items() if run_name(f).startswith(pre)), None)
        if cs == case_study and "solution" in d:
            out.append((f, d))
    return out


BASELINE_LABELS = {  # (name in titles, format of the score delta)
    "datahall": ("greedy", "+.1f"),
    "datahall_heldout": ("greedy", "+.1f"),
    "qaplib": ("2-swap", "+.4f"),
    "qaplib_unseen6": ("2-swap", "+.4f"),
    "qaplib_dc": ("2-swap", "+.4f"),
    "alebench": ("greedy", "+,.0f"),
    "kernelbench": ("eager", "+.2f"),
    "inference_opt": ("v0", "+.3f"),
}

# v0 is the baseline: each run's v0 cold row (the starting team, empty brain) is titled "Baseline: v0 cold (<run>)"
# and its last version "<run> final". Classical programs are no longer submitted; the "Reference (no LLM): ..." and
# older "Baseline: greedy" rows already on the boards cannot be deleted, so the dashboard hides them.
V0_BOARDS = {"datahall_heldout", "qaplib", "qaplib_unseen6", "qaplib_six", "qaplib_dc", "kernelbench", "inference_opt"}
FINAL_CONFIG = re.compile(r"^v(\d+) \+ final brain$")
UNFINISHED_RUNS = {"qaplib-evolve-5"}  # stopped by hand; only its cold arm reached the held-out set


def reframed(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The v0-baseline rows for one board, built from the entries the case study already produces."""
    out: list[dict[str, Any]] = []
    finals: dict[str, tuple[int, dict[str, Any]]] = {}
    for e in entries:
        config, _, run = e["title"].rpartition(", ")
        run = e.get("group") or run or e["submitted_by"]
        if run in UNFINISHED_RUNS:
            continue
        if "(cold)" in config:
            out.append({**e, "title": f"v0 cold ({run})", "baseline": True})
        elif (m := FINAL_CONFIG.match(config)) and int(m[1]) >= finals.get(run, (-1, None))[0]:
            finals[run] = (int(m[1]), e)
    out += [{**e, "title": f"{run} final", "baseline": False, "plain": True} for run, (_, e) in sorted(finals.items())]
    unique, seen = [], set()
    for e in out:  # one row per title even when several run groups carry the same payload
        key = (e["title"], digest(e["payload"]))
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique


CASE_STUDIES = {
    "datahall": (datahall_definition, datahall_entries, "http://127.0.0.1:9000/health"),
    "qaplib": (qaplib_definition, qaplib_entries, "http://127.0.0.1:9001/health"),
    "qaplib_unseen6": (qaplib_unseen_definition, qaplib_unseen_entries, "http://127.0.0.1:9001/health"),
    "qaplib_six": (qaplib_six_definition, qaplib_six_entries, "http://127.0.0.1:9001/health"),
    "qaplib_dc": (dc_qap_definition, dc_qap_entries, "http://127.0.0.1:9001/health"),
    "alebench": (alebench_definition, alebench_entries, "http://127.0.0.1:9002/health"),
    "datahall_heldout": (datahall_heldout_definition, datahall_heldout_entries, "http://127.0.0.1:9003/health"),
    "kernelbench": (kernelbench_definition, kernelbench_entries, "http://127.0.0.1:9005/health"),
    "inference_opt": (inference_opt_definition, inference_opt_entries, "http://127.0.0.1:9004/health"),
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
    ap.add_argument("--resubmit-outdated", action="store_true",
                    help="resubmit entries whose verifier feedback has changed (adds a new row; the dashboard keeps "
                         "the old one). Use with --only: verifiers that time programs never repeat their feedback")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for verification before printing boards")
    args = ap.parse_args()
    base = checkpoint_url(args.checkpoint_ref)
    print(f"base_checkpoint_url: {base}")
    boards = []
    for name in args.only or list(CASE_STUDIES):
        definition_fn, entries_fn, health = CASE_STUDIES[name]
        definition = definition_fn()
        if definition is None:  # the case study has nothing to define on this machine yet
            print(f"\n[{name}] skipped")
            continue
        print(f"\n[{name}] {definition['title']}")
        if not healthy(health):
            print(f"  verifier not reachable at {health}; skipped (start it, then rerun)")
            continue
        bid = find_or_create(definition, args.dry_run)
        entries = entries_fn()
        if name in V0_BOARDS:
            entries = reframed(entries)
        new_ids = submit_all(bid, entries, base, args, definition, BASELINE_LABELS.get(name))
        if new_ids and not args.no_wait:
            wait_for(new_ids)
        if bid:
            boards.append((bid, definition["title"]))
    for bid, title in boards:
        print_leaderboard(bid, title)


if __name__ == "__main__":
    main()
