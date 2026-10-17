"""Collect every benchmark result from LIVE Crusoe runs into reports/results.json and reports/RESULTS.md.

Usage (repo venv):
  .venv/bin/python scripts/collect_results.py [--root DIR] [--headroom DIR] [--out-dir DIR]

Rerun it whenever a run finishes; it rewrites both files from what is on disk.

Only the live runs named in CASE_STUDIES (plus the live smokes and the model-settings experiment) are read.
Offline and scripted runs are tests: they never appear as results, only by name under "not counted".
A run is "done" only when its usage.json shows paid Crusoe calls (the scripted LLM costs $0 and writes none),
"running" while its ledger grows without a final usage.json, "blocked" when its log ends in an exception or a
final report has no usage.json, and "not started" when nothing is on disk. Partial files are tolerated.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HEADROOM_DIR = Path("/private/tmp/claude-501/-Users-iamdaone-Documents-RecursiveAgentsHackathonProject/"
                    "068a120f-637f-4002-856f-3eb6683d41a7/scratchpad/headroom")
COMMAND = ".venv/bin/python scripts/collect_results.py"
SCHEMA_VERSION = 1
FRESH_S = 15 * 60  # an experiment file touched this recently counts as still running

DATAHALL = dict(verifier="DoneTwin checker: hard feasibility rules (collision, 2N power, CDU capacity, hose reach, "
                         "zones); score = DoneTwin score / 100, pass = feasible",
                baseline_name="greedy layout (no LLM)", digits=3, baseline_kind="datahall")
CASE_STUDIES: list[dict[str, Any]] = [
    dict(id="datahall-live-1", name="Data hall layout, live run 1", dirs=["reports/datahall_live_1"],
         aliases=["runs/datahall-live-1"], log="reports/datahall_live_1/run.log", **DATAHALL),
    dict(id="datahall-live-2", name="Data hall layout, live run 2", dirs=["runs/datahall-live-2"],
         log="runs/datahall-live-2.log", **DATAHALL),
    dict(id="qaplib-evolve-1", name="QAPLIB (quadratic assignment)", dirs=["runs/qaplib-evolve-1"],
         log="runs/qaplib-evolve-1.log", digits=4, baseline_kind="qaplib",
         verifier="exact QAP cost of the printed permutation; score = BKS / cost (1.0 = best known), "
                  "pass = gap to BKS <= 1%",
         baseline_name="2-swap local search, 10 s restarts (no LLM)"),
    dict(id="alebench-live-1", name="ALE-Bench AHC039 (Purse Seine Fishing)", dirs=["runs/alebench-live-1"],
         log="runs/alebench-live-1.log", digits=3, baseline_kind="fixed", baseline_score=1.0,
         verifier="ALE-Bench AHC039 scorer on private Lite seeds; score = mean case score / greedy case score "
                  "(1.0 = greedy); rank vs 583 contestants on all 150 private seeds",
         baseline_name="reference greedy (no LLM)"),
    dict(id="kernelbench-mini", name="KernelBench mini (CUDA kernels)", glob="runs/kernelbench-*",
         digits=2, baseline_kind="fixed", baseline_score=1.0,
         verifier="compile, correctness on 5 signed input sets plus a held-back seed, CUDA-event timing; "
                  "score = speedup over torch eager if correct (cap 10), else 0",
         baseline_name="torch eager (speedup 1.0x)",
         missing_status=("blocked", "needs a Crusoe GPU VM (no runs/kernelbench-* yet)")),
]
SMOKES = ["runs/qaplib-one", "runs/alebench-one", "runs/datahall-one", "runs/kernelbench-one"]
# The connection probe printed to the console only; these figures were copied from that live output.
CONNECTION_PROBE = {
    "name": "Crusoe connection probe", "source": "copied from the console output of scripts/crusoe_smoke.py; no file on disk",
    "rows": [
        {"model": "Nemotron-3 Nano 30B", "latency_s": 0.55, "cost_usd_per_call": 0.000013, "cached_tokens": 0},
        {"model": "Nemotron-3 Super 120B", "latency_s": 2.0, "cost_usd_per_call": 0.000147, "cached_tokens": 0},
    ],
    "reasoning_off": {"tokens_before": 180, "latency_before_s": 2.0, "tokens_after": 7, "latency_after_s": 0.2},
}
VARIANT_FALLBACK = {
    "A": "Nano 30B, reasoning off", "B": "Nano 30B, reasoning on", "B32": "Nano 30B, reasoning on, 32k tokens",
    "C": "Super 120B, reasoning off", "D": "Super 120B, reasoning on", "E": "Super 120B, reasoning on, examples",
    "F": "Super 120B, reasoning off, small-edit playbook", "G": "Nano 30B, reasoning off, small-edit playbook",
}


# ---------------------------------------------------------------- tolerant readers

def read_json(path: Path | None) -> Any:
    """None for a missing, half-written or unreadable file."""
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_jsonl(path: Path | None) -> list[dict]:
    """Every complete JSON line; a line still being written is skipped."""
    if path is None or not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(errors="replace")


def md_tables(text: str) -> list[list[dict[str, str]]]:
    """Each markdown pipe table as a list of {header: cell} rows."""
    tables: list[list[dict[str, str]]] = []
    header: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not (line.startswith("|") and line.endswith("|")):
            header = None
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells
            tables.append([])
        elif not all(set(c) <= set("-: ") for c in cells):
            tables[-1].append(dict(zip(header, cells)))
    return tables


def to_float(s: Any) -> float | None:
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        return float(s)
    if not isinstance(s, str):
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", s)
    if not m:
        return None
    v = float(m.group(0).replace(",", ""))
    return v / 100 if s.strip().endswith("%") else v


def short(s: Any, n: int = 90) -> str:
    text = " ".join(str(s).split())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def stamp(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts).astimezone().strftime("%H:%M %Z")


# ---------------------------------------------------------------- one evolve run

def find_file(dirs: list[Path], names: tuple[str, ...]) -> Path | None:
    for d in dirs:
        for sub in ("", "reports", "versions"):
            for name in names:
                p = (d / sub / name) if sub else d / name
                if p.is_file():
                    return p
    return None


def log_facts(text: str) -> dict[str, Any]:
    lines = [l for l in text.splitlines() if l.strip()]
    last = lines[-1].strip() if lines else ""
    wall = re.search(r"^wall ([\d.]+)s", text, re.M)
    greedy = re.search(r"greedy baseline:.*held-out ([\d.]+) \(of 100\)", text)
    heldout = re.search(r"^held-out: (.*)$", text, re.M)
    estimate = re.search(r"^Estimated LLM calls:.*$", text, re.M)
    crashed = "Traceback (most recent call last)" in text and bool(re.match(r"^[\w.]+(Error|Exception)\b", last))
    return {"wall_s": float(wall.group(1)) if wall else None,
            "greedy_heldout": float(greedy.group(1)) / 100 if greedy else None,
            "heldout_names": re.findall(r"(\w+)\(n=\d+\)", heldout.group(1)) if heldout else [],
            "live_rpm_line": bool(estimate and "RPM" in estimate.group(0)),
            "crash": crash_summary(last) if crashed else None}


def crash_summary(line: str) -> str:
    """'openai.RateLimitError: Error code: 429 - {...Reason: max_requests_limit...}' -> 'RateLimitError, HTTP 429, max_requests_limit'."""
    name = line.split(":", 1)[0].split(".")[-1]
    code = re.search(r"Error code: (\d+)", line)
    reason = re.search(r"Reason: ([\w-]+)", line)
    parts = [name] + ([f"HTTP {code.group(1)}"] if code else []) + ([reason.group(1)] if reason else [])
    return ", ".join(parts) if len(parts) > 1 else short(line, 120)


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def describe_mutation(m: dict) -> str:
    kind = m.get("type", "?")
    if kind == "prompt_edit":
        return f"{m.get('role')} prompt: {short(m.get('text', ''), 70)}"
    if kind == "playbook_add":
        return f"{m.get('role')} step: {short(m.get('step', ''), 70)}"
    if kind == "tool_set":
        return f"tool {m.get('tool')} enabled={m.get('enabled')}"
    if kind == "config_set":
        return f"{m.get('key')} = {json.dumps(m.get('value'))}"
    rest = {k: v for k, v in m.items() if k not in ("type", "claim", "rationale")}
    return short(json.dumps(rest, default=str), 80)


def describe_claim(claim: Any, m: dict) -> str:
    if isinstance(claim, dict) and claim:
        parts = []
        if isinstance(claim.get("dev_gain"), (int, float)):
            parts.append(f"dev gain {claim['dev_gain']:+.3f}")
        if isinstance(claim.get("cost_change_pct"), (int, float)):
            parts.append(f"cost {claim['cost_change_pct']:+.1f}%")
        why = claim.get("why")
        return ", ".join(parts) + (f": {short(why, 70)}" if why else "")
    if m.get("rationale"):
        return "no numeric claim; rationale: " + short(m["rationale"], 60)
    return "none recorded (ledger predates claims)"


def generation_rows(ledger: list[dict]) -> list[dict]:
    rows = []
    for r in ledger:
        m = r.get("mutation") if isinstance(r.get("mutation"), dict) else {}
        paper = r.get("paper") if isinstance(r.get("paper"), dict) else {}
        claim = paper.get("claim") or m.get("claim")
        rows.append({"gen": r.get("generation"), "parent": r.get("parent"), "mutation_type": m.get("type"),
                     "change": describe_mutation(m), "claim": describe_claim(claim, m),
                     "measured_gain": r.get("gain"), "measured_cost_change_pct": r.get("cost_change_pct"),
                     "dev_n": r.get("n"), "accepted": bool(r.get("accepted")),
                     "accepted_by": r.get("accepted_by"), "reason": r.get("reason"),
                     "verdict": paper.get("verdict")})
    return rows


def dev_means(ledger: list[dict], history: list[dict]) -> list[dict]:
    if history:
        return [{"gen": h.get("generation"), "parent": h.get("parent"), "version": h.get("version"),
                 "train_mean": h.get("train_mean"), "dev_mean": h.get("dev_mean"), "accepted": h.get("accepted")}
                for h in history]
    seen: dict[Any, dict] = {}
    for r in ledger:
        if r.get("parent_mean") is not None and r.get("generation") not in seen:
            seen[r["generation"]] = {"gen": r["generation"], "parent": r.get("parent"), "version": None,
                                     "train_mean": None, "dev_mean": r["parent_mean"], "accepted": None}
    return list(seen.values())


def baseline_for(spec: dict, root: Path, vs_md: str, rows_json: Any, facts: dict) -> tuple[float | None, str, str]:
    """(score, name, where it came from)."""
    name = spec["baseline_name"]
    if isinstance(rows_json, list):
        for r in rows_json:
            if isinstance(r, dict) and r.get("version") is None and r.get("mean_score") is not None:
                return float(r["mean_score"]), r.get("config", name), "final held-out table"
    for table in md_tables(vs_md)[:1]:
        for r in table:
            if "no LLM" in r.get("Config", "") and to_float(r.get("Mean score")) is not None:
                return to_float(r["Mean score"]), r["Config"], "final held-out table"
    if spec["baseline_kind"] == "fixed":
        return spec["baseline_score"], name, "1.0 by definition of the score"
    if spec["baseline_kind"] == "datahall" and facts["greedy_heldout"] is not None:
        return facts["greedy_heldout"], name, "run log (greedy is deterministic)"
    if spec["baseline_kind"] == "qaplib":
        data = read_json(root / "reports" / "qaplib_baseline.json") or {}
        inst = data.get("instances") or {}
        names = [n for n in facts["heldout_names"] if n in inst]
        if names:
            return (round(mean(inst[n]["two_swap"]["score"] for n in names), 4), name,
                    f"reports/qaplib_baseline.json on this run's {len(names)} held-out instances")
        s = ((data.get("summary") or {}).get("heldout") or {}).get("two_swap") or {}
        if s.get("mean_score") is not None:
            return s["mean_score"], name, "reports/qaplib_baseline.json, all held-out instances"
    return None, name, "not available yet"


def heldout_rows(ba: Any, rows_json: Any, vs_md: str, base: float | None) -> tuple[list[dict], list[str]]:
    notes: list[str] = []
    rows = rows_json if isinstance(rows_json, list) else (ba or {}).get("rows") if isinstance(ba, dict) else None
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    out, seen = [], set()
    for r in rows:
        key = (r.get("config"), r.get("version"), r.get("mean_score"), r.get("tokens"))
        if key in seen:
            notes.append(f"'{r.get('config')}' appears twice in the final report: no candidate passed the gate, "
                         "so the final version is v0 and its arm repeats 'v0 + final brain'; shown once.")
            continue
        seen.add(key)
        out.append({"config": r.get("config"), "version": r.get("version"), "is_baseline": r.get("version") is None,
                    "n": r.get("n"), "mean_score": r.get("mean_score"), "pass_rate": r.get("pass_rate"),
                    "tokens": r.get("tokens", 0 if r.get("version") is None else None),
                    "cost_usd": r.get("cost_usd", 0.0 if r.get("version") is None else None),
                    "mean_gap_pct": r.get("mean_gap_pct"), "at_bks": r.get("at_bks"), "invalid": r.get("invalid")})
    if out and not any(r["is_baseline"] for r in out):
        for table in md_tables(vs_md)[:1]:
            for r in table:
                if "no LLM" in r.get("Config", ""):
                    out.append({"config": r["Config"], "version": None, "is_baseline": True, "n": out[0]["n"],
                                "mean_score": to_float(r.get("Mean score")), "pass_rate": to_float(r.get("Pass rate")),
                                "tokens": 0, "cost_usd": 0.0, "mean_gap_pct": None, "at_bks": None, "invalid": None})
    for r in out:
        s = r["mean_score"]
        r["vs_baseline"] = None if s is None or base is None else round(s - base, 6)
        r["vs_baseline_pct"] = None if s is None or not base else round((s - base) / base * 100, 3)
    return out, notes


def run_status(dirs: list[Path], log: Path | None, usage: Any, ba: Any, facts: dict) -> tuple[str, str]:
    if not any(d.exists() for d in dirs) and not (log and log.exists()):
        return "not started", "nothing on disk yet"
    total = (usage or {}).get("total") or {}
    if total:
        if (total.get("calls") or 0) > 0 and (total.get("cost_usd") or 0) > 0:
            return ("done", "final report and live usage written") if ba else (
                "running", "usage written, final report missing")
        return "blocked", "usage.json shows $0: a scripted run, not a live result"
    if facts["crash"]:
        return "blocked", f"crashed: {facts['crash']}"
    if ba:
        return "blocked", "final report but no usage.json: offline or scripted, not a live result"
    return "running", "no final report yet"


def sa_library_counts(root: Path, run_ids: list[str], timeout_s: float = 10.0) -> dict[str, dict] | None:
    """Papers and verdicts per run from SA's Library client, when team/sa_library.py exists. Best effort: any
    import error, unknown API or slow server returns None and the section is skipped."""
    if not (root / "team" / "sa_library.py").is_file():
        return None
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            mod = importlib.import_module("team.sa_library")
            client = mod
            for cls in ("SALibrary", "SALibraryClient", "LibraryClient", "Client"):
                if callable(getattr(mod, cls, None)):
                    client = getattr(mod, cls)()
                    break
            fn = next(getattr(client, f) for f in ("papers", "list_papers", "papers_for_run", "query")
                      if callable(getattr(client, f, None)))
            out = {}
            for rid in run_ids:
                papers = list(fn(run_id=rid) or [])
                verdicts = Counter(str(p.get("verdict")) if isinstance(p, dict) else "unknown" for p in papers)
                out[rid] = {"papers": len(papers), "verdicts": dict(verdicts)}
            box["out"] = out
        except Exception:  # noqa: BLE001 - optional source, skip silently
            box["out"] = None

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)
    return box.get("out")


def collect_evolve(spec: dict, root: Path, dirs: list[Path], log: Path | None) -> dict[str, Any]:
    ba_path = find_file(dirs, ("before_after.json",))
    usage_path = find_file(dirs, ("usage.json",))
    vs_path = find_file(dirs, ("vs_greedy.md", "vs_2swap.md", "vs_baseline.md"))
    rows_path = find_file(dirs, ("qaplib_rows.json", "rows.json"))
    rating_path = find_file(dirs, ("rating.json",))
    ledger_path = find_file(dirs, ("version_ledger.jsonl", "ledger.jsonl"))
    if ledger_path and ledger_path.parent.name == "brain":  # the brain's ledger is not the version ledger
        ledger_path = None
    ba, usage, rows_json, rating = map(read_json, (ba_path, usage_path, rows_path, rating_path))
    vs_md, log_text = read_text(vs_path), read_text(log)
    ledger = read_jsonl(ledger_path)
    facts = log_facts(log_text)
    status, detail = run_status(dirs, log, usage, ba, facts)

    base, base_name, base_src = baseline_for(spec, root, vs_md, rows_json, facts)
    rows, notes = heldout_rows(ba, rows_json, vs_md, base) if status == "done" else ([], [])
    team = [r for r in rows if not r["is_baseline"] and r["mean_score"] is not None]
    best = max(team, key=lambda r: (r["mean_score"], -(r["cost_usd"] or 0))) if team else None
    gens = generation_rows(ledger)
    history = (ba or {}).get("history") if isinstance(ba, dict) else None
    verdicts = Counter(g["verdict"] for g in gens if g["verdict"])

    total = (usage or {}).get("total") or {}
    totals = None
    if status == "done" and total:
        pt, ct = total.get("prompt_tokens", 0), total.get("completion_tokens", 0)
        totals = {"calls": total.get("calls"), "errors": total.get("errors"), "prompt_tokens": pt,
                  "completion_tokens": ct, "cached_tokens": total.get("cached_tokens"), "tokens": pt + ct,
                  "cost_usd": total.get("cost_usd"), "wall_s": facts["wall_s"],
                  "by_model": {m: {"calls": v.get("calls"), "cost_usd": v.get("cost_usd")}
                               for m, v in ((usage or {}).get("by_model") or {}).items()}}

    files = [p for p in (ba_path, vs_path, rows_path, rating_path, usage_path, ledger_path, log) if p and p.is_file()]
    mtimes = [p.stat().st_mtime for p in files] + [p.stat().st_mtime for d in dirs if d.exists() for p in d.rglob("*")
                                                   if p.is_file()]
    v0 = find_file(dirs, ("v0.json",))
    started = v0.stat().st_mtime if v0 else (min(mtimes) if mtimes else None)
    last = max(mtimes) if mtimes else None
    if status == "running":
        gen_now = max((g["gen"] for g in gens if isinstance(g["gen"], int)), default=None)
        detail = (f"{plural(len(gens), 'candidate')} scored on dev" + (f", generation {gen_now}" if gen_now is not None else "")
                  + (f"; started {stamp(started)}, last write {stamp(last)}" if started else ""))
    elif status == "blocked" and facts["crash"]:
        detail += f"; {plural(len(gens), 'candidate')} scored before the crash; last write {stamp(last)}"

    live = ("usage.json: " + f"{total.get('calls')} paid calls, ${total.get('cost_usd'):.4f}") if (
        status == "done") else ("log shows the Crusoe RPM cap" if facts["live_rpm_line"] else "named live run")
    rating_rows = rating if isinstance(rating, list) and status == "done" else None
    return {
        "id": spec["id"], "name": spec["name"], "verifier": spec["verifier"], "digits": spec["digits"],
        "baseline": {"name": base_name, "score": base, "source": base_src},
        "status": status, "status_detail": detail, "live_evidence": live,
        "started_at": stamp(started), "last_update": stamp(last),
        "heldout": rows, "best": None if best is None else {
            "config": best["config"], "score": best["mean_score"], "gain": best["vs_baseline"],
            "gain_pct": best["vs_baseline_pct"]},
        "rating": rating_rows, "history": dev_means(ledger, history or []), "generations": gens,
        "papers": {"total": len(gens), "accepted": sum(g["accepted"] for g in gens), "verdicts": dict(verdicts)},
        "totals": totals, "sources": [rel(p, root) for p in files], "notes": notes,
    }


def collect_case(spec: dict, root: Path) -> dict[str, Any]:
    if "glob" in spec:
        # Offline runs land in the same runs/<name>-* pattern; they are tests, so they never stand in for a live run.
        results = [collect_evolve(spec, root, [d], root / "runs" / f"{d.name}.log")
                   for d in sorted(root.glob(spec["glob"])) if d.is_dir()]
        results = [r for r in results if "not a live result" not in r["status_detail"]]
        if not results:
            status, detail = spec["missing_status"]
            return {"id": spec["id"], "name": spec["name"], "verifier": spec["verifier"], "digits": spec["digits"],
                    "baseline": {"name": spec["baseline_name"], "score": spec.get("baseline_score"),
                                 "source": "1.0 by definition of the score"},
                    "status": status, "status_detail": detail, "live_evidence": None, "started_at": None,
                    "last_update": None, "heldout": [], "best": None, "rating": None, "history": [],
                    "generations": [], "papers": {"total": 0, "accepted": 0, "verdicts": {}}, "totals": None,
                    "sources": [], "notes": []}
        order = {"done": 0, "running": 1, "blocked": 2, "not started": 3}
        return sorted(results, key=lambda r: order[r["status"]])[0]
    log = root / spec["log"] if spec.get("log") else None
    return collect_evolve(spec, root, [root / d for d in spec["dirs"]], log)


# ---------------------------------------------------------------- experiment and smokes

def variant_labels(hdir: Path) -> dict[str, str]:
    labels = dict(VARIANT_FALLBACK)
    for m in re.finditer(r'^\s*"(\w+)": dict\((.*?)\),', read_text(hdir / "headroom.py"), re.M):
        body = m.group(2)
        model = "Super 120B" if "SUPER" in body else "Nano 30B" if "NANO" in body else "?"
        parts = [model, "reasoning on" if "thinking=True" in body else "reasoning off"]
        tok = re.search(r"max_tokens=(\d+)", body)
        if tok:
            parts.append(f"max {int(tok.group(1)) // 1000}k tokens")
        if "examples=True" in body:
            parts.append("nearest examples")
        if "playbook=True" in body:
            parts.append("small-edit playbook")
        labels[m.group(1)] = ", ".join(parts)
    return labels


def collect_model_settings(hdir: Path, now: float) -> dict[str, Any]:
    out: dict[str, Any] = {"id": "model-settings", "name": "Model settings: Nano vs Super, reasoning on/off",
                           "verifier": "DoneTwin checker on 4 train halls (Layout role varies; Planner and Checker "
                                       "stay on Nano); score = DoneTwin score / 100",
                           "baseline_name": "greedy layout (no LLM)", "groups": [], "sources": []}
    files = sorted(hdir.glob("results_*.jsonl")) if hdir.is_dir() else []
    logs = sorted(hdir.glob("*.log")) if hdir.is_dir() else []
    rows = [r for p in files for r in read_jsonl(p) if "variant" in r and "score" in r]
    touched = [p.stat().st_mtime for p in files + logs]
    if not rows and not logs:
        out.update(status="not started", status_detail="no results_*.jsonl yet", baseline_score=None, cost_usd=0.0)
        return out
    fresh = bool(touched) and now - max(touched) < FRESH_S
    labels = variant_labels(hdir)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["variant"], r.get("tag", "")), []).append(r)
    for (v, tag), rs in sorted(groups.items()):
        atts = [a for r in rs for a in r.get("attempts", [])]
        out["groups"].append({
            "variant": v, "label": labels.get(v, v), "tag": tag, "halls": len(rs),
            "mean_score": round(mean(r["score"] for r in rs), 4),
            "greedy": round(mean(r["greedy"] for r in rs) / 100, 4),
            "mean_gain_pts": round(mean(r.get("gain", 0.0) for r in rs), 3),
            "improved": sum(r.get("gain", 0) > 0 for r in rs),
            "feasible_attempts": sum(bool(a.get("feasible")) for a in atts), "attempts": len(atts),
            "calls_per_hall": round(mean(r.get("calls", 0) for r in rs), 2),
            "tokens_per_hall": round(mean(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in rs)),
            "cost_usd": round(sum(r.get("cost_usd", 0.0) for r in rs), 6),
            "wall_s_per_hall": round(mean(r.get("wall_s", 0.0) for r in rs), 1),
            "errors": sum(len(r.get("errors") or []) for r in rs)})
    halls = {r["hall"]: r["greedy"] for r in rows if "hall" in r and "greedy" in r}
    out.update(status="running" if fresh else "done",
               status_detail=f"{len(rows)} hall runs in {len(out['groups'])} configs; last write "
                             f"{stamp(max(touched)) if touched else 'n/a'}",
               baseline_score=round(mean(halls.values()) / 100, 4) if halls else None,
               cost_usd=round(sum(r.get("cost_usd", 0.0) for r in rows), 6),
               calls=sum(r.get("calls", 0) for r in rows),
               sources=[str(p) for p in files])
    if out["groups"]:
        best = max(out["groups"], key=lambda g: (g["mean_score"], -g["cost_usd"]))
        out["best"] = {"config": f"{best['variant']} ({best['label']})", "score": best["mean_score"],
                       "gain": round(best["mean_score"] - best["greedy"], 6)}
    return out


def collect_smokes(root: Path) -> list[dict[str, Any]]:
    smokes: list[dict[str, Any]] = [dict(CONNECTION_PROBE, kind="probe")]
    for rel_dir in SMOKES:
        d = root / rel_dir
        if not d.is_dir():
            continue
        usage = read_json(d / "usage.json") or read_json(d / "reports" / "usage.json")
        total = (usage or {}).get("total") or {}
        tasks = []
        for p in sorted((d / "logs").glob("*.json")) if (d / "logs").is_dir() else []:
            data = read_json(p)
            if not isinstance(data, dict):
                continue
            atts = [a for a in data.get("attempts", []) if isinstance(a, dict)]
            ok = [a for a in atts if a.get("status") == "ok"]
            scores = [a.get("score", a.get("mean_ratio")) for a in ok]
            scores = [s for s in scores if isinstance(s, (int, float))]
            best = data.get("score") if isinstance(data.get("score"), (int, float)) else max(scores, default=None)
            gaps = [a["gap_pct"] for a in ok if isinstance(a.get("gap_pct"), (int, float))]
            tasks.append({"task": data.get("task", p.stem), "best_score": best,
                          "best_gap_pct": round(min(gaps), 3) if gaps else None,
                          "attempts_ok": len(ok), "attempts": len(atts),
                          "statuses": [a.get("status") for a in atts],
                          "techniques": [short(a.get("technique", ""), 60) for a in atts]})
        smokes.append({"name": rel_dir.split("/")[-1], "kind": "run_one", "tasks": tasks,
                       "calls": total.get("calls"), "prompt_tokens": total.get("prompt_tokens"),
                       "completion_tokens": total.get("completion_tokens"),
                       "cached_tokens": total.get("cached_tokens"), "cost_usd": total.get("cost_usd"),
                       "latency_s": total.get("latency_s"), "models": sorted((usage or {}).get("by_model") or {}),
                       "sources": [rel(p, root) for p in sorted(d.rglob("*.json"))]})
    return smokes


def ignored_runs(root: Path, cases: list[dict]) -> list[dict[str, str]]:
    """Run directories this report does not count, by name only, so nothing is silently dropped."""
    used = set()
    for spec in CASE_STUDIES:
        used.update(Path(d).name for d in spec.get("dirs", []) + spec.get("aliases", []))
    used.update(Path(s).name for s in SMOKES)
    for c in cases:
        used.update(Path(s).parts[1] for s in c["sources"] if s.startswith("runs/") and len(Path(s).parts) > 2)
    runs = root / "runs"
    out = []
    for d in sorted(runs.iterdir()) if runs.is_dir() else []:
        if d.is_dir() and d.name not in used:
            live = (read_json(d / "reports" / "usage.json") or {}).get("total", {}).get("cost_usd")
            out.append({"run": f"runs/{d.name}",
                        "reason": "not a named live run" if live else "no paid usage.json: offline, scripted or a service log"})
    return out


def collect(root: Path = ROOT, headroom_dir: Path = HEADROOM_DIR, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    cases = [collect_case(spec, root) for spec in CASE_STUDIES]
    sa = sa_library_counts(root, [c["id"] for c in cases])
    if sa:
        for c in cases:
            c["sa_library"] = sa.get(c["id"])
    return {"schema_version": SCHEMA_VERSION,
            "generated_at": datetime.fromtimestamp(now).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
            "command": COMMAND, "case_studies": cases,
            "experiments": [collect_model_settings(headroom_dir, now)],
            "smoke_tests": collect_smokes(root), "not_counted": ignored_runs(root, cases)}


# ---------------------------------------------------------------- rendering

def cell(x: Any) -> str:
    return str(x).replace("|", "/").replace("\n", " ").strip()


def f_score(x: Any, d: int) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def f_delta(delta: Any, pct: Any, d: int) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.{d}f}" + ("" if pct is None else f" ({pct:+.1f}%)")


def f_money(x: Any) -> str:
    return "n/a" if x is None else f"${x:.4f}"


def f_int(x: Any) -> str:
    return "n/a" if x is None else f"{int(x):,}"


def summary_row(c: dict) -> list[str]:
    d = c["digits"]
    b = c["baseline"]
    base = f"{b['name']} {f_score(b['score'], d)}" if b["score"] is not None else f"{b['name']} (pending)"
    status = c["status"] if c["status"] in ("done", "not started") else f"{c['status']}: {c['status_detail']}"
    if c["best"]:
        best = f"{c['best']['config']} {f_score(c['best']['score'], d)}"
        team_rating = [r for r in c.get("rating") or [] if str(r.get("entry", "")).startswith("team")]
        if team_rating:
            top = max(team_rating, key=lambda r: r.get("total", 0))
            best += f"; rank {top.get('rank')}/{top.get('entrants')}, perf {top.get('performance')}"
        gain = f_delta(c["best"]["gain"], c["best"]["gain_pct"], d)
    else:
        best, gain = ("pending", "pending") if c["status"] in ("running", "not started") else ("n/a", "n/a")
    t = c["totals"]
    cost = (f"{f_money(t['cost_usd'])} ({f_int(t['calls'])} calls)" if t else
            "pending (usage.json written at run end)" if c["status"] == "running" else
            "n/a (crashed before usage.json)" if c["status_detail"].startswith("crashed") else "n/a")
    return [c["name"], base, best, gain, status, cost]


def render_case(c: dict) -> list[str]:
    d = c["digits"]
    b = c["baseline"]
    out = [f"## {c['name']}", "",
           f"- Verifier: {c['verifier']}.",
           f"- Baseline: {b['name']}, held-out mean {f_score(b['score'], d)} (source: {b['source']}).",
           f"- Status: {c['status']}; {c['status_detail']}."]
    if c.get("live_evidence"):
        out.append(f"- Live evidence: {c['live_evidence']}.")
    t = c["totals"]
    if t:
        wall = "n/a" if t["wall_s"] is None else f"{t['wall_s']:.1f} s"
        out.append(f"- Totals: {f_int(t['calls'])} calls, {f_int(t['tokens'])} tokens ({f_int(t['prompt_tokens'])} "
                   f"prompt + {f_int(t['completion_tokens'])} completion, {f_int(t['cached_tokens'])} cached), "
                   f"{f_money(t['cost_usd'])}, {f_int(t['errors'])} errors, wall {wall}.")
        if t.get("by_model"):
            out.append("- By model: " + "; ".join(f"{m.split('/')[-1]} {f_int(v['calls'])} calls {f_money(v['cost_usd'])}"
                                                  for m, v in t["by_model"].items()) + ".")
    elif c["status"] == "running":
        out.append("- Totals: n/a until the run ends (usage.json is written last).")
    elif c["status_detail"].startswith("crashed"):
        out.append("- Totals: n/a; the run crashed before writing usage.json, so its bill is not on disk.")
    out.append("")
    if c["heldout"]:
        gap = any(r.get("mean_gap_pct") is not None for r in c["heldout"])
        out += ["### Held-out", "",
                "| Config | n | Mean score | Pass rate | " + ("Gap to BKS (%) | " if gap else "")
                + "Tokens | Cost (USD) | vs baseline |",
                "|---|---|---|---|" + ("---|" if gap else "") + "---|---|---|"]
        for r in c["heldout"]:
            pr = "n/a" if r["pass_rate"] is None else f"{r['pass_rate'] * 100:.0f}%"
            g = ("n/a" if r.get("mean_gap_pct") is None else f"{r['mean_gap_pct']:.2f}") + " | " if gap else ""
            out.append(f"| {cell(r['config'])} | {r['n']} | {f_score(r['mean_score'], d)} | {pr} | {g}"
                       f"{f_int(r['tokens'])} | {f_money(r['cost_usd'])} | "
                       f"{'baseline' if r['is_baseline'] else f_delta(r['vs_baseline'], r['vs_baseline_pct'], d)} |")
        out.append("")
    for note in c["notes"]:
        out += [f"Note: {note}", ""]
    if c.get("rating"):
        out += ["### AtCoder rank on all 150 private seeds (583 contestants)", "",
                "| Entry | Total | Rank | Performance |", "|---|---|---|---|"]
        for r in c["rating"]:
            out.append(f"| {cell(r.get('entry'))} | {f_int(r.get('total'))} | {r.get('rank')}/{r.get('entrants')} | "
                       f"{r.get('performance')} |")
        out.append("")
    if c["history"]:
        out += ["Dev mean of the parent by generation: " + ", ".join(
            f"gen {h['gen']} {f_score(h['dev_mean'], d)}" for h in c["history"] if h.get("dev_mean") is not None)
            + ".", ""]
    if c["generations"]:
        p = c["papers"]
        verdicts = ", ".join(f"{v} {k}" for k, v in sorted(p["verdicts"].items(), key=lambda kv: -kv[1]))
        out += ["### Generations (each candidate is gated on the dev set)", "",
                f"{p['total']} candidates, {p['accepted']} accepted"
                + (f"; claim vs measured verdicts: {verdicts}" if verdicts else "") + ".", "",
                "| Gen | Mutation | Change | Claim | Measured dev gain | Decision | Why |",
                "|---|---|---|---|---|---|---|"]
        for g in c["generations"]:
            gain = "n/a" if g["measured_gain"] is None else f"{g['measured_gain']:+.4f}"
            dec = f"accepted ({g['accepted_by']})" if g["accepted"] else "rejected"
            out.append(f"| {g['gen']} | {cell(g['mutation_type'])} | {cell(g['change'])} | {cell(g['claim'])} | "
                       f"{gain} | {dec} | {cell(g['reason'])} |")
        out.append("")
    if c.get("sa_library"):
        s = c["sa_library"]
        out += [f"SA's Library: {s['papers']} papers; verdicts " + ", ".join(f"{k} {v}" for k, v in s["verdicts"].items())
                + ".", ""]
    if c["sources"]:
        out += ["Sources: " + ", ".join(f"`{s}`" for s in c["sources"]) + ".", ""]
    return out


def render_experiment(e: dict) -> list[str]:
    out = [f"## {e['name']}", "", f"- Verifier: {e['verifier']}.",
           f"- Baseline: {e['baseline_name']}, mean over the halls {f_score(e.get('baseline_score'), 4)}.",
           f"- Status: {e['status']}; {e['status_detail']}.",
           f"- Cost so far: {f_money(e.get('cost_usd'))} over {f_int(e.get('calls', 0))} calls.", ""]
    if e["groups"]:
        out += ["| Config | Layout model and settings | Run tag | Halls | Mean score | Greedy | Mean gain (pts of 100) | "
                "Improved | Feasible attempts | Calls/hall | Tokens/hall | Cost (USD) | Wall s/hall | Errors |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in e["groups"]:
            out.append(f"| {g['variant']} | {cell(g['label'])} | {cell(g['tag'])} | {g['halls']} | "
                       f"{g['mean_score']:.4f} | {g['greedy']:.4f} | {g['mean_gain_pts']:+.2f} | "
                       f"{g['improved']}/{g['halls']} | {g['feasible_attempts']}/{g['attempts']} | "
                       f"{g['calls_per_hall']:.1f} | {g['tokens_per_hall']:,} | {f_money(g['cost_usd'])} | "
                       f"{g['wall_s_per_hall']:.0f} | {g['errors']} |")
        out.append("")
    if e["sources"]:
        out += ["Sources: " + ", ".join(f"`{s}`" for s in e["sources"]) + ".", ""]
    return out


def render_smokes(smokes: list[dict]) -> list[str]:
    out = ["## Live smoke tests", ""]
    for s in smokes:
        if s["kind"] == "probe":
            out.append(f"**{s['name']}**, {s['source']}:")
            out.append("")
            out += ["| Model | Latency per call (s) | Cost per call (USD) | Cached tokens |", "|---|---|---|---|"]
            for r in s["rows"]:
                out.append(f"| {r['model']} | {r['latency_s']:.2f} | {r['cost_usd_per_call']:.6f} | {r['cached_tokens']} |")
            ro = s["reasoning_off"]
            out += ["", f"Reasoning off cut one call from {ro['tokens_before']} tokens and {ro['latency_before_s']:.1f} s "
                        f"to {ro['tokens_after']} tokens and {ro['latency_after_s']:.1f} s.", ""]
            continue
        bill = (f"{f_int(s['calls'])} calls, {f_int(s['prompt_tokens'])} prompt + {f_int(s['completion_tokens'])} "
                f"completion tokens ({f_int(s['cached_tokens'])} cached), {f_money(s['cost_usd'])}, "
                f"{s['latency_s']:.1f} s of call latency" if s["calls"] else "bill printed to the console only, not saved")
        out.append(f"**{s['name']}**: one train task with v0; {bill}"
                   + (f"; model {', '.join(m.split('/')[-1] for m in s['models'])}" if s["models"] else "") + ".")
        out.append("")
        for t in s["tasks"]:
            gap = "" if t["best_gap_pct"] is None else f", best gap to BKS {t['best_gap_pct']:.2f}%"
            out.append(f"- {t['task']}: best score {f_score(t['best_score'], 4)}{gap}; {t['attempts_ok']}/{t['attempts']} "
                       f"programs ran ({', '.join(cell(x) for x in t['statuses'])}).")
        out += ["", "Sources: " + ", ".join(f"`{x}`" for x in s["sources"]) + ".", ""]
    return out


def render_markdown(data: dict) -> str:
    cases, exps = data["case_studies"], data["experiments"]
    lines = ["# Benchmark results (live Crusoe runs only)", "",
             "Every number here comes from a live run on Crusoe Managed Inference (Nemotron-3 Nano 30B and Super 120B). "
             "Offline and scripted runs are tests and are excluded. Scores: higher is better; gain = best team "
             "version minus baseline on the held-out set.", "",
             "## Summary", "", "| Case study | Baseline | Best team version | Gain | Status | Cost |",
             "|---|---|---|---|---|---|"]
    for c in cases:
        lines.append("| " + " | ".join(cell(x) for x in summary_row(c)) + " |")
    for e in exps:
        best = e.get("best")
        status = e["status"] if e["status"] in ("done", "not started") else f"{e['status']}: {e['status_detail']}"
        lines.append("| " + " | ".join(cell(x) for x in [
            e["name"] + " (4 train halls)",
            f"{e['baseline_name']} {f_score(e.get('baseline_score'), 4)}",
            f"{best['config']} {best['score']:.4f}" if best else "pending",
            f_delta(best["gain"], None, 4) if best else "pending", status,
            f"{f_money(e.get('cost_usd'))} ({f_int(e.get('calls', 0))} calls)"]) + " |")
    lines.append("")
    for c in cases:
        lines += render_case(c)
    for e in exps:
        lines += render_experiment(e)
    lines += render_smokes(data["smoke_tests"])
    if data["not_counted"]:
        lines += ["## Not counted", "",
                  "These run directories are not results: " + "; ".join(
                      f"`{r['run']}` ({r['reason']})" for r in data["not_counted"]) + ".", ""]
    lines.append(f"Generated {data['generated_at']}. Regenerate with `{data['command']}` "
                 "(rewrites reports/RESULTS.md and reports/results.json).")
    text = "\n".join(lines) + "\n"
    return text.replace("—", "-").replace("–", "-")


def write_outputs(data: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / "results.json", out_dir / "RESULTS.md"
    jp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    mp.write_text(render_markdown(data))
    return jp, mp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT, help="repo root to read runs/ and reports/ from")
    ap.add_argument("--headroom", type=Path, default=HEADROOM_DIR, help="model-settings experiment directory")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <root>/reports")
    args = ap.parse_args(argv)
    data = collect(args.root, args.headroom)
    jp, mp = write_outputs(data, args.out_dir or args.root / "reports")
    print(f"wrote {jp} and {mp}")
    for c in data["case_studies"]:
        print(f"  {c['id']}: {c['status']} ({c['status_detail']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
