"""Build dashboard/public/datacenter.json: the four AI data center layers, from live run artifacts only.

Usage (repo venv):
  .venv/bin/python scripts/build_datacenter_view.py [--root DIR] [--out FILE]

Rerun it after any run finishes; it rewrites the file from what is on disk and is safe to run while runs are in
progress (a run without its final report counts as running while its process is alive, else as stopped).
Runs whose usage.json shows $0 or whose log or command line says --offline are tests and are skipped.
Parsing is shared with scripts/collect_results.py (imported, not copied).
It also writes runs_index (every live run by layer, with its hill-climb series) and reports/RUNS.md.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_results as cr  # noqa: E402  (shared tolerant readers and run parsing)

ROOT = cr.ROOT
COMMAND = ".venv/bin/python scripts/build_datacenter_view.py"
PITCH = "Every layer of an AI data center as a verifiable optimization problem."


def live_commands() -> str:
    try:
        out = subprocess.run(["ps", "-ax", "-o", "command"], capture_output=True, text=True, timeout=5).stdout
        # Only the run scripts count: a `tail -F runs/x.log` watcher does not make run x alive.
        return "\n".join(line for line in out.splitlines() if "python" in line.lower() and "scripts/run_" in line)
    except (OSError, subprocess.SubprocessError):
        return ""


def is_offline(run_dir: Path, log: Path | None) -> bool:
    text = cr.read_text(log)[:4000] if log else ""
    return "--offline" in text or "offline" in run_dir.name or "scripted LLM" in text


def run_kind(config: str | None) -> str:
    c = (config or "").lower()
    if "no llm" in c or "baseline" in c:
        return "baseline"
    if "cold" in c or "empty brain" in c:
        return "cold"
    if "final" in c and c.startswith("v0 "):
        return "brain"
    return "final"


def vs_v0(bars: list[dict]) -> dict[str, Any]:
    """The headline pair: v0 cold vs final on the same held-out tasks. v0 cold is the only baseline shown."""
    k = {b["kind"]: b for b in bars}
    v0, fin = k.get("cold"), k.get("final")
    return {"v0": v0 and v0["score"], "final": fin and fin["score"], "final_label": fin and fin["label"],
            "delta": round(fin["score"] - v0["score"], 4) if v0 and fin else None}


def collect_run(spec: dict, run_dir: Path, log: Path | None, procs: str) -> dict[str, Any] | None:
    if not run_dir.exists() or is_offline(run_dir, log):
        return None
    r = cr.collect_evolve(spec, ROOT, [run_dir], log)
    if "not a live result" in r["status_detail"]:
        return None
    alive = run_dir.name in procs or (log is not None and str(log) in procs)
    if r["status"] != "done" and not (r["status"] == "running" and alive):
        return None  # crashed, stopped or blocked: not a result, and not in progress
    bars, seen = [], set()
    for row in r["heldout"]:
        kind = run_kind(row["config"]) if not row["is_baseline"] else (
            "ablation" if "polish" in (row["config"] or "").lower() else "baseline")
        if kind in seen or row["mean_score"] is None:
            continue
        seen.add(kind)
        bars.append({"label": row["config"], "kind": kind, "score": row["mean_score"],
                     "pass_rate": row.get("pass_rate"), "n": row.get("n"), "cost_usd": row.get("cost_usd")})
    # No candidate passed the gate when the final version is v0: the "final" bar is the v0 + brain arm.
    if bars and not any(b["kind"] == "final" for b in bars):
        brain = next((b for b in bars if b["kind"] == "brain"), None)
        if brain:
            bars.append(dict(brain, kind="final", label="final (v0: no change passed the gate)"))
    order = {"baseline": 0, "ablation": 1, "cold": 2, "brain": 3, "final": 4}
    bars.sort(key=lambda b: order[b["kind"]])
    ledger_path = cr.find_file([run_dir], ("version_ledger.jsonl", "ledger.jsonl"))
    if ledger_path and ledger_path.parent.name == "brain":
        ledger_path = None
    candidates = [{"gen": e.get("generation"), "score": e.get("cand_mean"), "parent_score": e.get("parent_mean"),
                   "accepted": bool(e.get("accepted")), "reason": cr.short(e.get("reason") or "", 80),
                   "change": cr.describe_mutation(e.get("mutation") if isinstance(e.get("mutation"), dict) else {})}
                  for e in cr.read_jsonl(ledger_path) if e.get("cand_mean") is not None]
    dev = [{"gen": h["gen"], "score": h["dev_mean"]} for h in r["history"] if h.get("dev_mean") is not None]
    if (bars and bars[0]["kind"] == "baseline" and r["baseline"]["score"] is not None
            and r["baseline"].get("name") == bars[0]["label"]):
        bars[0]["score"] = r["baseline"]["score"]  # unrounded; the markdown table rounds to 3 digits
    total = (r.get("totals") or {})
    # No-LLM heuristics (greedy, 2-swap, greedy + polish) are left out: v0 cold is the only baseline on this tab.
    bars = [b for b in bars if b["kind"] in ("cold", "brain", "final")]
    head = vs_v0(bars)
    return {"id": run_dir.name, "status": r["status"], "status_detail": r["status_detail"],
            "baseline": {"name": "v0 cold", "score": head["v0"]}, "vs_v0": head, "heldout": bars, "dev": dev,
            "candidates": candidates,
            "cost_usd": total.get("cost_usd"), "calls": total.get("calls"), "last_update": r["last_update"],
            "n_heldout": bars[0]["n"] if bars else None}


def layer_status(runs: list[dict]) -> str:
    if any(r["status"] == "done" for r in runs):
        return "done"
    if any(r["status"] == "running" for r in runs):
        return "running"
    return "not run"


def best_of(runs: list[dict]) -> dict | None:
    """The finished run with the largest final - v0 cold on its held-out tasks, preferring runs whose v0 cold had no
    invalid result (a 0 there inflates the delta); ties go to the latest."""
    out, out_key = None, None
    for r in sorted(runs, key=lambda r: str(r.get("last_update") or "")):
        g = r.get("vs_v0") or {}
        if r["status"] != "done" or g.get("delta") is None:
            continue
        invalid = (qaplib_rows(r["id"]).get("cold") or {}).get("invalid") or 0
        cand = {"run": r["id"], "label": g["final_label"], "score": g["final"], "baseline": g["v0"],
                "delta": g["delta"], "n": r["n_heldout"], "family": r.get("family"), "v0_invalid": invalid}
        key = (invalid == 0, cand["delta"])
        if out_key is None or key >= out_key:
            out, out_key = cand, key
    return out


def lead(best: dict, noun: str, d: int = 3) -> str:
    return (f"From v0 {fmt(best['baseline'], d)} to {fmt(best['score'], d)} ({signed(best['delta'], d)}) "
            f"on {best['n']} unseen {noun}.")


def fmt(x: float | None, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def signed(x: float, d: int = 3) -> str:
    return f"{'+' if x >= 0 else '-'}{abs(x):.{d}f}"


def progress_note(runs: list[dict]) -> str:
    live = [r for r in runs if r["status"] == "running"]
    if not live:
        return ""
    r = live[0]
    acc = sum(c["accepted"] for c in r["candidates"])
    return f" {r['id']} is running: {cr.plural(len(r['candidates']), 'candidate change')} scored on dev so far, {acc} accepted."


def hall_layer(procs: str) -> dict[str, Any]:
    spec = dict(cr.CASE_STUDIES[0])
    runs = []
    for d, log in [(ROOT / "reports" / "datahall_live_1", ROOT / "reports" / "datahall_live_1" / "run.log")] + [
            (p, ROOT / "runs" / f"{p.name}.log") for p in sorted((ROOT / "runs").glob("datahall-live-*"))
            if p.is_dir() and p.name != "datahall-live-1"]:
        r = collect_run(dict(spec, id=d.name), d, log, procs)
        if r:
            r["id"] = "datahall-live-1" if d.name == "datahall_live_1" else r["id"]
            runs.append(r)
    ms = cr.collect_model_settings(cr.HEADROOM_DIR, time.time())
    settings = [{"label": g["label"], "tag": g["tag"],
                 "variant": g["variant"], "gain_pts": g["mean_gain_pts"], "score": g["mean_score"],
                 "greedy": g["greedy"], "feasible": g["feasible_attempts"], "attempts": g["attempts"],
                 "halls": g["halls"], "cost_usd": g["cost_usd"]}
                for g in ms.get("groups", []) if g["cost_usd"] > 0 and g["attempts"] > 0]
    best = best_of(runs)
    summary = "No finished live run yet."
    if best:
        summary = lead(best, "halls")
    settings = []  # the model-settings sweep is scored against greedy, a no-LLM reference: not shown on this tab
    return {"id": "hall", "name": "Hall layout", "problem": "Place racks, CDUs and power in a data hall (DoneTwin).",
            "verified": spec["verifier"], "score_label": "DoneTwin score / 100", "digits": 3,
            "baseline": {"name": "v0 cold", "score": best["baseline"] if best else None},
            "note": "v0 starts from the greedy layout, so v0 cold equals greedy here.",
            "best": best, "status": layer_status(runs), "summary": summary + progress_note(runs),
            "runs": runs, "settings": settings}


def qaplib_rows(run_id: str) -> dict[str, dict]:
    rows = cr.read_json(ROOT / "runs" / run_id / "reports" / "qaplib_rows.json") or []
    out: dict[str, dict] = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("version") is not None:
            out.setdefault(run_kind(row.get("config")), row)
    return out


def qaplib_sota(runs: list[dict]) -> dict | None:
    """QAPLIB best-known solutions are the published SOTA: report the finished run whose final is closest to them."""
    best = None
    for r in runs:
        rows = qaplib_rows(r["id"])
        fin, cold = rows.get("final") or rows.get("brain"), rows.get("cold")
        if not fin or not cold or fin.get("mean_gap_pct") is None or cold.get("mean_gap_pct") is None:
            continue
        if best is None or fin["mean_gap_pct"] <= best["final_gap"]:
            best = {"run": r["id"], "n": fin.get("n"), "final_gap": fin["mean_gap_pct"], "v0_gap": cold["mean_gap_pct"]}
    if best:
        best["headline"] = (f"{best['final_gap']:.1f}% from best-known solutions (SOTA) on {best['n']} unseen instances "
                            f"({best['run']} final).")
        best["secondary"] = f"v0 cold: {best['v0_gap']:.1f}% from best-known."
    return best


def invalid_note(best: dict) -> str | None:
    """A v0 cold row with invalid placements (scored 0) drags its mean down; say so next to the delta."""
    cold = qaplib_rows(best["run"]).get("cold") or {}
    k, n = cold.get("invalid") or 0, cold.get("n") or 0
    if not k or n <= k or best.get("baseline") is None:
        return None
    rest = best["baseline"] * n / (n - k)
    return (f"v0 cold produced an invalid placement on {k} of {n} instances (scored 0). On the other {n - k}, v0 averages "
            f"{rest:.3f}; final is {best['score']:.3f} over all {n} ({signed(best['score'] - rest)}).")


def placement_layer(procs: str) -> dict[str, Any]:
    spec = dict(cr.CASE_STUDIES[2])
    dirs = [p for p in sorted((ROOT / "runs").glob("qaplib-evolve-*")) if p.is_dir() and p.name != "qaplib-evolve-1"]
    dirs += [p for p in sorted((ROOT / "runs").glob("qaplib-research-*")) if p.is_dir()]
    dirs += [p for p in sorted((ROOT / "runs").glob("qaplib-dc-*")) if p.is_dir()]
    runs = [r for p in dirs if (r := collect_run(dict(spec, id=p.name), p, ROOT / "runs" / f"{p.name}.log", procs))]
    for r in runs:
        r["family"] = "GPU rack placement" if r["id"].startswith("qaplib-dc") else "QAPLIB"
    best = best_of(runs)
    summary = "No finished live run yet."
    if best:
        summary = lead(best, "rack placements" if best["family"] == "GPU rack placement" else "QAPLIB instances")
    sota = qaplib_sota([r for r in runs if r["family"] == "QAPLIB" and r["status"] == "done"])
    if sota:
        summary += f" QAPLIB: {sota['headline'][:-1]}; {sota['secondary']}"
    notes = [n for n in [best and invalid_note(best)] if n]
    return {"id": "placement", "name": "Equipment placement",
            "problem": "Quadratic assignment: put equipment in slots to cut flow x distance.",
            "verified": spec["verifier"], "score_label": "best-known / cost", "digits": 3,
            "baseline": {"name": "v0 cold", "score": best["baseline"] if best else None},
            "sota": sota, "caveats": notes,
            "best": best, "status": layer_status(runs), "summary": summary + progress_note(runs), "runs": runs}


def kernel_rows(text: str) -> dict[str, dict]:
    """The final held-out table in a KernelBench run log: fast_1, correct %, mean speedup of correct kernels."""
    out: dict[str, dict] = {}
    for m in re.finditer(r"^\| (v\d+ \+ [^|]+?) \| (\d+%) \| (\d+%) \| ([\d.]+x|n/a) \| ([\d.]+) \|", text, re.M):
        out.setdefault(run_kind(m.group(1)), {"fast_1": m.group(2), "correct": m.group(3), "speedup": m.group(4)})
    if "final" not in out and "brain" in out:
        out["final"] = out["brain"]  # no version change passed the gate: final is v0 + final brain
    return out


def kernel_layer(procs: str) -> dict[str, Any]:
    spec = dict(cr.CASE_STUDIES[4])
    runs, device = [], None
    for p in sorted((ROOT / "runs").glob("kernelbench-mps-*")):
        log = ROOT / "runs" / f"{p.name}.log"
        r = collect_run(dict(spec, id=p.name), p, log, procs) if p.is_dir() else None
        if not r:
            continue
        text = cr.read_text(log)
        m = re.search(r"GPU evaluator: \w+ \(([^,]+),", text)
        r["device"] = m.group(1) if m else "Apple GPU (MPS)"
        r["family"] = f"timed on {r['device']}"
        r["kernel"] = kernel_rows(text)
        device = device or r["device"]
        runs.append(r)
    best = best_of(runs)
    summary, caveats = "Not run: our Crusoe project has zero GPU quota, so there is no live result to show.", []
    if best:
        k = next((r["kernel"] for r in runs if r["id"] == best["run"]), {})
        cold, fin = k.get("cold") or {}, k.get("final") or {}
        summary = lead(best, "problems")
        if cold.get("correct") and fin.get("correct"):
            caveats.append(f"correct kernels {cold['correct']} → {fin['correct']}")
        fast = fin.get("fast_1")
        caveats.append((f"no kernel beat PyTorch eager yet (fast_1 {fast})" if fast == "0%" else
                        f"fast_1 {fast or 'n/a'} (correct and faster than PyTorch eager)")
                       + "; MPS timings are not comparable to CUDA")
    elif runs:
        summary = "Running: no held-out result yet." + progress_note(runs)
    return {"id": "kernels", "name": "GPU kernels", "problem": "KernelBench mini: write faster GPU kernels.",
            "verified": spec["verifier"], "score_label": "KernelBench score (mean)", "digits": 3,
            "device": device, "baseline": {"name": "v0 cold", "score": best["baseline"] if best else None},
            "best": best, "status": layer_status(runs) if runs else "not run", "caveats": caveats,
            "summary": summary + (progress_note(runs) if best else ""), "runs": runs}


def inference_caveats(points: list[dict], base: dict, top: dict) -> list[str]:
    """Small print for the inference card: a one-question accuracy change is noise, and the latency trade."""
    out = []
    q = base.get("questions")
    if base.get("correct") is not None and top.get("correct") is not None and q:
        gap = top["correct"] - base["correct"]
        if abs(gap) <= 1:
            out.append(f"{base['correct']}/{q} to {top['correct']}/{q} is {'one question' if gap else 'no change'}, "
                       "so no accuracy gain is claimed.")
    data = cr.read_json(ROOT / "runs" / top["run"] / "reports" / "heldout_cost_latency.json") or {}
    rows = {r.get("config"): r for r in data.get("rows", [])}
    a, b = rows.get(base["label"]) or {}, rows.get(top["label"]) or {}

    def moved(x: float | None, y: float | None, unit: str, d: int) -> str | None:
        if not x or y is None:
            return None
        word = "doubled" if y / x >= 1.8 else "halved" if y / x <= 0.55 else "went"
        return f"{word} ({x:.{d}f} {unit} to {y:.{d}f} {unit})"
    lat = moved(a.get("p50_call_latency_s"), b.get("p50_call_latency_s"), "s", 1)
    wall = moved(a.get("p50_task_wall_s"), b.get("p50_task_wall_s"), "s", 0)
    if lat and wall:
        out.append(f"Median call latency {lat}; time per hall {wall}.")
    return out


def inference_layer(procs: str) -> dict[str, Any]:
    spec = dict(cr.CASE_STUDIES[5])
    d = ROOT / "runs" / "inference-opt-1"
    runs, points = [], []
    for p in sorted((ROOT / "runs").glob("inference-opt-*")):
        if not p.is_dir() or p.name.endswith("verifier"):
            continue
        r = collect_run(dict(spec, id=p.name), p, ROOT / "runs" / f"{p.name}.log", procs)
        if not r:
            continue
        runs.append(r)
        data = cr.read_json(p / "reports" / "heldout_cost_latency.json") if r["status"] == "done" else None
        for row in (data or {}).get("rows", []):
            if row.get("cost_per_correct_usd") is None:
                continue
            points.append({"run": p.name, "label": row["config"], "accuracy": row["accuracy"],
                           "cost_per_correct_usd": row["cost_per_correct_usd"], "correct": row.get("correct"),
                           "questions": row.get("questions"), "baseline": "cold" in row["config"]})
    base = next((pt for pt in points if pt["baseline"]), None)
    best, caveats = None, []
    summary = ("Running: no held-out result yet." if runs else "No live run yet.") + progress_note(runs)
    if base:
        final = [pt for pt in points if not pt["baseline"] and pt["accuracy"] >= base["accuracy"] - 0.02]
        top = min(final, key=lambda pt: pt["cost_per_correct_usd"], default=None)
        if top:
            ratio = top["cost_per_correct_usd"] / base["cost_per_correct_usd"]
            best = {"run": top["run"], "label": top["label"], "score": top["cost_per_correct_usd"],
                    "baseline": base["cost_per_correct_usd"], "delta": round(ratio - 1, 4),
                    "accuracy": top["accuracy"], "baseline_accuracy": base["accuracy"]}
            summary = (f"From v0 ${base['cost_per_correct_usd']:.6f} to ${top['cost_per_correct_usd']:.6f} per correct "
                       f"answer, {1 - ratio:.0%} less, on {base.get('questions') or '?'} unseen questions. "
                       "v0 is the expensive starting setup.")
            caveats = inference_caveats(points, base, top)
    return {"id": "inference", "name": "Inference cost",
            "problem": "Answer exact data-center ops questions on Crusoe for the least $ per correct answer.",
            "verified": spec["verifier"], "score_label": "$ per correct answer (lower is better)", "digits": 6,
            "baseline": {"name": "v0 cold: Super, reasoning on, 4000 tokens, 1 question per call",
                         "score": base["cost_per_correct_usd"] if base else None},
            "caveats": caveats,
            "best": best, "status": layer_status(runs), "summary": summary, "runs": runs, "scatter": points}


# ---------------------------------------------------------------- runs index and hill-climb series

# Run-name prefix -> (layer id, benchmark, spec index in cr.CASE_STUDIES). Longest prefix wins.
PREFIXES = [("datahall-live-", "hall", "DoneTwin data hall", 0),
            ("qaplib-dc-research-", "placement", "GPU rack placement", 2),
            ("qaplib-dc-", "placement", "GPU rack placement", 2),
            ("qaplib-research-", "placement", "QAPLIB", 2),
            ("qaplib-evolve-", "placement", "QAPLIB", 2),
            ("kernelbench-live-", "kernels", "KernelBench", 4),
            ("kernelbench-mps-", "kernels", "KernelBench (Apple GPU, MPS)", 4),
            ("inference-opt-", "inference", "Crusoe ops Q&A", 5)]
LAYER_NAMES = {"hall": "Hall layout", "placement": "Equipment placement", "kernels": "GPU kernels",
               "inference": "Inference cost"}
RUNNING_WINDOW_S = 180


def short_model(name: str | None) -> str | None:
    if not name:
        return None
    for key in ("Nano", "Super", "Lightning", "Ultra"):
        if key.lower() in name.lower():
            return key
    return re.sub(r"^(nvidia/)?(NVIDIA-)?", "", name.strip())


def log_models(text: str) -> tuple[str | None, str | None]:
    m = re.search(r"(?:solver|worker) calls \(([^)]+)\).*?reviewer calls \(([^)]+)\)", text)
    if m:
        return short_model(m.group(1)), short_model(m.group(2))
    solver = re.search(r"^v0: (\w+) for every type", text, re.M) or re.search(r"--solver-model[= ](\S+)", text)
    reviewer = (re.search(r"reviewer (NVIDIA-[\w.-]+|nvidia/[\w.-]+)", text)
                or re.search(r"--reviewer-model[= ](\S+)", text))
    return (short_model(solver.group(1)) if solver else None, short_model(reviewer.group(1)) if reviewer else None)


def dev_baseline(layer: str, benchmark: str, text: str) -> tuple[float | None, str | None]:
    if layer == "hall":
        m = re.search(r"greedy baseline:.*?dev ([\d.]+)", text)
        return (float(m.group(1)) / 100, "greedy layout, dev") if m else (None, None)
    if benchmark == "QAPLIB":
        names = re.findall(r"(\w+)\(n=\d+\)", (re.search(r"^dev: (.*)$", text, re.M) or [None, ""])[1] or "")
        inst = (cr.read_json(ROOT / "reports" / "qaplib_baseline.json") or {}).get("instances") or {}
        scores = [inst[n]["two_swap"]["score"] for n in names if n in inst and "two_swap" in inst[n]]
        if scores and len(scores) == len(names):
            return round(sum(scores) / len(scores), 4), "2-swap, dev"
    return None, None


def started_at(log: Path, run_dir: Path) -> float | None:
    v0 = run_dir / "versions" / "v0.json"
    stats = [getattr(log.stat(), "st_birthtime", None) if log.exists() else None,
             v0.stat().st_mtime if v0.exists() else None]
    stats = [s for s in stats if s]
    return min(stats) if stats else None


def iso(ts: float | None) -> str | None:
    return None if ts is None else time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def climb_series(ledger: list[dict]) -> dict[str, Any]:
    research = {e.get("generation"): e for e in ledger if "research_query" in e and "cand_mean" not in e}
    cands, gens = [], []
    gens_start = None
    incumbent, cur_gen, per_gen = None, None, 0
    for e in ledger:
        if e.get("cand_mean") is None:
            continue
        gen = e.get("generation")
        if gen != cur_gen:
            if cur_gen is not None:
                gens.append({"gen": cur_gen, "start": gens_start, "end": incumbent})
            cur_gen, per_gen = gen, 0
            incumbent = e.get("parent_mean") if e.get("parent_mean") is not None else incumbent
            gens_start = incumbent
        m = e.get("mutation") if isinstance(e.get("mutation"), dict) else {}
        paper = e.get("paper") if isinstance(e.get("paper"), dict) else {}
        claim = paper.get("claim") or m.get("claim") or {}
        claim = claim if isinstance(claim, dict) else {}
        target = m.get("key") or m.get("role") or m.get("tool")
        res = research.get(gen) or (e if e.get("sources") else {})
        accepted = bool(e.get("accepted"))
        if accepted and (incumbent is None or e["cand_mean"] > incumbent):
            incumbent = e["cand_mean"]
        cands.append({
            "i": len(cands), "gen": gen, "id": f"g{gen}-c{per_gen}", "parent": e.get("parent"),
            "type": m.get("type"), "target": target, "change": cr.describe_mutation(m),
            "rationale": cr.short(m.get("rationale") or claim.get("why") or "", 120),
            "claim_gain": claim.get("dev_gain") if isinstance(claim.get("dev_gain"), (int, float)) else None,
            "claim_cost_pct": claim.get("cost_change_pct") if isinstance(claim.get("cost_change_pct"), (int, float)) else None,
            "score": e.get("cand_mean"), "parent_score": e.get("parent_mean"), "gain": e.get("gain"),
            "cost": e.get("cand_cost"), "parent_cost": e.get("parent_cost"), "cost_pct": e.get("cost_change_pct"),
            "n": e.get("n"), "accepted": accepted, "accepted_by": e.get("accepted_by"),
            "reason": cr.short(e.get("reason") or "", 120), "verdict": paper.get("verdict"),
            "research_query": cr.short(res.get("research_query") or "", 160) or None,
            "sources": [s for s in (res.get("sources") or []) if isinstance(s, str) and s.startswith("http")][:6],
            "incumbent": incumbent})
        per_gen += 1
    if cur_gen is not None:
        gens.append({"gen": cur_gen, "start": gens_start, "end": incumbent})
    return {"candidates": cands, "generations": gens}


def run_status_index(text: str, usage_total: dict, alive: bool, log: Path) -> str:
    has_usage = bool(re.search(r"^usage: \{", text, re.M)) or (usage_total.get("cost_usd") or 0) > 0
    if has_usage:
        return "done"
    if "Traceback (most recent call last)" in text:
        return "crashed"
    if alive or (log.exists() and time.time() - log.stat().st_mtime < RUNNING_WINDOW_S):
        return "running"
    return "stopped"


def heldout_summary(layer: str, run: dict | None, run_dir: Path) -> dict | None:
    if not run or run["status"] != "done":
        return None
    if layer == "inference":
        rows = (cr.read_json(run_dir / "reports" / "heldout_cost_latency.json") or {}).get("rows", [])
        base = next((r for r in rows if "cold" in r.get("config", "") and r.get("cost_per_correct_usd")), None)
        rest = [r for r in rows if r is not base and r.get("cost_per_correct_usd")]
        top = min(rest, key=lambda r: r["cost_per_correct_usd"], default=None)
        if base and top:
            return {"metric": "$ per correct", "final": top["cost_per_correct_usd"], "v0": base["cost_per_correct_usd"],
                    "delta": round(top["cost_per_correct_usd"] - base["cost_per_correct_usd"], 6),
                    "final_label": top["config"], "lower_is_better": True}
        return None
    g = run.get("vs_v0") or {}
    if g.get("delta") is None:
        return None
    return {"metric": "held-out mean", "final": g["final"], "v0": g["v0"], "delta": g["delta"],
            "final_label": g["final_label"], "lower_is_better": False,
            "v0_invalid": (qaplib_rows(run["id"]).get("cold") or {}).get("invalid") or 0}


def index_runs(procs: str) -> list[dict[str, Any]]:
    out = []
    for d in sorted((ROOT / "runs").iterdir()):
        if not d.is_dir():
            continue
        match = next((p for p in PREFIXES if d.name.startswith(p[0])), None)
        log = ROOT / "runs" / f"{d.name}.log"
        if not match or is_offline(d, log):
            continue
        _, layer, benchmark, spec_i = match
        text = cr.read_text(log)
        usage = cr.read_json(cr.find_file([d], ("usage.json",))) or {}
        total = usage.get("total") or {}
        if total and (total.get("cost_usd") or 0) <= 0:
            continue  # scripted: usage.json shows $0
        alive = d.name in procs
        status = run_status_index(text, total, alive, log)
        spec = dict(cr.CASE_STUDIES[spec_i], id=d.name)
        run = collect_run(spec, d, log, procs) if status == "done" else None
        ledger_path = cr.find_file([d], ("version_ledger.jsonl", "ledger.jsonl"))
        if ledger_path and ledger_path.parent.name == "brain":
            ledger_path = None
        series = climb_series(cr.read_jsonl(ledger_path))
        cands = series["candidates"]
        solver, reviewer = log_models(text)
        research = re.search(r"^Research: .*\((on|off)\)", text, re.M)
        start = started_at(log, d)
        end = log.stat().st_mtime if log.exists() and status != "running" else None
        held = heldout_summary(layer, run, d)
        heldout_points = [{"kind": b["kind"], "label": b["label"], "score": b["score"]} for b in (run or {}).get("heldout", [])]
        # The dashed baseline is v0 itself: its dev score at generation 0, as the gate measured it.
        base_dev = next((c["parent_score"] for c in cands if c["parent_score"] is not None), None)
        base_dev_label = "v0 at gen 0, dev" if base_dev is not None else None
        out.append({
            "name": d.name, "layer": layer, "layer_name": LAYER_NAMES[layer], "benchmark": benchmark,
            "status": status, "started": iso(start), "ended": iso(end), "start_ts": start,
            "solver": solver, "reviewer": reviewer, "research": bool(research and research.group(1) == "on"),
            "generations": len(series["generations"]), "candidates": len(cands),
            "accepted": sum(c["accepted"] for c in cands), "heldout": held,
            "cost_usd": total.get("cost_usd"), "calls": total.get("calls"),
            "climb": {"metric": "dev score" if layer != "inference" else "dev accuracy",
                      "baseline": base_dev, "baseline_label": base_dev_label,
                      "candidates": cands, "generations": series["generations"], "heldout": heldout_points}})
    out.sort(key=lambda r: (list(LAYER_NAMES).index(r["layer"]), r["start_ts"] or 0))
    return out


def md_num(x: float | None, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def write_runs_md(runs: list[dict], path: Path, generated: str) -> None:
    lines = ["# Runs by layer", "",
             f"Live Crusoe runs only, built by `{COMMAND}` at {generated}. Offline runs and runs with a $0 usage.json "
             "are left out. Status: done = final usage line written; running = process alive or log written in the "
             "last 3 min; crashed = Traceback without usage; stopped = none of these.", ""]
    for layer, name in LAYER_NAMES.items():
        rows = [r for r in runs if r["layer"] == layer]
        lines += [f"## {name}", ""]
        if not rows:
            lines += ["No live run.", ""]
            continue
        lines += ["| Run | Benchmark | Status | Start | End | Solver | Reviewer | Research | Gens | Scored | Accepted "
                  "| v0 cold | Final | Δ vs v0 | Cost (USD) | Calls |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            h = r["heldout"] or {}
            d = 6 if layer == "inference" else 3
            delta = "n/a" if h.get("delta") is None else f"{h['delta']:+.{d}f}"
            lines.append("| " + " | ".join(cr.cell(x) for x in [
                r["name"], r["benchmark"], r["status"], r["started"] or "n/a", r["ended"] or "n/a",
                r["solver"] or "n/a", r["reviewer"] or "n/a", "on" if r["research"] else "off", r["generations"],
                r["candidates"], r["accepted"], md_num(h.get("v0"), d), md_num(h.get("final"), d), delta,
                md_num(r["cost_usd"], 4), r["calls"] if r["calls"] is not None else "n/a"]) + " |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build(root: Path) -> dict[str, Any]:
    global ROOT
    ROOT = cr.ROOT = root
    procs = live_commands()
    runs = index_runs(procs)
    for r in runs:
        r.pop("start_ts", None)
    return {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "command": COMMAND, "pitch": PITCH,
            "layers": [hall_layer(procs), placement_layer(procs), kernel_layer(procs), inference_layer(procs)],
            "runs_index": runs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root / "dashboard" / "public" / "datacenter.json"
    data = build(args.root.resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    tmp.replace(out)
    write_runs_md(data["runs_index"], args.root / "reports" / "RUNS.md", data["generated_at"])
    for layer in data["layers"]:
        print(f"{layer['name']}: {layer['status']}; {layer['summary']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
