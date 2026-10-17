"""Terminal entry point: run the DoneTwin loop and stream it to the console.

    python run.py                                   # default scenario + goal
    python run.py --goal "..." --sandbox docker
    python run.py --mock                            # offline pipeline test (scripted LLM)
"""
import argparse
import json
import os
import time

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from donetwin.llm import LLM
from donetwin.loop import Run, execute
from donetwin.mock import MockLLM

console = Console()
TAG = {"spec": "magenta", "layout": "cyan", "sandbox": "blue", "verifier": "yellow", "vision": "green",
       "supervisor": "red"}


def show(ev):
    t = ev["type"]
    if t == "log":
        console.print(f"[{TAG.get(ev['agent'], 'white')}]{ev['agent']:>10}[/] | {ev['msg']}")
    elif t == "spec":
        tb = Table(title="DEFINITION OF DONE (agent-written)", show_lines=False)
        tb.add_column("check")
        tb.add_column("params")
        tb.add_column("why")
        for c in ev["spec"]["checks"]:
            tb.add_row(c["type"], json.dumps(c.get("params", {}))[:70], c.get("why", "")[:60])
        console.print(tb)
        for a in ev["spec"].get("assumptions", []):
            console.print(f"  [magenta]assumption[/] {a}")
    elif t == "iteration_start":
        console.rule(f"[bold]ITERATION {ev['iter']}")
    elif t == "layout":
        console.print(Panel(Syntax(ev["code"], "python", theme="monokai", line_numbers=True),
                            title="layout program (agent-written)", expand=False))
    elif t == "sandbox":
        status = "[green]ran OK[/]" if ev["ok"] else f"[red]FAILED[/] {ev['error']}"
        console.print(f"   sandbox ({ev.get('mode')}) {status}  {ev.get('stdout', '').strip()}")
    elif t == "verify":
        r = ev["report"]
        for c in r["checks"]:
            mark = "[green]PASS[/]" if c["passed"] else "[red]FAIL[/]"
            console.print(f"   {mark} {c['type']}")
            for v in c["violations"]:
                console.print(f"        [red]{v['detail']}[/]\n        [dim]fix: {v['fix']}[/]")
        console.print(f"   [bold]{r['passed']}/{r['total']} pass[/]  penalty={r['penalty']}")
    elif t == "vision":
        console.print(f"   inspector: [bold]{ev['verdict']}[/]")
        for i in ev.get("issues", []):
            console.print(f"     - ({i.get('severity')}) {i.get('description')}")
        for a in ev.get("adjudication", []):
            console.print(f"     [yellow]adjudication[/] {a['ruling']}: {a['measured']}")
    elif t == "decision":
        console.print(f"   [bold red]SUPERVISOR -> {ev['action'].upper()}[/]  {ev['reason']}")
    elif t == "done":
        color = "green" if ev["status"] == "verified" else "yellow"
        console.print(Panel(f"{ev['summary']}\nusage: {ev['usage']}\nUSD: {ev['usd_path']}",
                            title=f"[{color}]{ev['status'].upper()}", expand=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="sorting_cell")
    ap.add_argument("--goal", default=None)
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--sandbox", choices=["subprocess", "docker", "openshell"], default="subprocess")
    ap.add_argument("--vision", choices=["final", "every", "off"], default="final")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--pace", type=float, default=None, help="seconds between events (default 0.7 with --mock)")
    a = ap.parse_args()
    pace = a.pace if a.pace is not None else (0.7 if a.mock else 0.0)

    with open(os.path.join("scenarios", f"{a.scenario}.json"), encoding="utf-8") as f:
        base = json.load(f)
    goal = a.goal or base["goal"]
    run = Run(time.strftime("%H%M%S-cli"), goal, a.scenario)
    run.listeners.append(lambda ev: (show(ev), time.sleep(pace if ev["type"] != "log" else pace / 3)))
    console.print(Panel(goal, title="GOAL"))
    execute(run, MockLLM() if a.mock else LLM(), base, a.max_iters, a.sandbox, a.vision)
    console.print(f"artifacts: runs/{run.run_id}/")


if __name__ == "__main__":
    main()
