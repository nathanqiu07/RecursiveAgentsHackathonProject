"""Backfill an Evolver run into SA's Library: each ledger mutation becomes a paper, its Gate outcome the verdict.

  .venv/bin/python scripts/publish_run_to_library.py runs/qaplib-evolve-1 [--benchmark qaplib] [--url URL]
  .venv/bin/python scripts/publish_run_to_library.py reports/datahall_live_1 --benchmark datahall

Reads versions/ledger.jsonl, ledger.jsonl or version_ledger.jsonl under the run directory. Safe to re-run on a
run that is still going: rows already published (recorded in library.jsonl beside the ledger) are skipped.
Prints the papers and verdicts read back from the Library's own API.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team.evolve import Gate, paper  # noqa: E402
from team.sa_library import DEFAULT_URL, LibraryPublisher, row_key  # noqa: E402

LEDGERS = ("versions/ledger.jsonl", "ledger.jsonl", "version_ledger.jsonl")


def find_ledger(run_dir: Path) -> Path:
    for name in LEDGERS:
        if (run_dir / name).exists():
            return run_dir / name
    raise SystemExit(f"no ledger ({', '.join(LEDGERS)}) under {run_dir}")


def publish(rows: list[dict], publisher: LibraryPublisher, gate: Gate) -> list[dict]:
    context = {"min_gain": gate.min_gain, "min_cost_cut": gate.min_cost_cut}
    out = []
    for row in rows:
        if "dispute" in row:
            verdict = publisher.publish_dispute(row)
            key = row_key(row["generation"], dispute=row["dispute"])
        elif "mutation" in row:
            g, parent, m = row["generation"], row.get("parent"), row["mutation"]
            key = row_key(g, parent, m)
            if str(row.get("reason", "")).startswith("invalid"):
                verdict = publisher.publish_invalid(g, parent, m, row["reason"], context)
            else:
                [paper_id] = publisher.publish_claims(g, parent, [m], {**context, "n_dev": row.get("n")})
                verdict = publisher.post_verdict(paper_id, m, row, row.get("paper") or paper(m, row))
        else:
            continue
        out.append({"key": key, "paper_id": publisher.papers.get(key, {}).get("paper_id"), "verdict": verdict})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--url", default=DEFAULT_URL, help="SA's Library API")
    ap.add_argument("--benchmark", help="case-study name used in paper titles (default: run directory name)")
    ap.add_argument("--db", help="Library SQLite file (default: $SA_LIBRARY_DB or library/data/library.sqlite3)")
    args = ap.parse_args(argv)

    ledger = find_ledger(args.run_dir)
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    publisher = LibraryPublisher(args.url, args.benchmark or args.run_dir.resolve().name,
                                 state_path=ledger.parent / "library.jsonl", db_path=args.db)
    if not publisher.check():
        print(f"SA Library unreachable at {args.url}; nothing published", file=sys.stderr)
        return 1
    results = publish(rows, publisher, Gate())
    print(f"{ledger}: {len(rows)} ledger rows, {len(results)} papers; new this call: "
          f"{publisher.counts['papers']} papers, {publisher.counts['verdicts']} verdicts, "
          f"{publisher.counts['failures']} failures")
    for r in results:  # read back through SA's API, not from our own state file
        if not r["paper_id"]:
            print(f"  {r['key']}: not published")
            continue
        got = publisher.client.get(r["paper_id"])
        report = got.get("report") or {}
        print(f"  {got['id']} {got['status']:<20} {report.get('verdict', '-'):<13} {got['submission']['title'][:70]}")
    return 0 if publisher.counts["failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
