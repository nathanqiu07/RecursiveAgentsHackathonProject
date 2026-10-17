"""HTTP verifier for QAPLIB, for the team dashboard's benchmark contract (stdlib only).

  .venv/bin/python -I benchmarks/qaplib/verifier_api.py [--port 9001]

Wraps `benchmarks.qaplib.verify`, which recomputes the objective exactly from the instance's .dat file.
No score is ever taken from the request; it is always recomputed here.

POST /evaluations
  {"submission": {"instance": "nug22", "permutation": [...]}}
      -> 200 {"status": "completed", "score": BKS/cost, "cost", "bks", "gap_pct", "feedback"}
  {"submission": {"solutions": [{"instance", "permutation"}, ...]}, "split": "heldout"}
      -> 200 {"status": "completed", "score": mean BKS/cost over every instance of the split, ...}
         An instance of the split that is missing or has an invalid permutation scores 0, so a partial
         set cannot raise the mean. Without "split", the mean is over the submitted instances.
  "feedback" opens with two plain sentences comparing the answer to the 2-swap baseline in reports/qaplib_baseline.json
  ("baseline": {"name", "score", "is_baseline"}, or null for an instance it does not cover), then "Details:".
  Unknown instance, unknown split, duplicate instance or malformed body -> 400/422 {"status": "rejected", "error"}
GET /health -> {"ok": true, "instances": N}
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.qaplib.verify import INSTANCES, load_index, verify  # noqa: E402

MAX_BODY = 4 * 1024 * 1024
BASELINE_REPORT = ROOT / "reports" / "qaplib_baseline.json"
TWO_SWAP_CACHE = ROOT / "runs" / "dashboard" / "qaplib_two_swap_heldout.json"


class Rejected(ValueError):
    """The request cannot be scored at all (as opposed to a valid request with an invalid answer, which scores 0)."""


def load_split(name: str) -> list[str]:
    splits = json.loads((ROOT / "benchmarks" / "qaplib" / "splits.json").read_text())
    if name not in ("train", "dev", "heldout"):
        raise Rejected(f"unknown split {name!r}")
    return list(splits[name])


def score_one(instance: Any, permutation: Any, index: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(instance, str) or instance not in index:
        raise Rejected(f"unknown instance {instance!r}")
    if isinstance(permutation, bool) or not isinstance(permutation, (list, str)):
        return {"instance": instance, "status": "invalid", "error": "permutation must be a list of integers",
                "score": 0.0, "cost": None, "bks": index[instance]["bks"], "gap_pct": None}
    if isinstance(permutation, list) and not all(isinstance(x, int) and not isinstance(x, bool) for x in permutation):
        return {"instance": instance, "status": "invalid", "error": "permutation must contain only integers",
                "score": 0.0, "cost": None, "bks": index[instance]["bks"], "gap_pct": None}
    v = verify(instance, permutation)
    v.pop("perm", None)
    return {"instance": instance, **v}


def line(r: dict[str, Any]) -> str:
    if r["status"] == "ok":
        return f"{r['instance']}: cost {r['cost']} vs BKS {r['bks']}, score {r['score']:.6f}, gap {r['gap_pct']:.3f}%"
    return f"{r['instance']}: {r['status']} ({r.get('error', '')}), score 0"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@lru_cache(maxsize=4)
def _two_swap_baseline(_report_mtime: float, _cache_mtime: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        inst = json.loads(BASELINE_REPORT.read_text())["instances"]
        out = {k: v["two_swap"] for k, v in inst.items() if isinstance(v.get("two_swap"), dict)}
    except (OSError, ValueError, KeyError):
        pass
    try:  # the exact 2-swap permutations on the dashboard's baseline row win, so that row reads as the baseline
        cached = json.loads(TWO_SWAP_CACHE.read_text())
    except (OSError, ValueError):
        cached = {}
    for name, perm in cached.items():
        v = verify(name, perm)
        if v.get("status") == "ok":
            out[name] = {"cost": v["cost"], "score": v["score"], "gap_pct": v["gap_pct"]}
    return out


def two_swap_baseline() -> dict[str, dict[str, Any]]:
    """The baseline: 2-swap local search per instance ({cost, score, gap_pct}), from reports/qaplib_baseline.json,
    overridden by the permutations scripts/publish_results_to_dashboard.py submitted as the board's baseline."""
    return _two_swap_baseline(_mtime(BASELINE_REPORT), _mtime(TWO_SWAP_CACHE))


def pct(gap: float) -> str:
    return f"{gap:.2f}%" if abs(gap) < 10 else f"{gap:.1f}%"


def vs_best(gap: float) -> str:
    return "the same as" if round(gap, 2) == 0 else f"{pct(abs(gap))} {'more' if gap > 0 else 'less'} than"


def explain(results: list[dict[str, Any]]) -> str:
    """Two plain sentences: the average cost gap to the best known answers, against the 2-swap baseline."""
    n = len(results)
    ok = [r for r in results if r["status"] == "ok" and r.get("gap_pct") is not None]
    base = two_swap_baseline()
    have_base = bool(ok) and all(r["instance"] in base for r in results)
    bad = n - len(ok)
    if not ok:
        return f"No valid answer in {n} problem{'s' if n > 1 else ''}, so it scores 0. Each answer must assign every facility to a different location."
    gap = sum(r["gap_pct"] for r in ok) / len(ok)
    where = f"for {ok[0]['instance']}" if n == 1 else f"on average over {len(ok)} problems"
    answers = "answer" if n == 1 else "answers"
    at_bks = sum(1 for r in ok if r["cost"] <= r["bks"])
    if have_base and not bad and all(r["cost"] == base[r["instance"]]["cost"] for r in ok):
        return (f"This is the baseline: 2-swap local search costs {vs_best(gap)} the best known {answers} {where}. "
                f"It matches the best known answer on {at_bks} of {n}; a higher score (1.0 = best known) means cheaper "
                "assignments.")
    else:
        first = f"Costs {vs_best(gap)} the best known {answers} {where}"
        if have_base:
            bgap = sum(base[r["instance"]]["gap_pct"] for r in ok) / len(ok)
            first += (f"; the 2-swap baseline matches {'it' if n == 1 else 'them'}" if round(bgap, 2) == 0 else
                      f"; the 2-swap baseline is {pct(bgap)} above {'it' if n == 1 else 'them'}")
        first += "."
    if n == 1:
        second = "It matches the best known answer." if at_bks else (
            "It beats 2-swap." if have_base and ok[0]["cost"] < base[ok[0]["instance"]]["cost"] else
            "It does not beat 2-swap." if have_base else "Lower cost is better.")
        return f"{first} {second}"
    second = f"It matches the best known answer on {at_bks} of {n}"
    if have_base:
        wins = sum(1 for r in ok if r["cost"] < base[r["instance"]]["cost"])
        losses = sum(1 for r in ok if r["cost"] > base[r["instance"]]["cost"])
        second += f" and beats 2-swap on {wins}, ties {len(ok) - wins - losses}, loses {losses}"
    if bad:
        second += f"; {bad} {'is' if bad == 1 else 'are'} missing or invalid and score 0"
    return f"{first} {second}."


def baseline_info(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    base = two_swap_baseline()
    if not all(r["instance"] in base for r in results):
        return None
    return {"name": "2-swap", "score": sum(float(base[r["instance"]]["score"]) for r in results) / len(results),
            "is_baseline": all(r["status"] == "ok" and r["cost"] == base[r["instance"]]["cost"] for r in results)}


def evaluate(req: Any) -> dict[str, Any]:
    if not isinstance(req, dict) or not isinstance(req.get("submission"), (dict, str)):
        raise Rejected("body must be a JSON object with a 'submission' object")
    sub = req["submission"]
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except ValueError as exc:
            raise Rejected(f"submission is not valid JSON: {exc}") from None
        if not isinstance(sub, dict):
            raise Rejected("submission must be an object")
    index = load_index()
    if "solutions" not in sub:
        r = score_one(sub.get("instance"), sub.get("permutation"), index)
        return {"status": "completed", "score": float(r["score"]), "cost": r["cost"], "bks": r["bks"],
                "gap_pct": r["gap_pct"], "results": [r], "baseline": baseline_info([r]),
                "feedback": explain([r]) + "\n\nDetails:\n" + line(r)}
    sols = sub["solutions"]
    if not isinstance(sols, list) or not sols:
        raise Rejected("'solutions' must be a non-empty list")
    by_name: dict[str, Any] = {}
    for s in sols:
        if not isinstance(s, dict):
            raise Rejected("each solution must be an object with 'instance' and 'permutation'")
        name = s.get("instance")
        if name in by_name:
            raise Rejected(f"duplicate instance {name!r}")
        by_name[name] = s.get("permutation")
    split = req.get("split")
    names = load_split(split) if split else list(by_name)
    extra = [n for n in by_name if n not in names]
    if extra:
        raise Rejected(f"instances outside split {split!r}: {extra}")
    results = []
    for name in names:
        if name not in by_name:
            results.append({"instance": name, "status": "missing", "error": "no permutation submitted",
                             "score": 0.0, "cost": None, "bks": index[name]["bks"], "gap_pct": None})
        else:
            results.append(score_one(name, by_name[name], index))
    mean = sum(r["score"] for r in results) / len(results)
    ok = [r for r in results if r["status"] == "ok"]
    at_bks = sum(1 for r in ok if r["cost"] <= r["bks"])
    head = (f"mean BKS/cost {mean:.6f} over {len(results)} instances"
            f"{' of split ' + split if split else ''}; {len(ok)} valid, {at_bks} at or below BKS")
    return {"status": "completed", "score": mean, "n": len(results), "valid": len(ok), "at_bks": at_bks,
            "results": results, "baseline": baseline_info(results),
            "feedback": explain(results) + "\n\nDetails:\n" + "\n".join([head] + [line(r) for r in results])}


# ---------------------------------------------------------------- GPU rack placement (data-center QAP)
# POST /dc/evaluations: the /evaluations contract for benchmarks/qaplib/dc_instances (same .dat format, same exact
# verify()). "bks" is each instance's stored reference cost (best of robust tabu and 2-swap), and the baseline is
# its stored 2-swap permutation. "split" names a dc split (train, dev, heldout).
DC_PATH = "/dc/evaluations"


def _dc_score_one(name: Any, permutation: Any, index: dict[str, Any]) -> dict[str, Any]:
    from benchmarks.qaplib.dc_instances import DC_DIR

    if not isinstance(name, str) or name not in index:
        raise Rejected(f"unknown data-center instance {name!r}")
    bks = index[name]["bks"]
    if isinstance(permutation, bool) or not isinstance(permutation, (list, str)) or (
            isinstance(permutation, list)
            and not all(isinstance(x, int) and not isinstance(x, bool) for x in permutation)):
        return {"instance": name, "status": "invalid", "error": "permutation must be a list of integers",
                "score": 0.0, "cost": None, "bks": bks, "gap_pct": None}
    v = verify(name, permutation, bks=bks, dat_path=DC_DIR / f"{name}.dat")
    v.pop("perm", None)
    return {"instance": name, **v}


def _dc_explain(results: list[dict[str, Any]], index: dict[str, Any]) -> str:
    """Two plain sentences: cost against the reference placements, then against the 2-swap baseline."""
    n = len(results)
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return (f"No valid placement in {n} hall{'s' if n > 1 else ''}, so it scores 0. Each answer must put every "
                "rack in a different slot.")
    gap = sum(r["gap_pct"] for r in ok) / len(ok)
    where = f"for {ok[0]['instance']}" if n == 1 else f"on average over {len(ok)} halls"
    size = f"{abs(gap):.2f}%"
    first = ("Traffic x cable distance matches the reference placement " + where if round(gap, 2) == 0 else
             f"Traffic x cable distance is {size} {'above' if gap > 0 else 'below'} the reference placement {where}")
    base = {r["instance"]: index[r["instance"]]["two_swap"]["cost"] for r in results}
    if all(r["status"] == "ok" and r["cost"] == base[r["instance"]] for r in results):
        return (f"This is the baseline: 2-swap local search. {first}; a higher score (1.0 = reference, from a "
                "time-boxed robust tabu search) means less traffic crossing long cables.")
    wins = sum(1 for r in ok if r["cost"] < base[r["instance"]])
    losses = sum(1 for r in ok if r["cost"] > base[r["instance"]])
    bad = n - len(ok)
    second = f"It beats 2-swap on {wins}, ties {len(ok) - wins - losses}, loses {losses}"
    if bad:
        second += f"; {bad} {'is' if bad == 1 else 'are'} missing or invalid and score 0"
    return f"{first}. {second}."


def evaluate_dc(req: Any) -> dict[str, Any]:
    from benchmarks.qaplib.dc_instances import load_index as dc_index, load_splits as dc_splits

    if not isinstance(req, dict) or not isinstance(req.get("submission"), (dict, str)):
        raise Rejected("body must be a JSON object with a 'submission' object")
    sub = req["submission"]
    if isinstance(sub, str):
        try:
            sub = json.loads(sub)
        except ValueError as exc:
            raise Rejected(f"submission is not valid JSON: {exc}") from None
        if not isinstance(sub, dict):
            raise Rejected("submission must be an object")
    index = dc_index()
    sols = sub["solutions"] if "solutions" in sub else [sub]
    if not isinstance(sols, list) or not sols or not all(isinstance(s, dict) for s in sols):
        raise Rejected("'solutions' must be a non-empty list of {instance, permutation} objects")
    by_name: dict[str, Any] = {}
    for s in sols:
        if s.get("instance") in by_name:
            raise Rejected(f"duplicate instance {s.get('instance')!r}")
        by_name[s.get("instance")] = s.get("permutation")
    split = req.get("split")
    if split is not None and split not in ("train", "dev", "heldout"):
        raise Rejected(f"unknown split {split!r}")
    names = list(dc_splits()[split]) if split else list(by_name)
    extra = [n for n in by_name if n not in names]
    if extra:
        raise Rejected(f"instances outside split {split!r}: {extra}")
    results = [_dc_score_one(n, by_name[n], index) if n in by_name else
               {"instance": n, "status": "missing", "error": "no permutation submitted", "score": 0.0, "cost": None,
                "bks": index[n]["bks"], "gap_pct": None} for n in names]
    mean = sum(r["score"] for r in results) / len(results)
    ok = [r for r in results if r["status"] == "ok"]
    base = [index[r["instance"]]["bks"] / index[r["instance"]]["two_swap"]["cost"] for r in results]
    head = (f"mean reference/cost {mean:.6f} over {len(results)} halls"
            f"{' of split ' + split if split else ''}; {len(ok)} valid")
    details = [head] + [(f"{r['instance']}: cost {r['cost']} vs reference {r['bks']}, score {r['score']:.6f}, "
                         f"gap {r['gap_pct']:.3f}%") if r["status"] == "ok" else
                        f"{r['instance']}: {r['status']} ({r.get('error', '')}), score 0" for r in results]
    return {"status": "completed", "score": mean, "n": len(results), "valid": len(ok),
            "cost": results[0]["cost"] if len(results) == 1 else None, "results": results,
            "baseline": {"name": "2-swap", "score": sum(base) / len(base),
                         "is_baseline": all(r["status"] == "ok" and r["cost"] == index[r["instance"]]["two_swap"]["cost"]
                                            for r in results)},
            "feedback": _dc_explain(results, index) + "\n\nDetails:\n" + "\n".join(details)}


class Handler(BaseHTTPRequestHandler):
    server_version = "QaplibVerifier/1.0"

    def _send(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0].rstrip("/") == "/health":
            return self._send(200, {"ok": True, "instances": len(load_index())})
        self._send(404, {"status": "rejected", "error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/evaluations", DC_PATH):
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
            self._send(200, evaluate_dc(req) if path == DC_PATH else evaluate(req))
        except Rejected as exc:
            self._send(422, {"status": "rejected", "error": str(exc)})
        except Exception as exc:  # a verifier bug must surface as an error, never as a score
            self._send(500, {"status": "failed", "error": f"could not score submission: {exc!r}"})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[qaplib-verifier] " + (fmt % args) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9001)
    args = ap.parse_args()
    if not (INSTANCES / "index.json").is_file():
        sys.exit(f"missing {INSTANCES / 'index.json'}; run benchmarks/qaplib/fetch.py first")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"qaplib verifier on http://127.0.0.1:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
