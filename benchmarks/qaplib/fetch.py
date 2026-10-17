"""Download QAPLIB, validate every best-known solution, and install the selected instances.

QAPLIB (Burkard, Cela, Karisch, Rendl; coral.ise.lehigh.edu/data-sets/qaplib) states no license,
so the repo commits only this script and `instances/index.json` (names, BKS, SHA-256);
the .dat/.sln files are fetched locally and git-ignored by `instances/.gitignore`.
Run: .venv/bin/python -m benchmarks.qaplib.fetch [--write-index]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import ssl
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.qaplib.verify import INSTANCES, cost, parse_dat, parse_sln  # noqa: E402

BASE = "https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/"
ARCHIVES = {"dat": BASE + "qapdata.tar.gz", "sln": BASE + "qapsoln.tar.gz"}

# 45 instances, 15 families, n 15..100. esc is left out (2-swap hits every BKS in <0.5 s and BKS
# values like 2 or 6 make bks/cost degenerate); had keeps only had20 as an easy calibration point.
SELECTED = [
    "bur26a", "chr15c", "chr18b", "chr20b", "chr22a", "chr25a", "els19", "had20", "kra30a", "kra30b",
    "lipa20a", "lipa40b", "lipa50a", "lipa70a", "lipa90a", "nug17", "nug20", "nug22", "nug25", "nug30",
    "rou15", "rou20", "scr20", "sko42", "sko56", "sko64", "sko81", "sko100a", "ste36a", "ste36c",
    "tai15a", "tai20a", "tai35a", "tai50a", "tai60a", "tai80a", "tai100a",
    "tai20b", "tai35b", "tai50b", "tai80b", "tho30", "tho40", "wil50", "wil100",
]

# Instances the QAPLIB solutions page lists with a lower bound and a gap, i.e. BKS not proven optimal.
OPEN = {"sko42", "sko56", "sko64", "sko81", "sko100a", "tai35a", "tai50a", "tai60a", "tai80a", "tai100a",
        "tai35b", "tai50b", "tai80b", "tho40", "wil50", "wil100"}

FAMILY_NOTES = {
    "bur": "Burkard-Offermann typewriter keyboard design",
    "chr": "Christofides-Benavent, flow matrix is a weighted tree",
    "els": "Elshafei hospital department layout",
    "had": "Hadley-Rendl-Wolkowicz, Manhattan grid distances",
    "kra": "Krarup-Pruzan hospital planning",
    "lipa": "Li-Pardalos generator with known optimum",
    "nug": "Nugent, Manhattan distances on a rectangular grid",
    "rou": "Roucairol, uniform random flows and distances",
    "scr": "Scriabin-Vergin, rectangular grid distances",
    "sko": "Skorin-Kapov, rectangular grid distances, random flows",
    "ste": "Steinberg backboard wiring",
    "tai-a": "Taillard, dense uniform random matrices (hard for local search)",
    "tai-b": "Taillard, real-life-like clustered flows and distances",
    "tho": "Thonemann-Boelte, rectangular grid distances",
    "wil": "Wilhelm-Ward, rectangular grid distances",
}


def family(name: str) -> str:
    """Taillard a/b are different generators, so they are separate families."""
    prefix = re.match(r"[a-z]+", name).group(0)
    return f"tai-{name[-1]}" if prefix == "tai" else prefix


def download(url: str) -> bytes:
    """The server returns 403 to the default urllib agent; python.org macOS builds ship no CA bundle, so prefer certifi."""
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    agent = "Mozilla/5.0 (qaplib-benchmark fetch)"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": agent})
        with urllib.request.urlopen(req, timeout=60, context=context) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        # The server's certificate chain fails Python's verifier on some hosts while curl's system store accepts it.
        print(f"urllib failed for {url} ({exc.reason}); retrying with curl")
        return subprocess.run(["curl", "-sSfL", "-m", "120", "-A", agent, url], check=True, capture_output=True).stdout


def members(archive: bytes, suffix: str) -> dict[str, bytes]:
    """Read regular files by basename only, never extracting paths from the archive."""
    out = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for m in tar.getmembers():
            base = Path(m.name).name
            if m.isfile() and base.endswith(suffix) and re.fullmatch(r"[a-z0-9]+\.[a-z]+", base):
                out[base[: -len(suffix)]] = tar.extractfile(m).read()
    return out


def check_solution(dat_path: Path, sln_path: Path) -> tuple[str | None, int]:
    """Return the convention ("direct" or "inverse") under which the .sln permutation reproduces its value.

    A few QAPLIB .sln files list p^-1; the optimum value of QAP(A,B) is unchanged, so the BKS stays valid.
    """
    n, a, b = parse_dat(dat_path)
    _, value, perm = parse_sln(sln_path)
    if sorted(perm) != list(range(n)):
        return None, value
    if cost(a, b, perm) == value:
        return "direct", value
    inverse = [0] * n
    for i, x in enumerate(perm):
        inverse[x] = i
    return ("inverse" if cost(a, b, inverse) == value else None), value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-index", action="store_true", help="(re)write index.json instead of checking against it")
    args = ap.parse_args()
    dats, slns = members(download(ARCHIVES["dat"]), ".dat"), members(download(ARCHIVES["sln"]), ".sln")
    INSTANCES.mkdir(parents=True, exist_ok=True)
    (INSTANCES / ".gitignore").write_text("*.dat\n*.sln\n")
    failures = []
    entries = []
    for name in sorted(slns):
        if name not in dats:
            continue
        dat_path, sln_path = INSTANCES / f"{name}.dat", INSTANCES / f"{name}.sln"
        dat_path.write_bytes(dats[name])
        sln_path.write_bytes(slns[name])
        convention, value = check_solution(dat_path, sln_path)
        if convention is None:
            failures.append({"name": name, "reason": f".sln permutation does not reproduce stated value {value}"})
        if name not in SELECTED or convention is None:
            dat_path.unlink()
            sln_path.unlink()
            continue
        n = parse_dat(dat_path)[0]
        fam = family(name)
        entries.append({"name": name, "n": n, "family": fam, "bks": value, "bks_proven_optimal": name not in OPEN,
                        "sln_convention": convention, "comment": FAMILY_NOTES.get(fam, ""),
                        "sha256": hashlib.sha256(dats[name]).hexdigest(),
                        "sln_sha256": hashlib.sha256(slns[name]).hexdigest()})
    missing = sorted(set(SELECTED) - {e["name"] for e in entries})
    print(f"validated {len(slns)} .sln files: {len(failures)} failed {[f['name'] for f in failures]}")
    print(f"installed {len(entries)} instances; missing from selection: {missing}")
    index_path = INSTANCES / "index.json"
    if args.write_index or not index_path.exists():
        index_path.write_text(json.dumps({"source": "QAPLIB, coral.ise.lehigh.edu/data-sets/qaplib",
                                          "archives": ARCHIVES, "excluded_bad_sln": failures,
                                          "instances": entries}, indent=1) + "\n")
        print(f"wrote {index_path}")
        return 0 if not missing else 1
    expected = {e["name"]: e["sha256"] for e in json.loads(index_path.read_text())["instances"]}
    drift = [e["name"] for e in entries if expected.get(e["name"]) != e["sha256"]]
    if drift or missing:
        print(f"ERROR: downloaded files differ from index.json for {drift}; missing {missing}")
        return 1
    print("all instance hashes match index.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
