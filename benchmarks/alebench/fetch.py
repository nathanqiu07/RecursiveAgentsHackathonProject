"""Download one ALE-Bench problem (AHC039) unmodified, then build the reference greedy's per-seed scores.

ALE-Bench (SakanaAI, huggingface.co/datasets/SakanaAI/ALE-Bench): code Apache-2.0, problem data
CC BY-ND 4.0, statements and tools by AtCoder Inc. The repo commits only this script and our own code;
the zip is fetched at runtime, checked against a pinned SHA-256, extracted untouched into the
git-ignored `problems/` directory, and never edited.
Run: .venv/bin/python -m benchmarks.alebench.fetch [--no-reference]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.alebench import scorer  # noqa: E402

URL = f"https://huggingface.co/datasets/SakanaAI/ALE-Bench/resolve/main/{scorer.PROBLEM}.zip"
SHA256 = "34d478f166334d6c8a0f7e0d03f6b5c17c557eeff9f163d5510ecc8605a9f3cd"
MAX_BYTES = 20 * 2**20


def _get(url: str) -> bytes:
    """python.org macOS builds ship no CA bundle, so prefer certifi, then fall back to curl's system store."""
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(url, timeout=60, context=context) as resp:
            return resp.read(MAX_BYTES + 1)
    except urllib.error.URLError as exc:
        print(f"urllib failed for {url} ({exc.reason}); retrying with curl")
        return subprocess.run(["curl", "-sSfL", "-m", "120", "--max-filesize", str(MAX_BYTES), url], check=True,
                              capture_output=True).stdout


def download(url: str = URL) -> bytes:
    body = _get(url)
    if len(body) > MAX_BYTES:
        raise RuntimeError(f"{url} is larger than {MAX_BYTES} bytes")
    digest = hashlib.sha256(body).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"{url} sha256 {digest} != pinned {SHA256}; refusing to extract")
    return body


def safe_extract(body: bytes, dest: Path) -> None:
    """Extract without following absolute paths or `..`, since the archive is downloaded data."""
    dest = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if dest not in target.parents and target != dest:
                raise RuntimeError(f"unsafe path in archive: {info.filename!r}")
        zf.extractall(dest)


def fetch(force: bool = False) -> Path:
    if (scorer.PROBLEM_DIR / "data.json").is_file() and not force:
        return scorer.PROBLEM_DIR
    scorer.PROBLEMS.mkdir(parents=True, exist_ok=True)
    safe_extract(download(), scorer.PROBLEMS / scorer.PROBLEM)
    scorer.metadata.cache_clear()
    return scorer.PROBLEM_DIR


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if the data is present")
    ap.add_argument("--no-reference", action="store_true", help="skip running the greedy on every split seed")
    args = ap.parse_args()
    path = fetch(args.force)
    print(f"{scorer.PROBLEM} data at {path}")
    if not args.no_reference:
        from benchmarks.alebench.splits import all_seeds
        ref = scorer.reference_scores(all_seeds())
        print(f"reference greedy scored on {len(ref)} seeds -> {scorer.REFERENCE_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
