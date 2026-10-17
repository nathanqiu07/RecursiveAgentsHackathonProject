#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runtime_dir="$project_dir/data/node-venv"

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    cat <<'EOF'
Usage: ./scripts/setup-node.sh --server URL [options]

Checks and registers this machine, then runs it as a verification node.

  --server URL              Verification server URL (required)
  --node-id ID              Stable ID; generated and saved on first run if omitted
  --state-dir PATH          Local config/cache/report directory (default: data/node)
  --runner-python PATH      Python used for job scripts and optional CUDA smoke test
  --label KEY=VALUE         Node label; may be repeated
  --min-free-disk-gb N       Required available disk in GiB (default: 1)
  --min-ram-gb N             Required total RAM in GiB (default: 1)
  --require-gpu             Require NVIDIA inventory and a CUDA compute smoke test
  --check-only              Check readiness and server access without registering
  --enroll-only             Check and register, then exit without claiming jobs

Set VERIFIER_API_KEY in the environment if the server requires authentication.
Optional GPU mode requires Linux and working NVIDIA CUDA drivers. No root access is used.
The script prepares an isolated Python environment; it does not install GPU drivers.
EOF
    exit 0
fi

if ! command -v git >/dev/null 2>&1; then
    echo "Git is required. Install Git and run this script again." >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    UV_PROJECT_ENVIRONMENT="$runtime_dir" uv sync --project "$project_dir" \
        --frozen --no-dev --python 3.12
else
    if ! command -v python3 >/dev/null 2>&1; then
        echo "Install uv or Python 3.10+ and run this script again." >&2
        exit 1
    fi
    if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
        echo "Python 3.10+ is required. Install uv or a supported Python." >&2
        exit 1
    fi
    if [ ! -x "$runtime_dir/bin/python" ]; then
        python3 -m venv "$runtime_dir"
    fi
    "$runtime_dir/bin/python" -m pip install --disable-pip-version-check "$project_dir"
fi

cd "$project_dir"
exec "$runtime_dir/bin/python" -m verification_api.setup_node "$@"
