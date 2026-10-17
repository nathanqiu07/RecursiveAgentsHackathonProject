#!/bin/sh
# Start SA's agent harness stack on localhost: You.com search shim (:8011),
# harness API (:8003) and harness model worker. Logs go to runs/localhost/.
#
# Keys come ONLY from the caller's environment; nothing here embeds them:
#   HARNESS_MODEL_API_KEY  Crusoe inference key
#   YDC_API_KEY            You.com search key (read by the shim only)
# Usage:
#   HARNESS_MODEL_API_KEY=... YDC_API_KEY=... sh scripts/start_sa_harness.sh
#
# Expects already running: dashboard API :3001, Library :8001, and a
# ResourceManager at HARNESS_RESOURCE_MANAGER_URL (default :8002). The
# ResourceManager source is not in this repo (/ResourceManager/ is gitignored)
# and it needs Docker for its sandboxes; without it runs fail at allocation.
# Do not set HARNESS_API_KEY: the dashboard proxies :8003 without a key.
set -eu

ROOT=/Users/iamdaone/Documents/RecursiveAgentsHackathonProject
PY=$ROOT/harness/.venv/bin/python
LOGS=$ROOT/runs/localhost
CA=$($PY -c 'import certifi; print(certifi.where())')

: "${HARNESS_MODEL_API_KEY:?set HARNESS_MODEL_API_KEY in the environment}"
: "${YDC_API_KEY:?set YDC_API_KEY in the environment}"
# Drop any log line carrying either key. awk reads the keys from its environment, so they never
# appear in this file or in any process's argument list.
FILTER="awk 'BEGIN{a=ENVIRON[\"HARNESS_MODEL_API_KEY\"]; b=ENVIRON[\"YDC_API_KEY\"]} index(\$0,a)==0 && index(\$0,b)==0 {print; fflush()}'"

# One-time venv (no uv on this machine): python3.12 -m venv + pip install deps.
if [ ! -x "$PY" ]; then
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 -m venv "$ROOT/harness/.venv"
  "$ROOT/harness/.venv/bin/pip" install -q "fastapi>=0.115,<1" "httpx>=0.28,<1" \
    "jsonschema[format]>=4.23,<5" "uvicorn>=0.34,<1"
fi
mkdir -p "$LOGS"

export HARNESS_DB_PATH=$LOGS/harness.sqlite3
export HARNESS_LIBRARY_URL=http://127.0.0.1:8001
export HARNESS_BENCHMARK_URL=http://127.0.0.1:3001
export HARNESS_RESOURCE_MANAGER_URL=${HARNESS_RESOURCE_MANAGER_URL:-http://127.0.0.1:8002}
export HARNESS_MODEL_BASE_URL=https://api.inference.crusoecloud.com/v1
export HARNESS_MODEL=${HARNESS_MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B}
export HARNESS_SEARCH_URL=http://127.0.0.1:8011/search
export HARNESS_MODEL_API_KEY YDC_API_KEY

# Search shim gets the You.com key; harness processes do not need it.
nohup sh -c "SSL_CERT_FILE=$CA $PY -u $ROOT/scripts/ydc_search_shim.py 2>&1 | $FILTER > $LOGS/ydc-shim.log" >/dev/null 2>&1 &

nohup sh -c "env -u YDC_API_KEY $PY -m uvicorn --app-dir $ROOT/harness agent_harness.main:app --host 127.0.0.1 --port 8003 2>&1 | $FILTER > $LOGS/harness-api.log" >/dev/null 2>&1 &

nohup sh -c "env -u YDC_API_KEY PYTHONPATH=$ROOT/harness $PY -u -m agent_harness.worker 2>&1 | $FILTER > $LOGS/harness-worker.log" >/dev/null 2>&1 &

echo "started: ydc shim :8011, harness API :8003, harness worker (logs in $LOGS)"
