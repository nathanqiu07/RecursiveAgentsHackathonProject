#!/bin/sh
set -eu
NODE_SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$NODE_SCRIPT_DIR/controller.py" enroll "$@"
