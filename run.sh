#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="${SBMEPST_PYTHON:-/home/wangzhenyao/.conda/envs/wzy-torch-py311/bin/python}"
command -v "$PY" >/dev/null 2>&1 || PY=python3

exec "$PY" eval_test.py "$@"
