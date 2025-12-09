#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/models.yaml}"
MODEL_DIR="${MODEL_DIR:-models}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/pull_models.py"

if ! command -v python >/dev/null 2>&1; then
  echo "Python is required to run ${PY_SCRIPT}." >&2
  exit 1
fi

python "$PY_SCRIPT" --config "$CONFIG_PATH" --out "$MODEL_DIR"
