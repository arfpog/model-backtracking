#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-backtracking}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

# Initialize conda for the current shell
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION"
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

echo "Environment '$ENV_NAME' ready. Activate via: conda activate $ENV_NAME"
