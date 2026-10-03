#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=${VENV_PATH:-.venv}
PYTHON_COMMAND=${PYTHON_COMMAND:-python3}

"$PYTHON_COMMAND" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -e .

echo "Environment ready. Before submitting, run:"
echo "export PYTHON_BIN=$(pwd)/$VENV_PATH/bin/python"
echo "export HF_HOME=\${SCRATCH:-$(pwd)}/huggingface"

