#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: bash slurm/submit_token_position_pipeline.sh CONFIG.yaml}
mkdir -p slurm/logs slurm/state

PROJECT_ROOT=$(pwd -P)
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
export HALOSCOPE_DIR PYTHON_BIN PEFT_OVERLAY
export PYTHONPATH="$PROJECT_ROOT/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" -c "import peft_probe; print('Submission Python:', __import__('sys').executable)"

CONFIG_NAME=$(basename "$CONFIG")
RUN_NAME=${CONFIG_NAME%.*}
STATE_ID=$("$PYTHON_BIN" -c \
  'import sys; from peft_probe.config import config_fingerprint, load_config; from peft_probe.versioning import PIPELINE_SCHEMA_VERSION; print(f"{config_fingerprint(load_config(sys.argv[1]))}-v{PIPELINE_SCHEMA_VERSION}")' \
  "$CONFIG")
STATE_FILE=${PIPELINE_STATE_FILE:-slurm/state/${RUN_NAME}-${STATE_ID}.token-position.step}

if [[ "${RESET_PIPELINE:-0}" == "1" ]]; then
  rm -f "$STATE_FILE" "${STATE_FILE}.tmp" "${STATE_FILE}.done"
fi

ACCOUNT_ARGS=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  ACCOUNT_ARGS=(--account "$SLURM_ACCOUNT")
fi

JOB_ID=$(sbatch --parsable --export=ALL,PIPELINE_STATE_FILE="$STATE_FILE" \
  --output=slurm/logs/%x-%j.out --error=slurm/logs/%x-%j.err \
  "${ACCOUNT_ARGS[@]}" \
  --job-name=memprobe-token-position \
  slurm/token_position_pipeline.sbatch "$CONFIG")

echo "Submitted one resumable token-position job: $JOB_ID"
echo "State file: $STATE_FILE"
echo "Monitor with: squeue -j $JOB_ID"
echo "Log: slurm/logs/memprobe-token-position-$JOB_ID.out"
