#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/pilot.yaml}
mkdir -p slurm/logs slurm/state

PROJECT_ROOT=$(pwd -P)
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
export HALOSCOPE_DIR PYTHON_BIN PEFT_OVERLAY
export PYTHONPATH="$PROJECT_ROOT/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  echo "Set HALOSCOPE_PYTHON to the correct absolute path and retry." >&2
  exit 2
fi
if [[ ! -d "$PEFT_OVERLAY" ]]; then
  echo "Dependency overlay does not exist: $PEFT_OVERLAY" >&2
  echo "Set PEFT_OVERLAY to the directory used during installation and retry." >&2
  exit 2
fi

"$PYTHON_BIN" -c "import peft_probe; print('Submission Python:', __import__('sys').executable)"

CONFIG_NAME=$(basename "$CONFIG")
RUN_NAME=${CONFIG_NAME%.*}
STATE_FILE=${PIPELINE_STATE_FILE:-slurm/state/${RUN_NAME}.step}

if [[ "${RESET_PIPELINE:-0}" == "1" ]]; then
  rm -f "$STATE_FILE" "${STATE_FILE}.tmp" "${STATE_FILE}.done"
fi

ACCOUNT_ARGS=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  ACCOUNT_ARGS=(--account "$SLURM_ACCOUNT")
fi

JOB_ID=$(sbatch --parsable --export=ALL,PIPELINE_STATE_FILE="$STATE_FILE" \
  --output=slurm/logs/%x-%j.out --error=slurm/logs/%x-%j.err \
  "${ACCOUNT_ARGS[@]}" --job-name=memprobe-serial \
  slurm/serial_pipeline.sbatch "$CONFIG")

cat <<EOF
Submitted one self-requeuing controlled-memory pipeline job: $JOB_ID
State file: $STATE_FILE
Monitor with: squeue -j $JOB_ID
Log: slurm/logs/memprobe-serial-$JOB_ID.out
EOF
