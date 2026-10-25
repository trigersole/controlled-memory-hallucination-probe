#!/usr/bin/env bash
#SBATCH --job-name=popqa_base_support
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/popqa-base-support-%j.out
#SBATCH --error=slurm/logs/popqa-base-support-%j.err

set -euo pipefail

CONFIG=${1:?Pass the existing experiment config}
TRANSFER_CONFIG=${2:-configs/popqa_base_contextual_support.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
TOTAL_STEPS=3

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs slurm/state
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

STATE_ID=$("$PYTHON_BIN" -c \
  'import sys
from peft_probe.base_contextual_transfer import (
    base_contextual_transfer_fingerprint,
    load_base_contextual_transfer_config,
)
from peft_probe.config import load_config
c = load_config(sys.argv[1])
s = load_base_contextual_transfer_config(sys.argv[2])
print(base_contextual_transfer_fingerprint(c, s))' \
  "$CONFIG" "$TRANSFER_CONFIG")
STATE_FILE="slurm/state/popqa-base-support-${STATE_ID}.step"

requeue_current() {
  echo "Time-limit warning received; requeuing at step ${STEP:-unknown}."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ -f "${STATE_FILE}.done" ]]; then
  echo "PopQA base-contextual transfer is already complete: ${STATE_FILE}.done"
  exit 0
fi
if [[ -f "$STATE_FILE" ]]; then STEP=$(<"$STATE_FILE"); else STEP=0; fi
if ! [[ "$STEP" =~ ^[0-9]+$ ]] || (( STEP > TOTAL_STEPS )); then
  echo "Invalid pipeline state '$STEP' in $STATE_FILE" >&2
  exit 2
fi

while (( STEP < TOTAL_STEPS )); do
  case "$STEP" in
    0) LABEL="generate and grade PopQA answers"; COMMAND="collect-popqa-candidates" ;;
    1) LABEL="score base contextual support with hard controls"; COMMAND="collect-popqa-base-support" ;;
    2) LABEL="evaluate ranking and popularity strata"; COMMAND="analyze-popqa-base-support" ;;
  esac
  echo "Starting step $STEP/$((TOTAL_STEPS - 1)): $LABEL"
  "$PYTHON_BIN" -m peft_probe --config "$CONFIG" \
    --base-contextual-transfer-config "$TRANSFER_CONFIG" "$COMMAND" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  unset CHILD_PID
  STEP=$((STEP + 1))
  printf '%s\n' "$STEP" > "${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "$STATE_FILE"
done

touch "${STATE_FILE}.done"
echo "PopQA base-contextual transfer complete: ${STATE_FILE}.done"
