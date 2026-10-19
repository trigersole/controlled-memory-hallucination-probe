#!/usr/bin/env bash
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
#SBATCH --output=slurm/logs/training-shift-%j.out
#SBATCH --error=slurm/logs/training-shift-%j.err

set -euo pipefail

CONFIG=${1:-configs/pilot.yaml}
SHIFT_CONFIG=${2:-configs/training_shift.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
TOTAL_STEPS=4

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
from peft_probe.config import load_config
from peft_probe.training_shift import load_training_shift_config, training_shift_fingerprint
c = load_config(sys.argv[1])
s = load_training_shift_config(sys.argv[2])
print(training_shift_fingerprint(c, s))' \
  "$CONFIG" "$SHIFT_CONFIG")
STATE_FILE="slurm/state/training-shift-${STATE_ID}.step"

requeue_current() {
  echo "Time-limit warning received; requeuing training-shift job $SLURM_JOB_ID at step ${STEP:-unknown}."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ -f "${STATE_FILE}.done" ]]; then
  echo "Training-shift analysis is already complete: ${STATE_FILE}.done"
  exit 0
fi
if [[ -f "$STATE_FILE" ]]; then
  STEP=$(<"$STATE_FILE")
else
  STEP=0
fi
if ! [[ "$STEP" =~ ^[0-9]+$ ]] || (( STEP > TOTAL_STEPS )); then
  echo "Invalid training-shift state '$STEP' in $STATE_FILE" >&2
  exit 2
fi

while (( STEP < TOTAL_STEPS )); do
  case "$STEP" in
    0) LABEL="base activations"; EXTRA=(collect-training-shift --source base) ;;
    1) LABEL="LoRA A activations"; EXTRA=(collect-training-shift --source adapter_a) ;;
    2) LABEL="LoRA B activations"; EXTRA=(collect-training-shift --source adapter_b) ;;
    3) LABEL="principal/update subspace analysis"; EXTRA=(analyze-training-shift) ;;
  esac
  echo "Starting training-shift step $STEP/$((TOTAL_STEPS - 1)): $LABEL"
  "$PYTHON_BIN" -m peft_probe --config "$CONFIG" \
    --training-shift-config "$SHIFT_CONFIG" "${EXTRA[@]}" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  unset CHILD_PID
  STEP=$((STEP + 1))
  printf '%s\n' "$STEP" > "${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "$STATE_FILE"
  echo "Completed $LABEL; saved next step $STEP."
done

touch "${STATE_FILE}.done"
echo "Training-shift analysis complete: ${STATE_FILE}.done"
