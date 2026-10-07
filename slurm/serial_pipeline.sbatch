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

set -euo pipefail

CONFIG=${1:-configs/pilot.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
CONFIG_NAME=$(basename "$CONFIG")
RUN_NAME=${CONFIG_NAME%.*}
STATE_FILE=${PIPELINE_STATE_FILE:-slurm/state/${RUN_NAME}.step}
TOTAL_STEPS=12

cd "${SLURM_SUBMIT_DIR}"
mkdir -p "$(dirname "$STATE_FILE")"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi
"$PYTHON_BIN" -c "import peft_probe; print('Job Python:', __import__('sys').executable)"

requeue_current() {
  echo "Time-limit warning received. Requeuing job $SLURM_JOB_ID without advancing step ${STEP:-unknown}."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ -f "${STATE_FILE}.done" ]]; then
  echo "Pipeline is already complete: ${STATE_FILE}.done"
  exit 0
fi

if [[ -f "$STATE_FILE" ]]; then
  STEP=$(<"$STATE_FILE")
else
  STEP=0
fi

if ! [[ "$STEP" =~ ^[0-9]+$ ]] || (( STEP > TOTAL_STEPS )); then
  echo "Invalid pipeline state '$STEP' in $STATE_FILE" >&2
  exit 2
fi

while (( STEP < TOTAL_STEPS )); do
  case "$STEP" in
    0)  LABEL="generate-data"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" generate-data) ;;
    1)  LABEL="train-adapter-a"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" train-adapter --adapter a) ;;
    2)  LABEL="train-adapter-b"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" train-adapter --adapter b) ;;
    3)  LABEL="collect-synthetic-adapter-a"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source adapter_a) ;;
    4)  LABEL="collect-synthetic-adapter-b"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source adapter_b) ;;
    5)  LABEL="collect-synthetic-base"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source base) ;;
    6)  LABEL="collect-benchmark-trivia-qa"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-benchmark --benchmark trivia_qa) ;;
    7)  LABEL="collect-benchmark-truthful-qa"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-benchmark --benchmark truthful_qa) ;;
    8)  LABEL="intervention-report"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" intervention-report) ;;
    9)  LABEL="train-probes"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" train-probes) ;;
    10) LABEL="evaluate-trivia-qa"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" evaluate --benchmark trivia_qa) ;;
    11) LABEL="evaluate-truthful-qa"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" evaluate --benchmark truthful_qa) ;;
  esac

  echo "Starting pipeline step $STEP/$((TOTAL_STEPS - 1)): $LABEL"
  "${CMD[@]}" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  unset CHILD_PID

  STEP=$((STEP + 1))
  printf '%s\n' "$STEP" > "${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "$STATE_FILE"
  echo "Completed $LABEL; saved next step $STEP."
done

touch "${STATE_FILE}.done"
echo "Pipeline complete. State marker: ${STATE_FILE}.done"
