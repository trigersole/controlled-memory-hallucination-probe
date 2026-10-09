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
#SBATCH --output=slurm/logs/memcompare-%j.out
#SBATCH --error=slurm/logs/memcompare-%j.err

set -euo pipefail

CONFIG=${1:-configs/pilot_llama2.yaml}
GEOMETRY_CONFIG=${2:-configs/geometry.yaml}
COMPARISON_CONFIG=${3:-configs/haloscope_comparison.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
TOTAL_STEPS=5

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs slurm/state
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

STATE_ID=$("$PYTHON_BIN" -c '
import sys
from peft_probe.comparison import comparison_fingerprint, load_comparison_config
from peft_probe.config import load_config
from peft_probe.geometry import load_geometry_config
c = load_config(sys.argv[1])
g = load_geometry_config(sys.argv[2])
x = load_comparison_config(sys.argv[3])
print(comparison_fingerprint(c, g, x))
' "$CONFIG" "$GEOMETRY_CONFIG" "$COMPARISON_CONFIG")
STATE_FILE="slurm/state/haloscope-comparison-${STATE_ID}.step"

requeue_current() {
  echo "Time-limit warning received; requeuing comparison job $SLURM_JOB_ID at step ${STEP:-unknown}."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ -f "${STATE_FILE}.done" ]]; then
  echo "Matched comparison is already complete: ${STATE_FILE}.done"
  exit 0
fi
if [[ -f "$STATE_FILE" ]]; then STEP=$(<"$STATE_FILE"); else STEP=0; fi
if ! [[ "$STEP" =~ ^[0-9]+$ ]] || (( STEP > TOTAL_STEPS )); then
  echo "Invalid comparison state '$STEP' in $STATE_FILE" >&2
  exit 2
fi

while (( STEP < TOTAL_STEPS )); do
  case "$STEP" in
    0) LABEL="synthetic-adapter-a"; EXTRA=(collect-comparison-synthetic --source adapter_a) ;;
    1) LABEL="synthetic-adapter-b"; EXTRA=(collect-comparison-synthetic --source adapter_b) ;;
    2) LABEL="benchmark-trivia-qa"; EXTRA=(collect-comparison-benchmark --benchmark trivia_qa) ;;
    3) LABEL="benchmark-truthful-qa"; EXTRA=(collect-comparison-benchmark --benchmark truthful_qa) ;;
    4) LABEL="analyze-matched-comparison"; EXTRA=(analyze-comparison) ;;
  esac
  echo "Starting comparison step $STEP/$((TOTAL_STEPS - 1)): $LABEL"
  "$PYTHON_BIN" -m peft_probe --config "$CONFIG" \
    --geometry-config "$GEOMETRY_CONFIG" \
    --comparison-config "$COMPARISON_CONFIG" "${EXTRA[@]}" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  unset CHILD_PID
  STEP=$((STEP + 1))
  printf '%s\n' "$STEP" > "${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "$STATE_FILE"
  echo "Completed $LABEL; saved next step $STEP."
done

touch "${STATE_FILE}.done"
echo "Matched comparison complete: ${STATE_FILE}.done"
