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
#SBATCH --output=slurm/logs/prompt-aligned-subspace-%j.out
#SBATCH --error=slurm/logs/prompt-aligned-subspace-%j.err

set -euo pipefail

CONFIG=${1:?Pass the derived concise-prompt experiment config}
SUBSPACE_CONFIG=${2:-configs/adapter_invariant_subspace_prompt_aligned.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
CONFIG_NAME=$(basename "$CONFIG")
RUN_NAME=${CONFIG_NAME%.*}
TOTAL_STEPS=6

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
from peft_probe.adapter_invariant_subspace import (
    adapter_invariant_subspace_fingerprint,
    load_adapter_invariant_subspace_config,
)
from peft_probe.config import load_config, output_dir, source_output_dir
c = load_config(sys.argv[1])
s = load_adapter_invariant_subspace_config(sys.argv[2])
if c.get("prompting", {}).get("instruction") != "Answer the question concisely":
    raise SystemExit("Derived config must use the concise prompt instruction")
if c.get("prompting", {}).get("synthetic_template_family") != "concise_diverse_v1":
    raise SystemExit("Derived config must use the concise-diverse template family")
if output_dir(c) == source_output_dir(c):
    raise SystemExit("Derived output_dir must differ from source_output_dir")
print(adapter_invariant_subspace_fingerprint(c, s))
' "$CONFIG" "$SUBSPACE_CONFIG")
STATE_FILE="slurm/state/${RUN_NAME}-${STATE_ID}.prompt-aligned.step"

requeue_current() {
  echo "Time-limit warning received. Requeuing job $SLURM_JOB_ID at step ${STEP:-unknown}."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ -f "${STATE_FILE}.done" ]]; then
  echo "Prompt-aligned subspace pipeline is already complete: ${STATE_FILE}.done"
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
    0)
      LABEL="collect-synthetic-adapter-a"
      CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source adapter_a)
      ;;
    1)
      LABEL="collect-synthetic-adapter-b"
      CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source adapter_b)
      ;;
    2)
      LABEL="collect-synthetic-base"
      CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source base)
      ;;
    3)
      LABEL="collect-benchmark-trivia-qa"
      CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-benchmark --benchmark trivia_qa)
      ;;
    4)
      LABEL="collect-benchmark-truthful-qa"
      CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-benchmark --benchmark truthful_qa)
      ;;
    5)
      LABEL="analyze-adapter-invariant-subspace"
      CMD=(
        "$PYTHON_BIN" -m peft_probe
        --config "$CONFIG"
        --adapter-invariant-subspace-config "$SUBSPACE_CONFIG"
        analyze-adapter-invariant-subspace
      )
      ;;
  esac

  echo "Starting prompt-aligned step $STEP/$((TOTAL_STEPS - 1)): $LABEL"
  "${CMD[@]}" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  unset CHILD_PID
  STEP=$((STEP + 1))
  printf '%s\n' "$STEP" > "${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "$STATE_FILE"
done

touch "${STATE_FILE}.done"
echo "Prompt-aligned subspace pipeline complete: ${STATE_FILE}.done"
