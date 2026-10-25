#!/usr/bin/env bash
#SBATCH --job-name=contextual_support_audit
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
#SBATCH --output=slurm/logs/contextual-support-audit-%j.out
#SBATCH --error=slurm/logs/contextual-support-audit-%j.err

set -euo pipefail

CONFIG=${1:?Pass the existing experiment config}
HARD_SUPPORT_CONFIG=${2:-configs/cross_model_support_hard_controls.yaml}
PAIRWISE_RANK_CONFIG=${3:-configs/cross_model_rank_gate_pairwise.yaml}
ORIGINAL_SUPPORT_CONFIG=${4:-configs/cross_model_support.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
TOTAL_STEPS=6

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs slurm/state
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

STATE_ID=$("$PYTHON_BIN" -c \
  'import hashlib
import json
import sys
from pathlib import Path
from peft_probe.config import load_config
from peft_probe.cross_model_support import (
    cross_model_support_fingerprint,
    load_cross_model_support_config,
)
from peft_probe.cross_model_rank_gate import load_cross_model_rank_gate_config
c = load_config(sys.argv[1])
s = load_cross_model_support_config(sys.argv[2])
r = load_cross_model_rank_gate_config(sys.argv[3])
payload = {"support": cross_model_support_fingerprint(c, s), "rank": r}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12])' \
  "$CONFIG" "$HARD_SUPPORT_CONFIG" "$PAIRWISE_RANK_CONFIG")
STATE_FILE="slurm/state/contextual-support-audit-${STATE_ID}.step"

requeue_current() {
  echo "Time-limit warning received; requeuing at step ${STEP:-unknown}."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ -f "${STATE_FILE}.done" ]]; then
  echo "Contextual-support audit is already complete: ${STATE_FILE}.done"
  exit 0
fi
if [[ -f "$STATE_FILE" ]]; then STEP=$(<"$STATE_FILE"); else STEP=0; fi
if ! [[ "$STEP" =~ ^[0-9]+$ ]] || (( STEP > TOTAL_STEPS )); then
  echo "Invalid pipeline state '$STEP' in $STATE_FILE" >&2
  exit 2
fi

while (( STEP < TOTAL_STEPS )); do
  case "$STEP" in
    0)
      LABEL="paired max-versus-base analysis on existing controls"
      SUPPORT_CONFIG="$ORIGINAL_SUPPORT_CONFIG"
      EXTRA=(--cross-model-rank-gate-config "$PAIRWISE_RANK_CONFIG" analyze-cross-model-rank-gate)
      ;;
    1)
      LABEL="hard-control base replay"
      SUPPORT_CONFIG="$HARD_SUPPORT_CONFIG"
      EXTRA=(collect-cross-model-support --evaluator base)
      ;;
    2)
      LABEL="hard-control LoRA A replay"
      SUPPORT_CONFIG="$HARD_SUPPORT_CONFIG"
      EXTRA=(collect-cross-model-support --evaluator adapter_a)
      ;;
    3)
      LABEL="hard-control LoRA B replay"
      SUPPORT_CONFIG="$HARD_SUPPORT_CONFIG"
      EXTRA=(collect-cross-model-support --evaluator adapter_b)
      ;;
    4)
      LABEL="hard-control synthetic-only selection and frozen evaluation"
      SUPPORT_CONFIG="$HARD_SUPPORT_CONFIG"
      EXTRA=(analyze-cross-model-support)
      ;;
    5)
      LABEL="hard-control paired rank analysis"
      SUPPORT_CONFIG="$HARD_SUPPORT_CONFIG"
      EXTRA=(--cross-model-rank-gate-config "$PAIRWISE_RANK_CONFIG" analyze-cross-model-rank-gate)
      ;;
  esac
  echo "Starting step $STEP/$((TOTAL_STEPS - 1)): $LABEL"
  "$PYTHON_BIN" -m peft_probe --config "$CONFIG" \
    --cross-model-support-config "$SUPPORT_CONFIG" "${EXTRA[@]}" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  unset CHILD_PID
  STEP=$((STEP + 1))
  printf '%s\n' "$STEP" > "${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "$STATE_FILE"
done

touch "${STATE_FILE}.done"
echo "Contextual-support audit complete: ${STATE_FILE}.done"
