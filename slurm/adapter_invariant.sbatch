#!/usr/bin/env bash
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:50:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/adapter-invariant-%j.out
#SBATCH --error=slurm/logs/adapter-invariant-%j.err

set -euo pipefail

CONFIG=${1:-configs/pilot_llama2_last_tokens.yaml}
INVARIANT_CONFIG=${2:-configs/adapter_invariant.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs slurm/state
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

STATE_ID=$("$PYTHON_BIN" -c '
import sys
from peft_probe.adapter_invariant import (
    adapter_invariant_fingerprint,
    load_adapter_invariant_config,
)
from peft_probe.config import load_config
c = load_config(sys.argv[1])
s = load_adapter_invariant_config(sys.argv[2])
print(adapter_invariant_fingerprint(c, s))
' "$CONFIG" "$INVARIANT_CONFIG")
STATE_FILE="slurm/state/adapter-invariant-${STATE_ID}.step"

requeue_current() {
  echo "Time-limit warning received; requeuing adapter-invariant job $SLURM_JOB_ID."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ -f "${STATE_FILE}.done" ]]; then
  echo "Adapter-invariant analysis is already complete: ${STATE_FILE}.done"
  exit 0
fi

"$PYTHON_BIN" -m peft_probe \
  --config "$CONFIG" \
  --adapter-invariant-config "$INVARIANT_CONFIG" \
  analyze-adapter-invariant &
CHILD_PID=$!
wait "$CHILD_PID"
unset CHILD_PID

printf '1\n' > "${STATE_FILE}.tmp"
mv "${STATE_FILE}.tmp" "$STATE_FILE"
touch "${STATE_FILE}.done"
echo "Adapter-invariant analysis complete: ${STATE_FILE}.done"
