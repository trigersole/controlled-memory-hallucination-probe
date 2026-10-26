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
#SBATCH --output=slurm/logs/factual-shift-%j.out
#SBATCH --error=slurm/logs/factual-shift-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
CONFIG=${1:-configs/factual_shift.yaml}
PYTHON_BIN=${PYTHON_BIN:-$PWD/.venv/bin/python}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
mkdir -p slurm/logs

requeue_current() {
  echo "Wall-time warning; stopping the current stage before requeue. Saved shards/checkpoints will resume."
  if [[ -n "${CHILD_PID:-}" ]]; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

"$PYTHON_BIN" -m peft_probe --config "$CONFIG" run-factual-shift &
CHILD_PID=$!
wait "$CHILD_PID"
