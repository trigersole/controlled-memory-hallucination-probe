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
COMMAND=${2:-run-factual-shift}
if [[ "$COMMAND" != "run-factual-shift" && "$COMMAND" != "analyze-factual-shift" && "$COMMAND" != "repair-factual-shift" && "$COMMAND" != "audit-factual-shift" && "$COMMAND" != "within-question-factual-shift" && "$COMMAND" != "final-checks-factual-shift" && "$COMMAND" != "run-study-factual-shift" ]]; then
  echo "Unsupported factual-shift command: $COMMAND" >&2
  exit 2
fi
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
DEFAULT_PYTHON="$PWD/.venv/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="$HALOSCOPE_DIR/.venv/bin/python"
fi
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$DEFAULT_PYTHON}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
export PYTHONPATH="$PWD/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p slurm/logs

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python is not executable: $PYTHON_BIN. Set PYTHON_BIN to your experiment environment." >&2
  exit 2
fi
echo "Python: $PYTHON_BIN"
echo "Dependency overlay: $PEFT_OVERLAY"
"$PYTHON_BIN" - <<'PY'
import importlib
import sys

for name in ("torch", "peft", "transformers", "datasets", "sklearn", "yaml"):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"Dependency check failed for {name} in {sys.executable}: {exc}", file=sys.stderr)
        print("Use the same PYTHON_BIN and PEFT_OVERLAY as your working experiment jobs.", file=sys.stderr)
        sys.exit(2)
    print(f"{name}: {getattr(module, '__version__', 'unknown')}")
PY

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

echo "Starting $COMMAND with config $CONFIG"
"$PYTHON_BIN" -m peft_probe --config "$CONFIG" "$COMMAND" &
CHILD_PID=$!
if wait "$CHILD_PID"; then
  echo "$COMMAND completed successfully."
else
  STATUS=$?
  echo "$COMMAND failed with exit code $STATUS; see analysis_progress.json and analysis_failure.json in results/." >&2
  exit "$STATUS"
fi
