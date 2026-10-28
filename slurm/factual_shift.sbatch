#!/usr/bin/env bash
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
case "$COMMAND" in
  run-factual-shift|analyze-factual-shift|repair-factual-shift|audit-factual-shift|within-question-factual-shift|final-checks-factual-shift|run-study-factual-shift|run-popqa-factual-shift) ;;
  *) echo "Unsupported factual-shift command: $COMMAND" >&2; exit 2 ;;
esac
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
RUN_LOG="slurm/logs/factual-run-${SLURM_JOB_ID:-local}.log"
STATUS_LOG="slurm/logs/factual-run-${SLURM_JOB_ID:-local}.status"
record_exit() {
  local status=$?
  printf 'exit_code=%s\ncommand=%s\nconfig=%s\n' "$status" "$COMMAND" "$CONFIG" > "$STATUS_LOG" || true
  echo "[launcher] Exit code: $status; status: $STATUS_LOG; combined log: $RUN_LOG"
}
trap record_exit EXIT

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
echo "Combined process log: $RUN_LOG"
"$PYTHON_BIN" -u scripts/run_factual_job.py --config "$CONFIG" "$COMMAND" > >(tee -a "$RUN_LOG") 2>&1 &
CHILD_PID=$!
if wait "$CHILD_PID"; then
  echo "$COMMAND completed successfully."
else
  STATUS=$?
  echo "$COMMAND failed with exit code $STATUS; see pipeline_progress.json and pipeline_failure.json in factual_shift_v1/ and $RUN_LOG." >&2
  exit "$STATUS"
fi
