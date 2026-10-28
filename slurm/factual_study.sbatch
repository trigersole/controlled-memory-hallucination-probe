#!/usr/bin/env bash
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --array=0-2%1
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/factual-study-%A_%a.out
#SBATCH --error=slurm/logs/factual-study-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
MANIFEST=${1:-configs/factual_confirmation/study.json}
DEFAULT_PYTHON="$PWD/.venv/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  DEFAULT_PYTHON="${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}/.venv/bin/python"
fi
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$DEFAULT_PYTHON}}
echo "Study wrapper starting: job ${SLURM_JOB_ID:-unknown}; array task ${SLURM_ARRAY_TASK_ID:-missing}; manifest $MANIFEST"
trap 'status=$?; echo "Study wrapper failed at line $LINENO with exit code $status" >&2' ERR
CONFIG=$("$PYTHON_BIN" - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
index = int(sys.argv[2])
if manifest["schema_version"] != 1 or not 0 <= index < len(manifest["runs"]):
    raise ValueError("Invalid study manifest or array index")
run = manifest["runs"][index]
path = Path(run["config"])
if hashlib.sha256(path.read_bytes()).hexdigest() != run["config_sha256"]:
    raise ValueError("Planned config was modified; refusing to run")
print(path)
PY
)
echo "Study $MANIFEST; array task $SLURM_ARRAY_TASK_ID; config $CONFIG"
# Replace this wrapper so wall-time/requeue signals reach the existing resumable launcher.
exec bash slurm/factual_shift.sbatch "$CONFIG" run-study-factual-shift
