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
#SBATCH --output=slurm/logs/memprobe-rerun-%j.out
#SBATCH --error=slurm/logs/memprobe-rerun-%j.err

set -euo pipefail

CONFIG=${1:-configs/pilot_llama2.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

requeue_current() {
  echo "Time-limit warning received; requeuing the same probe-rerun job."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

OUTPUT_DIR=$("$PYTHON_BIN" -c \
  'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment"]["output_dir"])' \
  "$CONFIG")
RESET_MARKER="$OUTPUT_DIR/results/.shared_linear_probe_v2_reset"

if [[ ! -f "$RESET_MARKER" ]]; then
  echo "Invalidating only the old probe-derived artifacts; adapters and features are preserved."
  for benchmark in trivia_qa truthful_qa; do
    if [[ -f "$OUTPUT_DIR/results/$benchmark/metrics.json" ]]; then
      cp "$OUTPUT_DIR/results/$benchmark/metrics.json" \
        "$OUTPUT_DIR/results/$benchmark/metrics.before_shared_linear_fix.json"
    fi
    rm -f "$OUTPUT_DIR/results/$benchmark/_SUCCESS.json"
  done
  find "$OUTPUT_DIR/probes" -type f \
    \( -name '_SUCCESS.json' -o -name 'last_checkpoint.pt' \) -delete
  touch "$RESET_MARKER"
fi

echo "Retraining probes with: $PYTHON_BIN"
"$PYTHON_BIN" -m peft_probe --config "$CONFIG" train-probes
"$PYTHON_BIN" -m peft_probe --config "$CONFIG" evaluate --benchmark trivia_qa
"$PYTHON_BIN" -m peft_probe --config "$CONFIG" evaluate --benchmark truthful_qa
"$PYTHON_BIN" -m peft_probe --config "$CONFIG" summarize
