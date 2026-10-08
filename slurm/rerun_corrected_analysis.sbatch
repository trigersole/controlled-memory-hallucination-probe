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
#SBATCH --output=slurm/logs/memprobe-corrected-%j.out
#SBATCH --error=slurm/logs/memprobe-corrected-%j.err

set -euo pipefail

CONFIG=${1:-configs/pilot_llama2.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}
TOTAL_STEPS=10

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs slurm/state
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "HaloScope Python is not executable: $PYTHON_BIN" >&2
  exit 2
fi

OUTPUT_DIR=$("$PYTHON_BIN" -c \
  'import pathlib, sys, yaml; print(pathlib.Path(yaml.safe_load(open(sys.argv[1]))["experiment"]["output_dir"]).resolve())' \
  "$CONFIG")
OUTPUT_ROOT=$("$PYTHON_BIN" -c 'import pathlib; print(pathlib.Path("outputs").resolve())')
case "$OUTPUT_DIR/" in
  "$OUTPUT_ROOT/"*) ;;
  *)
    echo "Refusing derived-artifact reset outside $OUTPUT_ROOT: $OUTPUT_DIR" >&2
    exit 2
    ;;
esac
CONFIG_NAME=$(basename "$CONFIG")
RUN_NAME=${CONFIG_NAME%.*}
STATE_ID=$("$PYTHON_BIN" -c \
  'import sys; from peft_probe.config import config_fingerprint, load_config; from peft_probe.versioning import PIPELINE_SCHEMA_VERSION; print(f"{config_fingerprint(load_config(sys.argv[1]))}-v{PIPELINE_SCHEMA_VERSION}")' \
  "$CONFIG")
STATE_FILE="slurm/state/${RUN_NAME}-corrected-${STATE_ID}.step"
RESET_MARKER="slurm/state/${RUN_NAME}-corrected-${STATE_ID}.reset"

requeue_current() {
  echo "Time-limit warning received; requeuing job $SLURM_JOB_ID at step ${STEP:-unknown}."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_current USR1

if [[ ! -f "$RESET_MARKER" ]]; then
  echo "Archiving small result JSON files and removing only regenerable derived artifacts."
  ARCHIVE_DIR="$OUTPUT_DIR/results/archive_before_schema_v3"
  mkdir -p "$ARCHIVE_DIR"
  for benchmark in trivia_qa truthful_qa; do
    if [[ -f "$OUTPUT_DIR/results/$benchmark/metrics.json" ]]; then
      cp "$OUTPUT_DIR/results/$benchmark/metrics.json" \
        "$ARCHIVE_DIR/${benchmark}_metrics.json"
    fi
  done
  if [[ -f "$OUTPUT_DIR/results/intervention_check.json" ]]; then
    cp "$OUTPUT_DIR/results/intervention_check.json" "$ARCHIVE_DIR/intervention_check.json"
  fi
  rm -rf -- \
    "$OUTPUT_DIR/features/synthetic" \
    "$OUTPUT_DIR/features/benchmarks" \
    "$OUTPUT_DIR/probes" \
    "$OUTPUT_DIR/results/trivia_qa" \
    "$OUTPUT_DIR/results/truthful_qa"
  rm -f -- "$OUTPUT_DIR/results/intervention_check.json" \
    "$OUTPUT_DIR/results/analysis_summary.md"
  touch "$RESET_MARKER"
fi

if [[ -f "$STATE_FILE" ]]; then
  STEP=$(<"$STATE_FILE")
else
  STEP=0
fi

if ! [[ "$STEP" =~ ^[0-9]+$ ]] || (( STEP > TOTAL_STEPS )); then
  echo "Invalid corrected-analysis state '$STEP' in $STATE_FILE" >&2
  exit 2
fi

while (( STEP < TOTAL_STEPS )); do
  case "$STEP" in
    0) LABEL="collect-synthetic-adapter-a"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source adapter_a) ;;
    1) LABEL="collect-synthetic-adapter-b"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source adapter_b) ;;
    2) LABEL="collect-synthetic-base"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source base) ;;
    3) LABEL="collect-benchmark-trivia-qa"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-benchmark --benchmark trivia_qa) ;;
    4) LABEL="collect-benchmark-truthful-qa"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-benchmark --benchmark truthful_qa) ;;
    5) LABEL="intervention-report"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" intervention-report) ;;
    6) LABEL="train-probes"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" train-probes) ;;
    7) LABEL="evaluate-trivia-qa"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" evaluate --benchmark trivia_qa) ;;
    8) LABEL="evaluate-truthful-qa"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" evaluate --benchmark truthful_qa) ;;
    9) LABEL="summarize"; CMD=("$PYTHON_BIN" -m peft_probe --config "$CONFIG" summarize) ;;
  esac

  echo "Starting corrected-analysis step $STEP/$((TOTAL_STEPS - 1)): $LABEL"
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
echo "Corrected analysis complete: ${STATE_FILE}.done"
