#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail

CONFIG=$1
STAGE=$2
PYTHON_BIN=${PYTHON_BIN:-python}

requeue_job() {
  echo "Time limit signal received; checkpointed work will resume after requeue."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_job USR1

cd "${SLURM_SUBMIT_DIR}"
export TOKENIZERS_PARALLELISM=false

case "$STAGE" in
  train-adapter)
    ADAPTERS=(a b)
    "$PYTHON_BIN" -m peft_probe --config "$CONFIG" train-adapter --adapter "${ADAPTERS[$SLURM_ARRAY_TASK_ID]}" &
    ;;
  collect-synthetic)
    SOURCES=(adapter_a adapter_b base)
    "$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-synthetic --source "${SOURCES[$SLURM_ARRAY_TASK_ID]}" &
    ;;
  collect-benchmark)
    BENCHMARKS=(trivia_qa truthful_qa)
    "$PYTHON_BIN" -m peft_probe --config "$CONFIG" collect-benchmark --benchmark "${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}" &
    ;;
  *)
    echo "Unknown GPU stage: $STAGE" >&2
    exit 2
    ;;
esac
wait $!

