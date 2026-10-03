#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

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

case "$STAGE" in
  generate-data)
    "$PYTHON_BIN" -m peft_probe --config "$CONFIG" generate-data &
    ;;
  intervention-report)
    "$PYTHON_BIN" -m peft_probe --config "$CONFIG" intervention-report &
    ;;
  train-probes)
    "$PYTHON_BIN" -m peft_probe --config "$CONFIG" train-probes &
    ;;
  evaluate)
    BENCHMARKS=(trivia_qa truthful_qa)
    "$PYTHON_BIN" -m peft_probe --config "$CONFIG" evaluate --benchmark "${BENCHMARKS[$SLURM_ARRAY_TASK_ID]}" &
    ;;
  *)
    echo "Unknown CPU stage: $STAGE" >&2
    exit 2
    ;;
esac
wait $!

