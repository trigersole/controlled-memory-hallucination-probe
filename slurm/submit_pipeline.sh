#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/pilot.yaml}
mkdir -p slurm/logs slurm/state

CONFIG_NAME=$(basename "$CONFIG")
RUN_NAME=${CONFIG_NAME%.*}
STATE_FILE=${PIPELINE_STATE_FILE:-slurm/state/${RUN_NAME}.step}

if [[ "${RESET_PIPELINE:-0}" == "1" ]]; then
  rm -f "$STATE_FILE" "${STATE_FILE}.tmp" "${STATE_FILE}.done"
fi

ACCOUNT_ARGS=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  ACCOUNT_ARGS=(--account "$SLURM_ACCOUNT")
fi

JOB_ID=$(sbatch --parsable --export=ALL,PIPELINE_STATE_FILE="$STATE_FILE" \
  --output=slurm/logs/%x-%j.out --error=slurm/logs/%x-%j.err \
  "${ACCOUNT_ARGS[@]}" --job-name=memprobe-serial \
  slurm/serial_pipeline.sbatch "$CONFIG")

cat <<EOF
Submitted one self-requeuing controlled-memory pipeline job: $JOB_ID
State file: $STATE_FILE
Monitor with: squeue -j $JOB_ID
Log: slurm/logs/memprobe-serial-$JOB_ID.out
EOF
