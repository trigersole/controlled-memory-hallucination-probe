#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/pilot.yaml}
PARTITION=${SLURM_PARTITION:-gpu}
CPU_PARTITION=${SLURM_CPU_PARTITION:-$PARTITION}
mkdir -p slurm/logs

ACCOUNT_ARGS=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  ACCOUNT_ARGS=(--account "$SLURM_ACCOUNT")
fi

COMMON=(--parsable --export=ALL --output=slurm/logs/%x-%A_%a.out --error=slurm/logs/%x-%A_%a.err)

DATA_JOB=$(sbatch "${COMMON[@]}" "${ACCOUNT_ARGS[@]}" --partition "$CPU_PARTITION" \
  --job-name=memprobe-data slurm/cpu.sbatch "$CONFIG" generate-data)

ADAPTER_JOB=$(sbatch "${COMMON[@]}" "${ACCOUNT_ARGS[@]}" --partition "$PARTITION" \
  --array=0-1 --dependency="afterok:$DATA_JOB" \
  --job-name=memprobe-lora slurm/gpu.sbatch "$CONFIG" train-adapter)

SYNTHETIC_JOB=$(sbatch "${COMMON[@]}" "${ACCOUNT_ARGS[@]}" --partition "$PARTITION" \
  --array=0-2 --dependency="afterok:$ADAPTER_JOB" \
  --job-name=memprobe-features slurm/gpu.sbatch "$CONFIG" collect-synthetic)

BENCHMARK_JOB=$(sbatch "${COMMON[@]}" "${ACCOUNT_ARGS[@]}" --partition "$PARTITION" \
  --array=0-1 --dependency="afterok:$DATA_JOB" \
  --job-name=memprobe-bench slurm/gpu.sbatch "$CONFIG" collect-benchmark)

INTERVENTION_JOB=$(sbatch "${COMMON[@]}" "${ACCOUNT_ARGS[@]}" --partition "$CPU_PARTITION" \
  --dependency="afterok:$SYNTHETIC_JOB" \
  --job-name=memprobe-check slurm/cpu.sbatch "$CONFIG" intervention-report)

PROBE_JOB=$(sbatch "${COMMON[@]}" "${ACCOUNT_ARGS[@]}" --partition "$CPU_PARTITION" \
  --dependency="afterok:$INTERVENTION_JOB" \
  --job-name=memprobe-probes slurm/cpu.sbatch "$CONFIG" train-probes)

EVALUATE_JOB=$(sbatch "${COMMON[@]}" "${ACCOUNT_ARGS[@]}" --partition "$CPU_PARTITION" \
  --array=0-1 --dependency="afterok:$PROBE_JOB:$BENCHMARK_JOB" \
  --job-name=memprobe-eval slurm/cpu.sbatch "$CONFIG" evaluate)

cat <<EOF
Submitted controlled-memory probe pipeline:
  data:         $DATA_JOB
  adapters:     $ADAPTER_JOB
  synthetic:    $SYNTHETIC_JOB
  benchmarks:   $BENCHMARK_JOB
  intervention: $INTERVENTION_JOB
  probes:       $PROBE_JOB
  evaluation:   $EVALUATE_JOB

Monitor with: squeue -j $DATA_JOB,$ADAPTER_JOB,$SYNTHETIC_JOB,$BENCHMARK_JOB,$INTERVENTION_JOB,$PROBE_JOB,$EVALUATE_JOB
EOF
