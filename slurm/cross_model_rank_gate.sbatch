#!/usr/bin/env bash
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/cross-model-rank-gate-%j.out
#SBATCH --error=slurm/logs/cross-model-rank-gate-%j.err

set -euo pipefail

CONFIG=${1:?Pass the existing experiment config}
SUPPORT_CONFIG=${2:-configs/cross_model_support.yaml}
RANK_CONFIG=${3:-configs/cross_model_rank_gate.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m peft_probe --config "$CONFIG" \
  --cross-model-support-config "$SUPPORT_CONFIG" \
  --cross-model-rank-gate-config "$RANK_CONFIG" \
  analyze-cross-model-rank-gate
