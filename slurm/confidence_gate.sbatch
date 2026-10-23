#!/usr/bin/env bash
#SBATCH --partition=MGPU-TC2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/confidence-gate-%j.out
#SBATCH --error=slurm/logs/confidence-gate-%j.err

set -euo pipefail

CONFIG=${1:?Pass the experiment config whose saved generations should be evaluated}
GATE_CONFIG=${2:-configs/confidence_gate.yaml}
HALOSCOPE_DIR=${HALOSCOPE_DIR:-$HOME/LLM_Haloscope}
PYTHON_BIN=${HALOSCOPE_PYTHON:-${PYTHON_BIN:-$HALOSCOPE_DIR/.venv/bin/python}}
PEFT_OVERLAY=${PEFT_OVERLAY:-${SCRATCH:-$HOME}/peft-probe-overlay}

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:$PEFT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m peft_probe \
  --config "$CONFIG" \
  --confidence-gate-config "$GATE_CONFIG" \
  analyze-confidence-gate
