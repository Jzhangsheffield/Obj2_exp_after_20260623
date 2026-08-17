#!/usr/bin/env bash
#SBATCH --job-name=tp_direct_N
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
SLURM_PACKAGE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")}"; source "${SLURM_PACKAGE_DIR}/common_env.sh"
"${PYTHON_BOOTSTRAP}" "${RUNNER}" pipeline --platform hpc --config "${EXP_CONFIG}" --stage takeput_direct --index "${SLURM_ARRAY_TASK_ID}" --fold dev_N
