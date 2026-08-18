#!/usr/bin/env bash
#SBATCH --job-name=rgb_locked
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
SLURM_PACKAGE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")}"; source "${SLURM_PACKAGE_DIR}/common_env.sh"
: "${STAGE:?Set the frozen stage name}"
: "${EXPERIMENT_ID:?Set the frozen experiment id selected on dev_N}"
FOLDS=(test_M test_J test_MR)
"${PYTHON_BOOTSTRAP}" "${RUNNER}" pipeline --platform hpc --config "${EXP_CONFIG}" --stage "${STAGE}" --experiment-id "${EXPERIMENT_ID}" --fold "${FOLDS[${SLURM_ARRAY_TASK_ID}]}"
