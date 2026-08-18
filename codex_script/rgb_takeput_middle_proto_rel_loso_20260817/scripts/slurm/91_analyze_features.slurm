#!/usr/bin/env bash
#SBATCH --job-name=rgb_features
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
SLURM_PACKAGE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")}"; source "${SLURM_PACKAGE_DIR}/common_env.sh"
: "${STAGE:?Set STAGE}"
: "${EXPERIMENT_ID:?Set EXPERIMENT_ID}"
: "${FOLD:?Set FOLD, for example dev_N}"
: "${POLICY:?Set POLICY to full or head_only}"
"${PYTHON_BOOTSTRAP}" "${RUNNER}" features --platform hpc --config "${EXP_CONFIG}" --stage "${STAGE}" --experiment-id "${EXPERIMENT_ID}" --fold "${FOLD}" --checkpoint-kind classifier --policy "${POLICY}"
