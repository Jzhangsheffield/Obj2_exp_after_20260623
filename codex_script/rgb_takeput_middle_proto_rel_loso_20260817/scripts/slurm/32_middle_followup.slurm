#!/usr/bin/env bash
#SBATCH --job-name=mid_follow
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
SLURM_PACKAGE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")}"; source "${SLURM_PACKAGE_DIR}/common_env.sh"
: "${STAGE:?Set STAGE to middle_rel_start, middle_combined, or middle_p2_sentinel}"
: "${INDEX:?Set INDEX to a valid zero-based row}"
"${PYTHON_BOOTSTRAP}" "${RUNNER}" pipeline --platform hpc --config "${EXP_CONFIG}" --stage "${STAGE}" --index "${INDEX}" --fold dev_N
