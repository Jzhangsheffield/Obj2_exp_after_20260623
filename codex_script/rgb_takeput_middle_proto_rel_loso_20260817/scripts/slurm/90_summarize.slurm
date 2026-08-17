#!/usr/bin/env bash
#SBATCH --job-name=rgb_summary
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
SLURM_PACKAGE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")}"; source "${SLURM_PACKAGE_DIR}/common_env.sh"
"${PYTHON_BOOTSTRAP}" "${RUNNER}" summarize --platform hpc --config "${EXP_CONFIG}"
