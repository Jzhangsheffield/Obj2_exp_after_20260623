#!/usr/bin/env bash
#SBATCH --job-name=rgb_manifest
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
SLURM_PACKAGE_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")}"; source "${SLURM_PACKAGE_DIR}/common_env.sh"
"${PYTHON_BOOTSTRAP}" "${RUNNER}" validate --platform hpc --config "${EXP_CONFIG}"
"${PYTHON_BOOTSTRAP}" "${RUNNER}" prepare --platform hpc --config "${EXP_CONFIG}"
