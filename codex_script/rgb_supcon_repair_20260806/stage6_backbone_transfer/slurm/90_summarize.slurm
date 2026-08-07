#!/bin/bash
#SBATCH --job-name=rgb_rp_s6_sum
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_supcon_repair_20260806/scripts/slurm/common_env.sh"
"$PYTHON_BIN" "$PACKAGE_ROOT/run.py" summarize --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
