#!/bin/bash
#SBATCH --job-name=rgb_rp_s0_prep
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
set -euo pipefail
source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_supcon_repair_20260806/scripts/slurm/common_env.sh"
"$PYTHON_BIN" "$PACKAGE_ROOT/run.py" prepare --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
