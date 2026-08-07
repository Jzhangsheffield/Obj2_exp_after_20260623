#!/bin/bash
#SBATCH --job-name=rgb_rp_s6c_cfcl
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --time=36:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_supcon_repair_20260806/scripts/slurm/common_env.sh"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/run.py" stage6c_confirm_pretrain --index "$SLURM_ARRAY_TASK_ID" --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
