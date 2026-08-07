#!/bin/bash
#SBATCH --job-name=rgb_rp_s4_cl
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-3
#SBATCH --time=60:00:00
#SBATCH --mem=72G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_supcon_repair_20260806/scripts/slurm/common_env.sh"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/run.py" pretrain --stage stage4 --index "$SLURM_ARRAY_TASK_ID" --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
