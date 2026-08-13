#!/bin/bash
#SBATCH --job-name=mvpr_cf_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_mvit_proto_rel_env_loso_20260810/confirmation_runner/scripts/slurm/common_env.sh"
: "${CONFIRM_MANIFEST:?CONFIRM_MANIFEST is required}"
"$PYTHON_BIN" -u "$CONFIRM_ROOT/run_confirmation.py" run-task --manifest "$CONFIRM_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --phase finetune --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
