#!/bin/bash
#SBATCH --job-name=mvpr_u_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_mvit_proto_rel_env_loso_20260810/confirmation_runner/scripts/slurm/common_env.sh"
: "${UNIFIED_MANIFEST:?UNIFIED_MANIFEST is required}"
"$PYTHON_BIN" -u "$CONFIRM_ROOT/run_unified.py" run-task --manifest "$UNIFIED_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --phase evaluate --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
