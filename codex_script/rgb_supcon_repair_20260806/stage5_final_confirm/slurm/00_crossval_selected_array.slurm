#!/bin/bash
#SBATCH --job-name=rgb_rp_cv
#SBATCH --array=0-2
#SBATCH --time=72:00:00
#SBATCH --mem=72G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
set -euo pipefail
source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_supcon_repair_20260806/scripts/slurm/common_env.sh"
profiles=(fold_M fold_J fold_MR)
profile=${profiles[$SLURM_ARRAY_TASK_ID]}
CV_STAGE=${CV_STAGE:-stage2}
CV_EXPERIMENT=${CV_EXPERIMENT:-s2_m3_dual}
if [[ "$CV_STAGE" == "stage4" ]]; then "$PYTHON_BIN" -u "$PACKAGE_ROOT/run.py" cache_teacher --split-profile "$profile" --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"; fi
"$PYTHON_BIN" -u "$PACKAGE_ROOT/run.py" pretrain --stage "$CV_STAGE" --experiment "$CV_EXPERIMENT" --split-profile "$profile" --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/run.py" diagnose --stage "$CV_STAGE" --experiment "$CV_EXPERIMENT" --split-profile "$profile" --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
