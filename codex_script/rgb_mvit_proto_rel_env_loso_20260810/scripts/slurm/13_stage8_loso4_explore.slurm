#!/bin/bash
#SBATCH --job-name=mvpr_s8x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --array=0-31%4
#SBATCH --time=48:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail

source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_mvit_proto_rel_env_loso_20260810/scripts/slurm/common_env.sh"

AUDIT_FILE="$PROJECT_ROOT/results/rgb_mvit_pr_env_loso_20260810/runtime/splits/protocol_audit.json"
if [[ ! -s "$AUDIT_FILE" ]]; then
    echo "ERROR: missing protocol audit: $AUDIT_FILE"
    echo "Run scripts/slurm/01_prepare.slurm before submitting Stage 8."
    exit 1
fi

EXP_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
FOLD_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
case "$FOLD_INDEX" in
    0) FOLD=M ;;
    1) FOLD=J ;;
    2) FOLD=MR ;;
    3) FOLD=N ;;
esac

echo "Stage 8 exploratory LOSO: experiment index=$EXP_INDEX, held-out fold=$FOLD"
"$PYTHON_BIN" -u "$PACKAGE_ROOT/run.py" pipeline \
    --stage stage8 --index "$EXP_INDEX" --fold "$FOLD" --outer-test \
    --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"

