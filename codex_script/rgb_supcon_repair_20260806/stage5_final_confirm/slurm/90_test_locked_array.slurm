#!/bin/bash
#SBATCH --job-name=rgb_rp_s5_test
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-8
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_supcon_repair_20260806/scripts/slurm/common_env.sh"
if [[ "${ALLOW_LOCKED_TEST:-}" != "YES" ]]; then echo "Locked N test; export ALLOW_LOCKED_TEST=YES only after validation is complete"; exit 2; fi
"$PYTHON_BIN" -u "$PACKAGE_ROOT/run.py" test --index "$SLURM_ARRAY_TASK_ID" --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
