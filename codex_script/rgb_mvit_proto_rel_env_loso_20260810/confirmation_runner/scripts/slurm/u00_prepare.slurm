#!/bin/bash
#SBATCH --job-name=mvpr_u_prep
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
DATASET_ROOT=${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset}
CONFIRM_ROOT="$PROJECT_ROOT/codex_script/rgb_mvit_proto_rel_env_loso_20260810/confirmation_runner"
ANALYSIS_PYTHON_BIN=${ANALYSIS_PYTHON_BIN:-python3}
"$ANALYSIS_PYTHON_BIN" -u "$CONFIRM_ROOT/run_unified.py" prepare --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT" --python-bin "$ANALYSIS_PYTHON_BIN"
