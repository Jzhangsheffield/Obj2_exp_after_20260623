#!/bin/bash
#SBATCH --job-name=mvpr_sum
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_mvit_proto_rel_env_loso_20260810/scripts/slurm/common_env.sh"
"$PYTHON_BIN" "$PACKAGE_ROOT/run.py" summarize --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
