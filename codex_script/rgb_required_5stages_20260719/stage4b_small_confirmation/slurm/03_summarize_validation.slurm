#!/bin/bash
#SBATCH --job-name=rgb_s4b_valsum
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module load Anaconda3/2022.05
source activate pytorch
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
export PROJECT_ROOT PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
"${CONDA_PREFIX}/bin/python" -u \
  "$PROJECT_ROOT/codex_script/rgb_required_5stages_20260719/stage4b_small_confirmation/scripts/summarize_confirmation.py" \
  --project-root "$PROJECT_ROOT"
