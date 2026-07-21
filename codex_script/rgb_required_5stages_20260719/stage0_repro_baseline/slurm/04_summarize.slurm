#!/bin/bash
#SBATCH --job-name=rgb_s0_sum
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
CFG="$PROJECT_ROOT/codex_script/rgb_required_5stages_20260719/stage0_repro_baseline/config/stage_config.json"
"${CONDA_PREFIX}/bin/python" -u "$PROJECT_ROOT/codex_script/rgb_required_5stages_20260719/common/summarize_results.py" --config "$CFG"

