#!/bin/bash
#SBATCH --job-name=rgb_s7_src
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
set -euo pipefail
module load Anaconda3/2022.05
source activate pytorch
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
export PROJECT_ROOT PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
CFG="$PROJECT_ROOT/codex_script/rgb_required_5stages_20260719/stage7_finetune/config/stage_config.json"
"${CONDA_PREFIX}/bin/python" -u "$PROJECT_ROOT/codex_script/rgb_required_5stages_20260719/common/validate_sources.py" --config "$CFG"

