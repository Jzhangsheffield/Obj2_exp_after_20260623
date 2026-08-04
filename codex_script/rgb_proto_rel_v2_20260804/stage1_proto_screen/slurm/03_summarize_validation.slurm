#!/bin/bash
#SBATCH --job-name=rgb_v2_s1_val
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
module load Anaconda3/2022.05
source activate pytorch
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
export PROJECT_ROOT PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
CFG="$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage1_proto_screen/config/stage_config.json"
PY="$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/common"
"${CONDA_PREFIX}/bin/python" -u "$PY/summarize_v2.py" --config "$CFG"
"${CONDA_PREFIX}/bin/python" -u "$PY/summarize_validation.py" --config "$CFG"
