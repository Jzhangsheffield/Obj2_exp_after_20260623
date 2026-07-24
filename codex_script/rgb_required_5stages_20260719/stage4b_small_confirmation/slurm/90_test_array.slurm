#!/bin/bash
#SBATCH --job-name=rgb_s4b_test
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-22
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
set -euo pipefail
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate pytorch
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
DATASET_ROOT=${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset}
export PROJECT_ROOT DATASET_ROOT PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
CFG="$PROJECT_ROOT/codex_script/rgb_required_5stages_20260719/stage4b_small_confirmation/config/stage_config.json"
srun "${CONDA_PREFIX}/bin/python" -u "$PROJECT_ROOT/codex_script/rgb_required_5stages_20260719/common/run_test.py" --config "$CFG" --index "$SLURM_ARRAY_TASK_ID"
