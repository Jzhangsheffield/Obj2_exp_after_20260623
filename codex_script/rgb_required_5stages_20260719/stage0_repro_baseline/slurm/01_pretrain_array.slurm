#!/bin/bash
#SBATCH --job-name=rgb_s0_cl
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=18:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
set -euo pipefail
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate pytorch
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
DATASET_ROOT=${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset}
export PROJECT_ROOT DATASET_ROOT PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
CFG="$PROJECT_ROOT/codex_script/rgb_required_5stages_20260719/stage0_repro_baseline/config/stage_config.json"
srun "${CONDA_PREFIX}/bin/python" -u "$PROJECT_ROOT/codex_script/rgb_required_5stages_20260719/common/run_pretrain.py" --config "$CFG" --index "$SLURM_ARRAY_TASK_ID"

