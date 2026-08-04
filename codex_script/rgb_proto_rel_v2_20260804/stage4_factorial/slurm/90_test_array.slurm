#!/bin/bash
#SBATCH --job-name=rgb_v2_s4_test
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
: "${ALLOW_LOCKED_TEST:?Set ALLOW_LOCKED_TEST=YES only after all validation-based selection is frozen}"
test "$ALLOW_LOCKED_TEST" = YES
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate pytorch
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
DATASET_ROOT=${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset}
export PROJECT_ROOT DATASET_ROOT PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
CFG="$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/stage4_factorial/config/stage_config.json"
srun "${CONDA_PREFIX}/bin/python" -u "$PROJECT_ROOT/codex_script/rgb_proto_rel_v2_20260804/common/delegate_legacy.py" test --config "$CFG" --index "$SLURM_ARRAY_TASK_ID"
