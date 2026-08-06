#!/bin/bash
set -euo pipefail
module load Anaconda3/2025.06-1
module load cuDNN/9.15.0.57-CUDA-12.9.1
source activate "${CONDA_ENV:-pytorch-el9}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
DATASET_ROOT=${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset}
PACKAGE_ROOT="$PROJECT_ROOT/codex_script/rgb_supcon_repair_20260806"
PYTHON_BIN=${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}
export PROJECT_ROOT DATASET_ROOT PACKAGE_ROOT PYTHON_BIN
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
