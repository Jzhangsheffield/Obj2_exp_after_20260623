#!/bin/bash
set -euo pipefail
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate "${CONDA_ENV:-pytorch}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
DATASET_ROOT=${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset}
PACKAGE_ROOT="$PROJECT_ROOT/codex_script/rgb_supcon_repair_20260806"
PYTHON_BIN=${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}
export PROJECT_ROOT DATASET_ROOT PACKAGE_ROOT PYTHON_BIN
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
