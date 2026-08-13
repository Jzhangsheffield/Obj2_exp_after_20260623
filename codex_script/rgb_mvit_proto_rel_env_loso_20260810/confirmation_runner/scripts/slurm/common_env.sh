#!/bin/bash
set -euo pipefail
module load Anaconda3/2025.06-1
module load cuDNN/9.15.0.57-CUDA-12.9.1
#module load Anaconda3/2022.05
#module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate "${CONDA_ENV:-pytorch}"
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
DATASET_ROOT=${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset}
PARENT_PACKAGE="$PROJECT_ROOT/codex_script/rgb_mvit_proto_rel_env_loso_20260810"
CONFIRM_ROOT="$PARENT_PACKAGE/confirmation_runner"
PYTHON_BIN=${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}
export PROJECT_ROOT DATASET_ROOT PARENT_PACKAGE CONFIRM_ROOT PYTHON_BIN
# Parent package must precede PROJECT_ROOT because confirmation_runner and its
# parent both contain a package named "common". run_unified.py also enforces
# and audits this ordering, while this export keeps child Python processes
# consistent on Stanage.
export PYTHONPATH="$PARENT_PACKAGE:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONHASHSEED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
