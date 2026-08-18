#!/usr/bin/env bash
set -euo pipefail
module load Anaconda3/2025.06-1
module load cuDNN/9.15.0.57-CUDA-12.9.1
#module load Anaconda3/2022.05
#module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate "${CONDA_ENV:-pytorch}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${PACKAGE_ROOT}/run.py"
EXP_CONFIG="${PACKAGE_ROOT}/config/experiment_config.json"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python}"
export PACKAGE_ROOT RUNNER EXP_CONFIG PYTHON_BOOTSTRAP

