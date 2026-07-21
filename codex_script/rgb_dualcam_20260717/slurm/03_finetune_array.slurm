#!/bin/bash
#SBATCH --job-name=rgb_dc_ft
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=14:00:00
#SBATCH --array=0-8%3
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/ft_and_test/out/rgb_dc_ft_%A_%a.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/ft_and_test/out/rgb_dc_ft_%A_%a.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset}"
PACKAGE="${PROJECT_ROOT}/codex_script/rgb_dualcam_20260717"
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate pytorch
if [[ -z "${CONDA_PREFIX:-}" || ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "[Error] pytorch conda environment was not activated correctly." >&2
    exit 2
fi
PYTHON_BIN="${CONDA_PREFIX}/bin/python"
"${PYTHON_BIN}" -c 'import sys, torch; assert sys.version_info >= (3, 10), sys.version; print("Python:", sys.executable, sys.version.split()[0]); print("PyTorch:", torch.__version__)'
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
STEP_PID=""
requeue_job() {
    if [[ -n "${STEP_PID}" ]]; then kill -TERM "${STEP_PID}" 2>/dev/null || true; wait "${STEP_PID}" || true; fi
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap requeue_job USR1
CMD=("${PYTHON_BIN}" -u "${PACKAGE}/run_finetune.py" --index "${SLURM_ARRAY_TASK_ID}" --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}")
if [[ "${DRY_RUN:-0}" == "1" ]]; then CMD+=(--dry-run); fi
"${CMD[@]}" &
STEP_PID=$!
set +e
wait "${STEP_PID}"
STATUS=$?
set -e
exit "${STATUS}"
