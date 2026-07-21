#!/bin/bash
#SBATCH --job-name=rgbx_test
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --mail-type=FAIL
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common_env.sh"
srun "$PYTHON_BIN" "$PACKAGE_ROOT/run_experiment.py" test --index "$SLURM_ARRAY_TASK_ID" --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT" --python-bin "$PYTHON_BIN"

