#!/bin/bash
#SBATCH --job-name=rgbx_summary
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=FAIL
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/common_env.sh"
srun "$PYTHON_BIN" "$PACKAGE_ROOT/run_experiment.py" summarize --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT" --python-bin "$PYTHON_BIN"
