#!/bin/bash
#SBATCH --job-name=mvpr_u_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common_env.sh"
: "${UNIFIED_MANIFEST:?UNIFIED_MANIFEST is required}"
"$PYTHON_BIN" -u "$CONFIRM_ROOT/run_unified.py" run-task --manifest "$UNIFIED_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --phase evaluate --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
