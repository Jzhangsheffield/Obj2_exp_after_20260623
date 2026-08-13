#!/bin/bash
#SBATCH --job-name=mvpr_cf_stat
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
CONFIRM_ROOT="$PROJECT_ROOT/codex_script/rgb_mvit_proto_rel_env_loso_20260810/confirmation_runner"
ANALYSIS_PYTHON_BIN=${ANALYSIS_PYTHON_BIN:-python3}
: "${CONFIRM_MANIFEST:?CONFIRM_MANIFEST is required}"
RESULTS_ROOT="$PROJECT_ROOT/results/rgb_mvit_pr_confirm_20260812"
"$ANALYSIS_PYTHON_BIN" -u "$CONFIRM_ROOT/tools/analyze_confirmation.py" --results-root "$RESULTS_ROOT" --manifest "$CONFIRM_MANIFEST"
