#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
AUDIT_FILE="$PROJECT_ROOT/results/rgb_mvit_pr_env_loso_20260810/runtime/splits/protocol_audit.json"

if [[ ! -s "$AUDIT_FILE" ]]; then
    echo "ERROR: protocol preparation has not completed."
    echo "Missing or empty: $AUDIT_FILE"
    echo "Run 01_prepare.slurm, verify the audit file, then retry."
    exit 1
fi

STAGE8_JOB=$(sbatch --parsable "$HERE/13_stage8_loso4_explore.slurm")
SUMMARY_JOB=$(sbatch --parsable --dependency="afterok:$STAGE8_JOB" "$HERE/11_summarize.slurm")
printf 'stage8=%s\nsummary=%s\n' "$STAGE8_JOB" "$SUMMARY_JOB"

