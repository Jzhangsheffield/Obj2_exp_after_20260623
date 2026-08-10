#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}
AUDIT_FILE="$PROJECT_ROOT/results/rgb_mvit_pr_env_loso_20260810/runtime/splits/protocol_audit.json"

if [[ ! -s "$AUDIT_FILE" ]]; then
    echo "ERROR: protocol preparation has not completed."
    echo "Missing or empty: $AUDIT_FILE"
    echo "Run 01_prepare manually in the working GPU environment, verify that it succeeds, then rerun this script."
    exit 1
fi

S1=$(sbatch --parsable "$HERE/02_stage1_screen.slurm")
S2=$(sbatch --parsable --dependency="afterok:$S1" "$HERE/03_stage2a_screen.slurm")
S3=$(sbatch --parsable --dependency="afterok:$S2" "$HERE/05_stage3a_screen.slurm")
S4=$(sbatch --parsable --dependency="afterok:$S3" "$HERE/07_stage4_sensor_transfer.slurm")
printf 'stage1=%s\nstage2a=%s\nstage3a=%s\nstage4=%s\n' "$S1" "$S2" "$S3" "$S4"
