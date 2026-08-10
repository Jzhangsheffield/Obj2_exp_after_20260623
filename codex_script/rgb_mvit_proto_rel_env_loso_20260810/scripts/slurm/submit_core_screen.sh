#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PREP=$(sbatch --parsable "$HERE/01_prepare.slurm")
S1=$(sbatch --parsable --dependency="afterok:$PREP" "$HERE/02_stage1_screen.slurm")
S2=$(sbatch --parsable --dependency="afterok:$S1" "$HERE/03_stage2a_screen.slurm")
S3=$(sbatch --parsable --dependency="afterok:$S2" "$HERE/05_stage3a_screen.slurm")
S4=$(sbatch --parsable --dependency="afterok:$S3" "$HERE/07_stage4_sensor_transfer.slurm")
printf 'prepare=%s\nstage1=%s\nstage2a=%s\nstage3a=%s\nstage4=%s\n' "$PREP" "$S1" "$S2" "$S3" "$S4"
