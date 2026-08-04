#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PRE=$(sbatch --parsable "$HERE/01_pretrain_array.slurm"); PRE=${PRE%%;*}
AUDIT=$(sbatch --parsable --dependency="afterok:$PRE" "$HERE/02_audit.slurm"); AUDIT=${AUDIT%%;*}
printf 'pretrain_job=%s\naudit_job=%s\nReview the null-path audit before starting Stage 1.\n' "$PRE" "$AUDIT"
