#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PRE=$(sbatch --parsable "$HERE/01_pretrain_array.slurm"); PRE=${PRE%%;*}
FT=$(sbatch --parsable --dependency="afterok:$PRE" "$HERE/02_finetune_array.slurm"); FT=${FT%%;*}
VAL=$(sbatch --parsable --dependency="afterok:$FT" "$HERE/03_summarize_validation.slurm"); VAL=${VAL%%;*}
printf 'pretrain_job=%s\nfinetune_job=%s\nvalidation_summary_job=%s\nLocked test was not submitted.\n' "$PRE" "$FT" "$VAL"
