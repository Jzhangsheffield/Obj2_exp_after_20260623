#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PRE=$(sbatch --parsable "$HERE/01b_pretrain_optional_r7_array.slurm")
PRE=${PRE%%;*}
FT=$(sbatch --parsable --dependency="afterok:$PRE" "$HERE/02b_finetune_optional_r7_array.slurm")
FT=${FT%%;*}
VAL=$(sbatch --parsable --dependency="afterok:$FT" "$HERE/03_summarize_validation.slurm")
VAL=${VAL%%;*}
printf 'optional_r7_pretrain_job=%s\noptional_r7_finetune_job=%s\nvalidation_summary_job=%s\n' "$PRE" "$FT" "$VAL"
