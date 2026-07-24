#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PRE=$(sbatch --parsable "$HERE/01_pretrain_required_array.slurm")
PRE=${PRE%%;*}
FT=$(sbatch --parsable --dependency="afterok:$PRE" "$HERE/02_finetune_required_array.slurm")
FT=${FT%%;*}
VAL=$(sbatch --parsable --dependency="afterok:$FT" "$HERE/03_summarize_validation.slurm")
VAL=${VAL%%;*}
printf 'required_pretrain_job=%s\nrequired_finetune_job=%s\nvalidation_summary_job=%s\n' "$PRE" "$FT" "$VAL"
printf 'Do not run the locked test while Stage 5 settings are still being selected.\n'
