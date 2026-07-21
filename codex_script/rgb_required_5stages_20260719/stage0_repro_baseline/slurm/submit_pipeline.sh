#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PRE=$(sbatch --parsable "$HERE/01_pretrain_array.slurm")
PRE=${PRE%%;*}
FT=$(sbatch --parsable --dependency="afterok:$PRE" "$HERE/02_finetune_array.slurm")
FT=${FT%%;*}
printf 'pretrain_job=%s\nfinetune_job=%s\n' "$PRE" "$FT"
printf 'Test remains locked. After freezing validation decisions run:\nsbatch --export=ALL,ALLOW_LOCKED_TEST=YES --dependency=afterok:%s %s/03_test_array.slurm\n' "$FT" "$HERE"
