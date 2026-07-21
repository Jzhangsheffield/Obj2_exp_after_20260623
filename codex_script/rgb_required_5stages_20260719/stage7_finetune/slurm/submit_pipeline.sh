#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(sbatch --parsable "$HERE/01_pretrain_array.slurm")
SRC=${SRC%%;*}
FT=$(sbatch --parsable --dependency="afterok:$SRC" "$HERE/02_finetune_array.slurm")
FT=${FT%%;*}
printf 'source_validation_job=%s\nfinetune_job=%s\n' "$SRC" "$FT"
printf 'After validation is frozen: sbatch --export=ALL,ALLOW_LOCKED_TEST=YES --dependency=afterok:%s %s/03_test_array.slurm\n' "$FT" "$HERE"
