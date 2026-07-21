#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
validate_job=$(sbatch --parsable "$SCRIPT_DIR/00_validate.slurm")
stats_job=$(sbatch --parsable --dependency="afterok:${validate_job}" "$SCRIPT_DIR/01_compute_crop_stats.slurm")
pretrain_job=$(sbatch --parsable --dependency="afterok:${stats_job}" "$SCRIPT_DIR/02_pretrain_array.slurm")
classifier_job=$(sbatch --parsable --dependency="afterok:${pretrain_job}" "$SCRIPT_DIR/03_classifier_array.slurm")
test_job=$(sbatch --parsable --dependency="afterok:${classifier_job}" "$SCRIPT_DIR/04_test_array.slurm")
features_job=$(sbatch --parsable --dependency="afterok:${pretrain_job}" "$SCRIPT_DIR/05_features_array.slurm")
summary_job=$(sbatch --parsable --dependency="afterok:${test_job}:${features_job}" "$SCRIPT_DIR/06_summarize.slurm")
echo "validate=$validate_job"
echo "stats=$stats_job"
echo "pretrain=$pretrain_job"
echo "classifier=$classifier_job"
echo "test=$test_job"
echo "features=$features_job"
echo "summary=$summary_job"
