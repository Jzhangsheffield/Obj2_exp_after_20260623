#!/bin/bash
#SBATCH --job-name=mvpr_test30
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail

source "${PROJECT_ROOT:-/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623}/codex_script/rgb_mvit_proto_rel_env_loso_20260810/scripts/slurm/common_env.sh"

AUDIT_FILE="$PROJECT_ROOT/results/rgb_mvit_pr_env_loso_20260810/runtime/splits/protocol_audit.json"
if [[ ! -s "$AUDIT_FILE" ]]; then
    echo "ERROR: missing protocol audit: $AUDIT_FILE"
    exit 1
fi

CHECKPOINT_ROOT="$PROJECT_ROOT/results/rgb_mvit_pr_env_loso_20260810/classifier/fold_MR"
CHECKPOINT_COUNT=$(find "$CHECKPOINT_ROOT" -type f -name best_val_balanced.pth | wc -l)
if [[ "$CHECKPOINT_COUNT" -lt 30 ]]; then
    echo "ERROR: expected at least 30 best_val_balanced.pth files under $CHECKPOINT_ROOT, found $CHECKPOINT_COUNT"
    exit 1
fi

run_stage() {
    local stage=$1
    local last_index=$2
    local index
    for index in $(seq 0 "$last_index"); do
        local experiment
        experiment=$("$PYTHON_BIN" "$PACKAGE_ROOT/run.py" list --stage "$stage" --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT" | "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin)[$index]['id'])")
        local result_csv="$PROJECT_ROOT/results/rgb_mvit_pr_env_loso_20260810/test/fold_MR/$stage/$experiment/test_results.csv"
        if [[ -s "$result_csv" ]]; then
            echo "===== SKIP completed fold_MR ${stage} ${experiment} ====="
            continue
        fi
        echo "===== TEST fold_MR ${stage} index=${index} ====="
        "$PYTHON_BIN" -u "$PACKAGE_ROOT/run.py" test \
            --stage "$stage" --index "$index" --fold MR \
            --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
    done
}

run_stage stage1 1
run_stage stage2a 10
run_stage stage3a 5
run_stage stage4 10

"$PYTHON_BIN" -u "$PACKAGE_ROOT/run.py" summarize \
    --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"

echo "Completed 30 fold_MR outer-test evaluations."
echo "Summary: $PROJECT_ROOT/results/rgb_mvit_pr_env_loso_20260810/summary/SUMMARY.md"
