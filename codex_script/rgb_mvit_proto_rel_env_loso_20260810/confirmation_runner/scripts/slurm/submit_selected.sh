#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common_env.sh"

CONFIGS=phase1_locked
SEEDS=2,3
SUBJECTS=M,J,N
STAGES=pretrain,finetune,test,summarize
NAME="confirm_$(date +%Y%m%d_%H%M%S)"
MAX_PARALLEL=4
DRY_RUN=0

usage() {
  echo "Usage: bash submit_selected.sh [--configs GROUP_OR_IDS] [--seeds 2,3] [--subjects M,J,N] [--stages pretrain,finetune,test,summarize] [--name NAME] [--max-parallel N] [--dry-run]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --configs) CONFIGS=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --subjects) SUBJECTS=$2; shift 2 ;;
    --stages) STAGES=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --max-parallel) MAX_PARALLEL=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

AUDIT_FILE="$PROJECT_ROOT/results/rgb_mvit_pr_env_loso_20260810/runtime/splits/protocol_audit.json"
if [[ ! -s "$AUDIT_FILE" ]]; then
  echo "ERROR: protocol splits are missing: $AUDIT_FILE"
  echo "Run the parent package 01_prepare manually once, then submit confirmation jobs."
  exit 1
fi

MANIFEST=$(
  "$PYTHON_BIN" "$CONFIRM_ROOT/run_confirmation.py" build-manifest \
    --configs "$CONFIGS" --seeds "$SEEDS" --subjects "$SUBJECTS" --name "$NAME" \
    --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT"
)
COUNT=$("$PYTHON_BIN" "$CONFIRM_ROOT/run_confirmation.py" count-manifest --manifest "$MANIFEST")
if [[ "$COUNT" -lt 1 ]]; then
  echo "ERROR: empty manifest"
  exit 1
fi
LAST=$((COUNT - 1))
ARRAY="0-${LAST}%${MAX_PARALLEL}"
echo "Manifest: $MANIFEST"
echo "Runs: $COUNT (configs=$CONFIGS; seeds=$SEEDS; subjects=$SUBJECTS)"
echo "Stages: $STAGES"

submit_array() {
  local script=$1
  local dependency=${2:-}
  local command=(sbatch --parsable --array="$ARRAY" --export="ALL,CONFIRM_MANIFEST=$MANIFEST")
  # Equal-sized arrays use task-wise dependency: index i only waits for index i.
  # One failed run therefore does not block unrelated configurations.
  if [[ -n "$dependency" ]]; then command+=(--dependency="aftercorr:$dependency"); fi
  command+=("$HERE/$script")
  if [[ "$DRY_RUN" -eq 1 ]]; then printf 'DRY-RUN:' >&2; printf ' %q' "${command[@]}" >&2; printf '\n' >&2; echo "dry_${script}"; else "${command[@]}"; fi
}

submit_single() {
  local script=$1
  local dependency=${2:-}
  local command=(sbatch --parsable --export="ALL,CONFIRM_MANIFEST=$MANIFEST")
  if [[ -n "$dependency" ]]; then command+=(--dependency="afterok:$dependency"); fi
  command+=("$HERE/$script")
  if [[ "$DRY_RUN" -eq 1 ]]; then printf 'DRY-RUN:' >&2; printf ' %q' "${command[@]}" >&2; printf '\n' >&2; echo "dry_${script}"; else "${command[@]}"; fi
}

dependency=""
declare -A JOBS
IFS=',' read -ra REQUESTED <<< "$STAGES"
last_rank=0
for stage in "${REQUESTED[@]}"; do
  case "$stage" in
    pretrain) rank=1 ;;
    finetune) rank=2 ;;
    test) rank=3 ;;
    summarize) rank=4 ;;
    *) echo "ERROR: invalid stage $stage"; exit 2 ;;
  esac
  if [[ "$rank" -le "$last_rank" ]]; then
    echo "ERROR: stages must be unique and ordered as pretrain,finetune,test,summarize"
    exit 2
  fi
  last_rank=$rank
  case "$stage" in
    pretrain) job=$(submit_array 01_pretrain_selected.slurm "$dependency") ;;
    finetune) job=$(submit_array 02_finetune_selected.slurm "$dependency") ;;
    test) job=$(submit_array 03_test_selected.slurm "$dependency") ;;
    summarize) job=$(submit_single 04_summarize_selected.slurm "$dependency") ;;
  esac
  JOBS[$stage]=$job
  dependency=$job
done
for stage in "${REQUESTED[@]}"; do echo "$stage=${JOBS[$stage]}"; done
