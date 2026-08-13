#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common_env.sh"

PRESET=""; TASK=""; PROTOCOL=""; CONFIGS=""; AUGS=""; SAMPLINGS=""; SEEDS=""; SUBJECTS=""; STAGES=""
NAME="unified_$(date +%Y%m%d_%H%M%S)"; MAX_PARALLEL=4; DRY_RUN=0
usage() { echo "Usage: bash submit_unified.sh [--preset NAME] [--task t15|t17 --protocol subject_dev|final_refit --configs IDS --augmentations IDS --samplings IDS --seeds IDS --subjects IDS] --stages pretrain,finetune,evaluate|test,summarize [--name NAME] [--max-parallel N] [--dry-run]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset) PRESET=$2; shift 2;; --task) TASK=$2; shift 2;; --protocol) PROTOCOL=$2; shift 2;;
    --configs) CONFIGS=$2; shift 2;; --augmentations) AUGS=$2; shift 2;; --samplings) SAMPLINGS=$2; shift 2;;
    --seeds) SEEDS=$2; shift 2;; --subjects) SUBJECTS=$2; shift 2;; --stages) STAGES=$2; shift 2;;
    --name) NAME=$2; shift 2;; --max-parallel) MAX_PARALLEL=$2; shift 2;; --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;; *) echo "Unknown option: $1"; usage; exit 2;;
  esac
done
if [[ -z "$STAGES" ]]; then echo "ERROR: --stages is required"; exit 2; fi
SPLIT_AUDIT="$PROJECT_ROOT/results/rgb_mvit_pr_unified_followup_20260813/runtime/splits/protocol_audit.json"
if [[ ! -s "$SPLIT_AUDIT" ]]; then
  echo "ERROR: prepared manifests are missing: $SPLIT_AUDIT"
  echo "Run 00_prepare_unified.bat locally or run_unified.py prepare on HPC before submitting GPU jobs."
  exit 1
fi

BUILD=("$PYTHON_BIN" "$CONFIRM_ROOT/run_unified.py" build-manifest --name "$NAME" --platform hpc --project-root "$PROJECT_ROOT" --dataset-root "$DATASET_ROOT")
[[ -n "$PRESET" ]] && BUILD+=(--preset "$PRESET"); [[ -n "$TASK" ]] && BUILD+=(--task "$TASK")
[[ -n "$PROTOCOL" ]] && BUILD+=(--protocol "$PROTOCOL"); [[ -n "$CONFIGS" ]] && BUILD+=(--configs "$CONFIGS")
[[ -n "$AUGS" ]] && BUILD+=(--augmentations "$AUGS"); [[ -n "$SAMPLINGS" ]] && BUILD+=(--samplings "$SAMPLINGS")
[[ -n "$SEEDS" ]] && BUILD+=(--seeds "$SEEDS"); [[ -n "$SUBJECTS" ]] && BUILD+=(--subjects "$SUBJECTS")
MANIFEST=$("${BUILD[@]}")
COUNT=$("$PYTHON_BIN" "$CONFIRM_ROOT/run_unified.py" count-manifest --manifest "$MANIFEST")
LAST=$((COUNT - 1)); ARRAY="0-${LAST}%${MAX_PARALLEL}"
echo "Manifest: $MANIFEST"; echo "Runs: $COUNT"; echo "Stages: $STAGES"

submit_array() {
  local script=$1 dependency=${2:-}; local command=(sbatch --parsable --array="$ARRAY" --export="ALL,UNIFIED_MANIFEST=$MANIFEST")
  [[ -n "$dependency" ]] && command+=(--dependency="aftercorr:$dependency"); command+=("$HERE/$script")
  if [[ "$DRY_RUN" -eq 1 ]]; then printf 'DRY-RUN:' >&2; printf ' %q' "${command[@]}" >&2; printf '\n' >&2; echo "dry_$script"; else "${command[@]}"; fi
}
submit_single() {
  local script=$1 dependency=${2:-}; local command=(sbatch --parsable --export="ALL,UNIFIED_MANIFEST=$MANIFEST")
  [[ -n "$dependency" ]] && command+=(--dependency="afterok:$dependency"); command+=("$HERE/$script")
  if [[ "$DRY_RUN" -eq 1 ]]; then printf 'DRY-RUN:' >&2; printf ' %q' "${command[@]}" >&2; printf '\n' >&2; echo "dry_$script"; else "${command[@]}"; fi
}

dependency=""; last_rank=0; IFS=',' read -ra REQUESTED <<< "$STAGES"
for stage in "${REQUESTED[@]}"; do
  case "$stage" in pretrain) rank=1;; finetune) rank=2;; evaluate|test) rank=3;; summarize) rank=4;; *) echo "ERROR: invalid stage $stage"; exit 2;; esac
  if [[ "$rank" -le "$last_rank" ]]; then echo "ERROR: stages must be unique and ordered"; exit 2; fi; last_rank=$rank
  case "$stage" in pretrain) job=$(submit_array u01_pretrain.slurm "$dependency");; finetune) job=$(submit_array u02_finetune.slurm "$dependency");; evaluate) job=$(submit_array u03_evaluate.slurm "$dependency");; test) job=$(submit_array u04_test.slurm "$dependency");; summarize) job=$(submit_single u05_summarize.slurm "$dependency");; esac
  echo "$stage=$job"; dependency=$job
done
