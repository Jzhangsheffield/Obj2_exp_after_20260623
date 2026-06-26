#!/bin/bash
#SBATCH --job-name=rgb_N_take_put_adamw_22
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=15:00:00
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --array=0-21%4
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/train/out/rgb_N_take_put_%x_%A_%a.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/train/out/rgb_N_take_put_%x_%A_%a.err

set -euo pipefail

# ============================================================
# RGB SupLoss ablation for N_as_test / train_manifest_take_put.jsonl
# 22 tasks: suploss_only + proto(1,2,3) + rel/proto_rel with proto(1,2,3) x preview_ema_momentum(0.3,0.5,0.8)
# Normalization is read from N_as_test/train_normalization_stats/take_put_rgb_stats.json and rounded to 4 decimals.
# ============================================================

# ------------------------------
# 1) Project / data paths
# ------------------------------
PROJECT_ROOT="/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623"
SCRIPT_PATH="${PROJECT_ROOT}/train/MoCo_main_supcon_mapstyle_varproto_debug_topk_adamw.py"
DATASET_ROOT="/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset"
TRAIN_MANIFEST_NAME="${DATASET_ROOT}/N_as_test/train_manifest_take_put.jsonl"
LABEL_MAP_JSON="${DATASET_ROOT}/label_map_take_put.json"
NORMALIZATION_STATS_JSON="${DATASET_ROOT}/N_as_test/train_normalization_stats/take_put_rgb_stats.json"
SAVE_ROOT="${PROJECT_ROOT}/results/rgb_N_take_put_adamw_22"
LOG_ROOT="${PROJECT_ROOT}/bash/logs"

# ------------------------------
# 2) Runtime / environment
# ------------------------------
PYTHON_BIN="python"
ANACONDA_MODULE="Anaconda3/2022.05"
CUDNN_MODULE="cuDNN/8.9.2.26-CUDA-12.1.1"
CONDA_ENV_NAME="pytorch"
SKIP_ENV_SETUP=0
DRY_RUN=${DRY_RUN:-0}

# ------------------------------
# 3) Dataset / RGB transform parameters
# ------------------------------
TIER_MODE="tier1"
N_FRAMES=16
RGB_MEAN=(0.3725 0.3828 0.3921)
RGB_STD=(0.2923 0.2715 0.2640)
RGB_OUT_HW=(224 224)
RRC_SCALE=(0.6 1.0)
RRC_RATIO=(0.75 1.3333333333)
RGB_APPLY_SPATIAL_AUG="true"
RGB_HFLIP_P=0.5
RGB_VFLIP_P=0.5
RGB_JITTER_P=0.5
RGB_JITTER_BRIGHTNESS=0.24
RGB_JITTER_CONTRAST=0.24
RGB_JITTER_SATURATION=0.24
RGB_JITTER_HUE=0.16
RGB_GRAY_P=0.5
RGB_BLUR_P=0.5
RGB_BLUR_KERNEL=5
RGB_BLUR_SIGMA=(0.1 1.0)

# ------------------------------
# 4) DataLoader parameters
# ------------------------------
BATCH_SIZE=64
NUM_WORKERS=8
PREFETCH_FACTOR=""
PIN_MEMORY=0
VERIFY_PATHS_ON_INIT=1
PROTO_REFRESH_BATCH_SIZE=64
PROTO_REFRESH_NUM_WORKERS=8
PROTO_REFRESH_PREFETCH_FACTOR=""
PROTO_REFRESH_PIN_MEMORY=""
PROTO_REFRESH_VERIFY_PATHS_ON_INIT=1

# ------------------------------
# 5) Model / contrastive parameters
# ------------------------------
MODEL_DEPTH=18
PROJ_DIM=128
K_QUEUE=1920
TEMPERATURE=0.1
USE_MLP=1
CONTRASTIVE_LOSS="suploss"
NUM_POSITIVE=6
EXCLUDE_INVALID_QUEUE=0

# ------------------------------
# 6) Prototype / relative loss parameters
# ------------------------------
WARMUP_EPOCHS=50
RECLUSTER_INTERVAL=10
NUM_PROTOTYPES_PER_CLASS=""
LAMBDA_PROTO=1.0
PROTO_TEMPERATURE=0.07
ENABLE_PROTOTYPE_TEMPERATURE_SCALING=0
PROTO_TEMPERATURE_EPS=1e-6
PROTO_KMEANS_RANDOM_STATE=42
PROTO_KMEANS_N_INIT=10
PROTO_KMEANS_MAX_ITER=300
LAMBDA_REL=1.0
PROTO_EMA_MOMENTUM=0.99
REL_SAME_MARGIN=0.01
REL_DIFF_MARGIN=0.01
REL_SAME_WEIGHT=1.0
REL_DIFF_WEIGHT=1.0
REL_TOPK_DIFF_CLASSES=3

# ------------------------------
# 7) Loss-stage schedule parameters
# ------------------------------
ENABLE_LOSS_STAGE_SCHEDULE=1
PROTO_LOSS_START_EPOCH=50
REL_LOSS_START_EPOCH=50
REL_LOSS_END_EPOCH=200
REL_LAMBDA_SCHEDULE="cosine"

# ------------------------------
# 8) Optimizer / training control
# ------------------------------
EPOCHS=200
START_EPOCH=0
LEARNING_RATE=1e-3
USE_COSINE_LR=0
SCHEDULE=(50 100 150)
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
OPTIMIZER="adamw"
ADAMW_BETAS=(0.9 0.999)
ADAMW_EPS=1e-8
ADAMW_AMSGRAD="false"
NO_DDP=1
SEED=""
PRINT_FREQ=10
SAVE_INTERVAL=50
USE_SYNCBN="true"
FIND_UNUSED_PARAMETERS="false"

# ------------------------------
# 9) Debug parameters
# ------------------------------
DEBUG_MODE=0
DEBUG_LOG_INTERVAL=20
DEBUG_GRAD_STATS="true"
DEBUG_PARAM_UPDATE_STATS="true"
DEBUG_BATCH_LABEL_STATS="true"
DEBUG_PROTO_STATS="true"
DEBUG_FEATURE_STATS="true"
DEBUG_NONFINITE_CHECK="true"
DEBUG_ABORT_ON_NONFINITE="false"
DEBUG_GRAD_TOPK=8
DEBUG_PARAM_PATTERNS="module.encoder_q.fc,encoder_q.fc,module.encoder_q.layer4,encoder_q.layer4,module.encoder_q.layer3,encoder_q.layer3"
DEBUG_PARAM_FALLBACK_LAST_N=4
DEBUG_WRITE_JSONL="false"
DEBUG_JSONL_NAME="debug_train_log.jsonl"

ABLATION_MODES=(
  "contrastive_only"
  "contrastive_proto" "contrastive_proto" "contrastive_proto"
  "contrastive_rel" "contrastive_rel" "contrastive_rel" "contrastive_rel" "contrastive_rel" "contrastive_rel" "contrastive_rel" "contrastive_rel" "contrastive_rel"
  "contrastive_proto_rel" "contrastive_proto_rel" "contrastive_proto_rel" "contrastive_proto_rel" "contrastive_proto_rel" "contrastive_proto_rel" "contrastive_proto_rel" "contrastive_proto_rel" "contrastive_proto_rel"
)

DEFAULT_NUM_PROTOTYPES=(
  "1"
  "1" "2" "3"
  "1" "1" "1" "2" "2" "2" "3" "3" "3"
  "1" "1" "1" "2" "2" "2" "3" "3" "3"
)

PREVIEW_EMA_MOMENTA=(
  ""
  "" "" ""
  "0.3" "0.5" "0.8" "0.3" "0.5" "0.8" "0.3" "0.5" "0.8"
  "0.3" "0.5" "0.8" "0.3" "0.5" "0.8" "0.3" "0.5" "0.8"
)

RUN_NAMES=(
  "suploss_only"
  "suploss_proto_p1" "suploss_proto_p2" "suploss_proto_p3"
  "suploss_rel_p1_prem0.3" "suploss_rel_p1_prem0.5" "suploss_rel_p1_prem0.8"
  "suploss_rel_p2_prem0.3" "suploss_rel_p2_prem0.5" "suploss_rel_p2_prem0.8"
  "suploss_rel_p3_prem0.3" "suploss_rel_p3_prem0.5" "suploss_rel_p3_prem0.8"
  "suploss_proto_rel_p1_prem0.3" "suploss_proto_rel_p1_prem0.5" "suploss_proto_rel_p1_prem0.8"
  "suploss_proto_rel_p2_prem0.3" "suploss_proto_rel_p2_prem0.5" "suploss_proto_rel_p2_prem0.8"
  "suploss_proto_rel_p3_prem0.3" "suploss_proto_rel_p3_prem0.5" "suploss_proto_rel_p3_prem0.8"
)

TOTAL_TASKS=${#RUN_NAMES[@]}
if [[ ${#ABLATION_MODES[@]} -ne ${TOTAL_TASKS} || ${#DEFAULT_NUM_PROTOTYPES[@]} -ne ${TOTAL_TASKS} || ${#PREVIEW_EMA_MOMENTA[@]} -ne ${TOTAL_TASKS} ]]; then
    echo "[Error] Experiment arrays have inconsistent lengths." >&2
    exit 2
fi

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${TOTAL_TASKS}" ]]; then
    echo "[Error] SLURM_ARRAY_TASK_ID=${IDX} is out of range. Expected 0-$((TOTAL_TASKS - 1))." >&2
    exit 2
fi

ABLATION_MODE="${ABLATION_MODES[$IDX]}"
DEFAULT_NUM_PROTOTYPE="${DEFAULT_NUM_PROTOTYPES[$IDX]}"
PREVIEW_EMA_MOMENTUM="${PREVIEW_EMA_MOMENTA[$IDX]}"
RUN_NAME="${RUN_NAMES[$IDX]}"
WEIGHT_SAVE_PATH="${SAVE_ROOT}/${RUN_NAME}"

mkdir -p "${SAVE_ROOT}" "${LOG_ROOT}"

if [[ "${SKIP_ENV_SETUP}" != "1" ]]; then
    module load "${ANACONDA_MODULE}"
    module load "${CUDNN_MODULE}"
    source activate "${CONDA_ENV_NAME}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

EXTRA_ARGS=()
add_bool_optional_arg() {
    local value="$1"
    local true_flag="$2"
    local false_flag="$3"
    case "${value}" in
        1|true|TRUE|yes|YES) EXTRA_ARGS+=("${true_flag}") ;;
        0|false|FALSE|no|NO) EXTRA_ARGS+=("${false_flag}") ;;
        "") ;;
        *) echo "[Error] Boolean option ${true_flag}/${false_flag} got invalid value: ${value}" >&2; exit 2 ;;
    esac
}

if [[ "${NO_DDP}" == "1" ]]; then EXTRA_ARGS+=(--no_ddp); fi
if [[ "${VERIFY_PATHS_ON_INIT}" == "1" ]]; then EXTRA_ARGS+=(--verify_paths_on_init); fi
if [[ "${PROTO_REFRESH_VERIFY_PATHS_ON_INIT}" == "1" ]]; then EXTRA_ARGS+=(--proto_refresh_verify_paths_on_init); fi
if [[ "${PIN_MEMORY}" == "1" ]]; then EXTRA_ARGS+=(--pin_memory); fi
if [[ "${USE_MLP}" == "1" ]]; then EXTRA_ARGS+=(--mlp); fi
if [[ "${EXCLUDE_INVALID_QUEUE}" == "1" ]]; then EXTRA_ARGS+=(--exclude_invalid_queue); fi
if [[ "${ENABLE_PROTOTYPE_TEMPERATURE_SCALING}" == "1" ]]; then EXTRA_ARGS+=(--enable_prototype_temperature_scaling); fi
if [[ "${ENABLE_LOSS_STAGE_SCHEDULE}" == "1" ]]; then EXTRA_ARGS+=(--enable_loss_stage_schedule); fi
if [[ "${USE_COSINE_LR}" == "1" ]]; then EXTRA_ARGS+=(--cos); fi
if [[ "${DEBUG_MODE}" == "1" ]]; then EXTRA_ARGS+=(--debug_mode); fi
if [[ -n "${PREFETCH_FACTOR}" ]]; then EXTRA_ARGS+=(--prefetch_factor "${PREFETCH_FACTOR}"); fi
if [[ -n "${PROTO_REFRESH_PREFETCH_FACTOR}" ]]; then EXTRA_ARGS+=(--proto_refresh_prefetch_factor "${PROTO_REFRESH_PREFETCH_FACTOR}"); fi
if [[ -n "${NUM_PROTOTYPES_PER_CLASS}" ]]; then EXTRA_ARGS+=(--num_prototypes_per_class "${NUM_PROTOTYPES_PER_CLASS}"); fi
if [[ -n "${SEED}" ]]; then EXTRA_ARGS+=(--seed "${SEED}"); fi
if [[ "${ABLATION_MODE}" == "contrastive_rel" || "${ABLATION_MODE}" == "contrastive_proto_rel" ]]; then EXTRA_ARGS+=(--preview_ema_momentum "${PREVIEW_EMA_MOMENTUM}"); fi

add_bool_optional_arg "${RGB_APPLY_SPATIAL_AUG}" "--rgb_apply_spatial_aug" "--no-rgb_apply_spatial_aug"
add_bool_optional_arg "${PROTO_REFRESH_PIN_MEMORY}" "--proto_refresh_pin_memory" "--no-proto_refresh_pin_memory"
add_bool_optional_arg "${ADAMW_AMSGRAD}" "--adamw_amsgrad" "--no-adamw_amsgrad"
add_bool_optional_arg "${USE_SYNCBN}" "--use_syncbn" "--no-use_syncbn"
add_bool_optional_arg "${FIND_UNUSED_PARAMETERS}" "--find_unused_parameters" "--no-find_unused_parameters"
add_bool_optional_arg "${DEBUG_GRAD_STATS}" "--debug_grad_stats" "--no-debug_grad_stats"
add_bool_optional_arg "${DEBUG_PARAM_UPDATE_STATS}" "--debug_param_update_stats" "--no-debug_param_update_stats"
add_bool_optional_arg "${DEBUG_BATCH_LABEL_STATS}" "--debug_batch_label_stats" "--no-debug_batch_label_stats"
add_bool_optional_arg "${DEBUG_PROTO_STATS}" "--debug_proto_stats" "--no-debug_proto_stats"
add_bool_optional_arg "${DEBUG_FEATURE_STATS}" "--debug_feature_stats" "--no-debug_feature_stats"
add_bool_optional_arg "${DEBUG_NONFINITE_CHECK}" "--debug_nonfinite_check" "--no-debug_nonfinite_check"
add_bool_optional_arg "${DEBUG_ABORT_ON_NONFINITE}" "--debug_abort_on_nonfinite" "--no-debug_abort_on_nonfinite"
add_bool_optional_arg "${DEBUG_WRITE_JSONL}" "--debug_write_jsonl" "--no-debug_write_jsonl"

TRAIN_CMD=(
    "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
    --dataset_root "${DATASET_ROOT}"
    --train_manifest_name "${TRAIN_MANIFEST_NAME}"
    --label_map_json "${LABEL_MAP_JSON}"
    --weight_save_path "${WEIGHT_SAVE_PATH}"
    --tier_mode "${TIER_MODE}"
    --n_frames "${N_FRAMES}"
    --rgb_mean "${RGB_MEAN[@]}"
    --rgb_std "${RGB_STD[@]}"
    --rgb_out_hw "${RGB_OUT_HW[@]}"
    --rrc_scale "${RRC_SCALE[@]}"
    --rrc_ratio "${RRC_RATIO[@]}"
    --rgb_hflip_p "${RGB_HFLIP_P}"
    --rgb_vflip_p "${RGB_VFLIP_P}"
    --rgb_jitter_p "${RGB_JITTER_P}"
    --rgb_jitter_brightness "${RGB_JITTER_BRIGHTNESS}"
    --rgb_jitter_contrast "${RGB_JITTER_CONTRAST}"
    --rgb_jitter_saturation "${RGB_JITTER_SATURATION}"
    --rgb_jitter_hue "${RGB_JITTER_HUE}"
    --rgb_gray_p "${RGB_GRAY_P}"
    --rgb_blur_p "${RGB_BLUR_P}"
    --rgb_blur_kernel "${RGB_BLUR_KERNEL}"
    --rgb_blur_sigma "${RGB_BLUR_SIGMA[@]}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --model_depth "${MODEL_DEPTH}"
    --proj_dim "${PROJ_DIM}"
    --K_queue "${K_QUEUE}"
    --temperature "${TEMPERATURE}"
    --contrastive_loss "${CONTRASTIVE_LOSS}"
    --num_positive "${NUM_POSITIVE}"
    --ablation_mode "${ABLATION_MODE}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --recluster_interval "${RECLUSTER_INTERVAL}"
    --default_num_prototypes "${DEFAULT_NUM_PROTOTYPE}"
    --lambda_proto "${LAMBDA_PROTO}"
    --proto_temperature "${PROTO_TEMPERATURE}"
    --proto_temperature_eps "${PROTO_TEMPERATURE_EPS}"
    --proto_kmeans_random_state "${PROTO_KMEANS_RANDOM_STATE}"
    --proto_kmeans_n_init "${PROTO_KMEANS_N_INIT}"
    --proto_kmeans_max_iter "${PROTO_KMEANS_MAX_ITER}"
    --proto_refresh_batch_size "${PROTO_REFRESH_BATCH_SIZE}"
    --proto_refresh_num_workers "${PROTO_REFRESH_NUM_WORKERS}"
    --lambda_rel "${LAMBDA_REL}"
    --proto_ema_momentum "${PROTO_EMA_MOMENTUM}"
    --rel_same_margin "${REL_SAME_MARGIN}"
    --rel_diff_margin "${REL_DIFF_MARGIN}"
    --rel_same_weight "${REL_SAME_WEIGHT}"
    --rel_diff_weight "${REL_DIFF_WEIGHT}"
    --rel_topk_diff_classes "${REL_TOPK_DIFF_CLASSES}"
    --proto_loss_start_epoch "${PROTO_LOSS_START_EPOCH}"
    --rel_loss_start_epoch "${REL_LOSS_START_EPOCH}"
    --rel_loss_end_epoch "${REL_LOSS_END_EPOCH}"
    --rel_lambda_schedule "${REL_LAMBDA_SCHEDULE}"
    --epochs "${EPOCHS}"
    --start_epoch "${START_EPOCH}"
    --learning_rate "${LEARNING_RATE}"
    --schedule "${SCHEDULE[@]}"
    --weight_decay "${WEIGHT_DECAY}"
    --momentum "${MOMENTUM}"
    --optimizer "${OPTIMIZER}"
    --adamw_betas "${ADAMW_BETAS[@]}"
    --adamw_eps "${ADAMW_EPS}"
    --print_freq "${PRINT_FREQ}"
    --save_interval "${SAVE_INTERVAL}"
    --debug_log_interval "${DEBUG_LOG_INTERVAL}"
    --debug_grad_topk "${DEBUG_GRAD_TOPK}"
    --debug_param_patterns "${DEBUG_PARAM_PATTERNS}"
    --debug_param_fallback_last_n "${DEBUG_PARAM_FALLBACK_LAST_N}"
    --debug_jsonl_name "${DEBUG_JSONL_NAME}"
    "${EXTRA_ARGS[@]}"
)

echo "======================================"
echo "SLURM_ARRAY_TASK_ID      = ${IDX}"
echo "RUN_NAME                 = ${RUN_NAME}"
echo "ABLATION_MODE            = ${ABLATION_MODE}"
echo "DEFAULT_NUM_PROTOTYPE    = ${DEFAULT_NUM_PROTOTYPE}"
echo "PREVIEW_EMA_MOMENTUM     = ${PREVIEW_EMA_MOMENTUM}"
echo "TRAIN_MANIFEST_NAME      = ${TRAIN_MANIFEST_NAME}"
echo "NORMALIZATION_STATS_JSON = ${NORMALIZATION_STATS_JSON}"
echo "RGB_MEAN                 = ${RGB_MEAN[*]}"
echo "RGB_STD                  = ${RGB_STD[*]}"
echo "WEIGHT_SAVE_PATH         = ${WEIGHT_SAVE_PATH}"
echo "OPTIMIZER                = ${OPTIMIZER}"
echo "======================================"
printf '[Command] '; printf '%q ' "${TRAIN_CMD[@]}"; printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DryRun] Command construction succeeded; training was not launched."
    exit 0
fi

"${PYTHON_BIN}" -c "import sys; print(sys.executable); import utils_; import loss; import backbone; print('imports ok')"
"${TRAIN_CMD[@]}"