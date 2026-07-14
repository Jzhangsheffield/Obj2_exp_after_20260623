#!/bin/bash
#SBATCH --job-name=ft_rgb_N_stage5_rand3_seed1
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=10:00:00
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/ft_and_test/out/ft_rgb_N_except_take_put_adamw_stage5_random_3_seed1_%j.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/ft_and_test/out/ft_rgb_N_except_take_put_adamw_stage5_random_3_seed1_%j.err

set -euo pipefail

# ============================================================
# RGB finetuning for rgb_N_except_take_put_adamw_stage5_random_3 contrastive weights: N_as_test / train_manifest_except_take_put.jsonl
# - seed=1 only
# - scratch baseline: full fine-tuning from random init
# - pretrained: recursively find every checkpoint_0200.pth under PRETRAIN_PARENT
# - pretrained runs: head_only and full fine-tuning
# ============================================================

PROJECT_ROOT="/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623"
SCRIPT_PATH="${PROJECT_ROOT}/ft_and_test/train_mapstyle_finetune_and_test.py"
DATASET_ROOT="/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset"

TRAIN_MANIFEST="${DATASET_ROOT}/N_as_test/train_manifest_except_take_put.jsonl"
VAL_MANIFEST="${DATASET_ROOT}/N_as_test/val_manifest_except_take_put.jsonl"
LABEL_MAP_JSON="${DATASET_ROOT}/label_map_except_take_put.json"

PRETRAIN_PARENT="${PROJECT_ROOT}/results/rgb_N_except_take_put_adamw_stage5_random_3"
CHECKPOINT_NAME="checkpoint_0200.pth"
EXPECTED_NUM_CKPTS=3
ALLOW_CKPT_COUNT_MISMATCH=0

OUTPUT_ROOT="${PROJECT_ROOT}/results/ft_rgb_N_except_take_put_adamw_stage5_random_3_seed1"
SAVE_ROOT="${OUTPUT_ROOT}/weights"
DATAMAP_ROOT="${OUTPUT_ROOT}/datamaps"
CHECKPOINT_LIST_TXT="${OUTPUT_ROOT}/checkpoint_0200_paths.txt"

PYTHON_BIN="python"
ANACONDA_MODULE="Anaconda3/2022.05"
CUDNN_MODULE="cuDNN/8.9.2.26-CUDA-12.1.1"
CONDA_ENV_NAME="pytorch"
SKIP_ENV_SETUP=0
DRY_RUN=${DRY_RUN:-0}

USE_MODALITY="rgb"
TIER_MODE="tier1"
N_FRAMES=16
NUM_CLASSES=15
MODEL_DEPTH=18
L2_NORMALIZE_BEFORE_FC=0

RGB_SIZE=224
DEPTH_SIZE=224
RGB_MEAN=(0.3752 0.3864 0.3960)
RGB_STD=(0.2934 0.2724 0.2644)

RGB_APPLY_SPATIAL_AUG=1
RRC_SCALE_MIN=0.6
RRC_SCALE_MAX=1.0
RRC_RATIO_MIN=0.75
RRC_RATIO_MAX=1.3333333333
RGB_HFLIP_P=0.5
RGB_VFLIP_P=0.5
RGB_JITTER_P=0.0
RGB_JITTER_BRIGHTNESS=0.24
RGB_JITTER_CONTRAST=0.24
RGB_JITTER_SATURATION=0.24
RGB_JITTER_HUE=0.16
RGB_GRAY_P=0.0
RGB_BLUR_P=0.0
RGB_BLUR_KERNEL=5
RGB_BLUR_SIGMA_MIN=0.1
RGB_BLUR_SIGMA_MAX=1.0
DISABLE_TRAIN_AUGMENTATION=0

NUM_WORKERS_TRAIN=8
NUM_WORKERS_VAL=6
PREFETCH_FACTOR_TRAIN=2
PREFETCH_FACTOR_VAL=2
DISABLE_VAL=0

SEED=1
EPOCHS=100
BATCH_SIZE=64
LEARNING_RATE=1e-3
MOMENTUM=0.9
WEIGHT_DECAY=1e-4
OPTIMIZER="adamw"
ADAMW_BETA1=0.9
ADAMW_BETA2=0.999
ADAMW_EPS=1e-8
USE_COSINE_LR=0
SCHEDULES=(50 75)
ENABLE_AMP=0
SAVE_PERIOD=20
BEST_AFTER_EPOCH=0

HEAD_ONLY_HEAD_LR=1e-3
FULL_BACKBONE_LR=1e-4
FULL_HEAD_LR=1e-3
USE_DISCRIMINATIVE_LR_FOR_FULL=1

KEEP_PRETRAINED_HEAD=0
PRETRAINED_STRICT=0
PRETRAINED_TAG_MODE="relative_to_anchor"
PRETRAINED_TAG_LAST_K=2
PRETRAINED_TAG_ANCHOR="rgb_N_except_take_put_adamw_stage5_random_3"

USE_WEIGHTED_SAMPLER=0
SAMPLER_TIER="${TIER_MODE}"
SAMPLER_MODE="sqrt_inv"
USE_WEIGHTED_CE=0
WEIGHT_METHOD="class_balanced"
CB_BETA=0.999
WEIGHT_NORMALIZE_MEAN=0
USE_FOCAL=0
FOCAL_GAMMA=2.0
FOCAL_USE_ALPHA=0

mkdir -p "${SAVE_ROOT}" "${DATAMAP_ROOT}" "${OUTPUT_ROOT}" "${PROJECT_ROOT}/ft_and_test/out"

if [[ "${SKIP_ENV_SETUP}" != "1" ]]; then
    module load "${ANACONDA_MODULE}"
    module load "${CUDNN_MODULE}"
    source activate "${CONDA_ENV_NAME}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

append_bool_optional_arg() {
    local value="$1"
    local true_flag="$2"
    local false_flag="$3"
    case "${value}" in
        1|true|TRUE|yes|YES) COMMON_ARGS+=("${true_flag}") ;;
        0|false|FALSE|no|NO) COMMON_ARGS+=("${false_flag}") ;;
        "") ;;
        *) echo "[Error] Invalid boolean value for ${true_flag}/${false_flag}: ${value}" >&2; exit 2 ;;
    esac
}

append_store_true_arg() {
    local value="$1"
    local flag="$2"
    case "${value}" in
        1|true|TRUE|yes|YES) COMMON_ARGS+=("${flag}") ;;
        0|false|FALSE|no|NO|"") ;;
        *) echo "[Error] Invalid boolean value for ${flag}: ${value}" >&2; exit 2 ;;
    esac
}

maybe_full_lr_args() {
    if [[ "${USE_DISCRIMINATIVE_LR_FOR_FULL}" == "1" ]]; then
        FULL_LR_ARGS=(--use_discriminative_lr --backbone_learning_rate "${FULL_BACKBONE_LR}" --head_learning_rate "${FULL_HEAD_LR}")
    else
        FULL_LR_ARGS=(--backbone_learning_rate "${FULL_BACKBONE_LR}" --head_learning_rate "${FULL_HEAD_LR}")
    fi
}

if [[ ! -d "${PRETRAIN_PARENT}" ]]; then
    echo "[Error] PRETRAIN_PARENT does not exist: ${PRETRAIN_PARENT}" >&2
    exit 1
fi

mapfile -t CKPTS < <(find "${PRETRAIN_PARENT}" -type f -name "${CHECKPOINT_NAME}" | sort)
NUM_CKPTS=${#CKPTS[@]}
printf "%s\n" "${CKPTS[@]}" > "${CHECKPOINT_LIST_TXT}"

if [[ "${NUM_CKPTS}" -eq 0 ]]; then
    echo "[Error] No ${CHECKPOINT_NAME} files found under: ${PRETRAIN_PARENT}" >&2
    exit 1
fi
if [[ "${NUM_CKPTS}" -ne "${EXPECTED_NUM_CKPTS}" && "${ALLOW_CKPT_COUNT_MISMATCH}" != "1" ]]; then
    echo "[Error] Expected ${EXPECTED_NUM_CKPTS} ${CHECKPOINT_NAME} files, but found ${NUM_CKPTS}." >&2
    echo "        Check ${CHECKPOINT_LIST_TXT}, or set ALLOW_CKPT_COUNT_MISMATCH=1 intentionally." >&2
    exit 1
fi
for i in "${!CKPTS[@]}"; do
    printf "[%02d] %s\n" "$((i + 1))" "${CKPTS[$i]}"
done

COMMON_ARGS=(
    --run_mode train
    --dataset_root "${DATASET_ROOT}"
    --label_map_json "${LABEL_MAP_JSON}"
    --train_manifest "${TRAIN_MANIFEST}"
    --val_manifest "${VAL_MANIFEST}"
    --use_modality "${USE_MODALITY}"
    --tier_mode "${TIER_MODE}"
    --n_frames "${N_FRAMES}"
    --num_classes "${NUM_CLASSES}"
    --model_depth "${MODEL_DEPTH}"
    --rgb_size "${RGB_SIZE}"
    --depth_size "${DEPTH_SIZE}"
    --rgb_mean "${RGB_MEAN[@]}"
    --rgb_std "${RGB_STD[@]}"
    --rrc_scale_min "${RRC_SCALE_MIN}"
    --rrc_scale_max "${RRC_SCALE_MAX}"
    --rrc_ratio_min "${RRC_RATIO_MIN}"
    --rrc_ratio_max "${RRC_RATIO_MAX}"
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
    --rgb_blur_sigma_min "${RGB_BLUR_SIGMA_MIN}"
    --rgb_blur_sigma_max "${RGB_BLUR_SIGMA_MAX}"
    --num_workers_train "${NUM_WORKERS_TRAIN}"
    --num_workers_val "${NUM_WORKERS_VAL}"
    --prefetch_factor_train "${PREFETCH_FACTOR_TRAIN}"
    --prefetch_factor_val "${PREFETCH_FACTOR_VAL}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --learning_rate "${LEARNING_RATE}"
    --momentum "${MOMENTUM}"
    --weight_decay "${WEIGHT_DECAY}"
    --optimizer "${OPTIMIZER}"
    --adamw_beta1 "${ADAMW_BETA1}"
    --adamw_beta2 "${ADAMW_BETA2}"
    --adamw_eps "${ADAMW_EPS}"
    --schedules "${SCHEDULES[@]}"
    --seed "${SEED}"
    --pretrained_tag_mode "${PRETRAINED_TAG_MODE}"
    --pretrained_tag_last_k "${PRETRAINED_TAG_LAST_K}"
    --pretrained_tag_anchor "${PRETRAINED_TAG_ANCHOR}"
    --sampler_tier "${SAMPLER_TIER}"
    --sampler_mode "${SAMPLER_MODE}"
    --weight_method "${WEIGHT_METHOD}"
    --cb_beta "${CB_BETA}"
    --focal_gamma "${FOCAL_GAMMA}"
    --save_period "${SAVE_PERIOD}"
    --best_after_epoch "${BEST_AFTER_EPOCH}"
)

append_bool_optional_arg "${RGB_APPLY_SPATIAL_AUG}" "--rgb_apply_spatial_aug" "--no-rgb_apply_spatial_aug"
append_store_true_arg "${DISABLE_TRAIN_AUGMENTATION}" "--disable_train_augmentation"
append_store_true_arg "${DISABLE_VAL}" "--disable_val"
append_store_true_arg "${L2_NORMALIZE_BEFORE_FC}" "--l2_normalize_before_fc"
append_store_true_arg "${USE_COSINE_LR}" "--cos"
append_store_true_arg "${ENABLE_AMP}" "--enable_amp"
append_store_true_arg "${KEEP_PRETRAINED_HEAD}" "--keep_pretrained_head"
append_store_true_arg "${PRETRAINED_STRICT}" "--pretrained_strict"
append_store_true_arg "${USE_WEIGHTED_SAMPLER}" "--use_weighted_sampler"
append_store_true_arg "${USE_WEIGHTED_CE}" "--use_weighted_ce"
append_store_true_arg "${WEIGHT_NORMALIZE_MEAN}" "--weight_normalize_mean"
append_store_true_arg "${USE_FOCAL}" "--use_focal"
append_store_true_arg "${FOCAL_USE_ALPHA}" "--focal_use_alpha"

print_command() {
    printf '[Command] '
    printf '%q ' "$@"
    printf '\n'
}

run_train() {
    local label="$1"
    shift
    echo "======================================"
    echo "Start ${label}"
    echo "PROJECT_ROOT        = ${PROJECT_ROOT}"
    echo "SCRIPT_PATH         = ${SCRIPT_PATH}"
    echo "PRETRAIN_PARENT     = ${PRETRAIN_PARENT}"
    echo "CHECKPOINT_NAME     = ${CHECKPOINT_NAME}"
    echo "NUM_CKPTS           = ${NUM_CKPTS}"
    echo "TRAIN_MANIFEST      = ${TRAIN_MANIFEST}"
    echo "VAL_MANIFEST        = ${VAL_MANIFEST}"
    echo "LABEL_MAP_JSON      = ${LABEL_MAP_JSON}"
    echo "NUM_CLASSES         = ${NUM_CLASSES}"
    echo "RGB_MEAN            = ${RGB_MEAN[*]}"
    echo "RGB_STD             = ${RGB_STD[*]}"
    echo "SAVE_ROOT           = ${SAVE_ROOT}"
    echo "DATAMAP_ROOT        = ${DATAMAP_ROOT}"
    echo "CHECKPOINT_LIST_TXT = ${CHECKPOINT_LIST_TXT}"
    echo "======================================"

    CMD=("${PYTHON_BIN}" -u "${SCRIPT_PATH}" "$@")
    print_command "${CMD[@]}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DryRun] ${label} command constructed; training skipped."
        return 0
    fi
    "${CMD[@]}"
    echo "Finished ${label}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" -c "import sys; print(sys.executable); import utils_; import loss; import backbone; print('imports ok')"
fi

maybe_full_lr_args

run_train "scratch_full" \
    "${COMMON_ARGS[@]}" \
    --save_path "${SAVE_ROOT}/scratch_full" \
    --datamap_csv_path "${DATAMAP_ROOT}/scratch_full" \
    --finetune_mode full \
    "${FULL_LR_ARGS[@]}"

run_train "pretrained_head_only" \
    "${COMMON_ARGS[@]}" \
    --pretrained_weight_paths "${CKPTS[@]}" \
    --save_path "${SAVE_ROOT}/head_only" \
    --datamap_csv_path "${DATAMAP_ROOT}/head_only" \
    --finetune_mode head_only \
    --head_learning_rate "${HEAD_ONLY_HEAD_LR}"

run_train "pretrained_full" \
    "${COMMON_ARGS[@]}" \
    --pretrained_weight_paths "${CKPTS[@]}" \
    --save_path "${SAVE_ROOT}/full" \
    --datamap_csv_path "${DATAMAP_ROOT}/full" \
    --finetune_mode full \
    "${FULL_LR_ARGS[@]}"

echo "All finetuning experiments finished."
