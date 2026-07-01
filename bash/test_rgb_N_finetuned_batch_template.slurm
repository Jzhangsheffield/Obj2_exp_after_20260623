#!/bin/bash
#SBATCH --job-name=test_rgb_N_ft
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-user=jzhang319@sheffield.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/ft_and_test/out/test_rgb_N_ft_%j.out
#SBATCH --error=/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623/ft_and_test/out/test_rgb_N_ft_%j.err

set -euo pipefail

# ============================================================
# Generic batch test script for fine-tuned N_as_test RGB checkpoints
#
# What it does:
#   - Recursively scans one RGB fine-tuned weight root.
#   - Matches selected checkpoint file names exactly:
#       best_val.pth / best_val_balanced.pth / best_val_macro_f1.pth / last.pth
#   - Uses the same RGB mean/std and model parameters as fine-tuning.
#   - Splits weights into batches to avoid command-line length limits.
#   - Exposes all practical test-time parameters as variables below.
#
# Notes:
#   - TEST_MANIFEST can be absolute, or relative to DATASET_ROOT if your Python
#     loader supports that pattern.
#   - Per-sample CSV and metrics JSON are written by the Python script next to
#     each tested weight.
# ============================================================

echo "============================================================"
echo "Job started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-NA}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
echo "============================================================"


# ============================================================
# 1) Top-level task / path config
# ============================================================
PROJECT_ROOT="/mnt/parscratch/users/mes19jz/objective2/thermal_crimp/experiments_after_260623"
PY_SCRIPT="${PROJECT_ROOT}/ft_and_test/train_mapstyle_finetune_and_test.py"
PYTHON_BIN="python"

DATASET_ROOT="/mnt/parscratch/users/mes19jz/datasets/thermal_crimp/Final_Mapstyle_Dataset"

# Choose one preset: take_put / except_take_put / custom
# If TASK_PRESET=custom, edit LABEL_MAP_JSON, NUM_CLASSES, TEST_MANIFEST,
# and CUSTOM_RGB_MEAN / CUSTOM_RGB_STD in section 7.
TASK_PRESET="take_put"

# These are filled by apply_task_preset unless TASK_PRESET=custom.
LABEL_MAP_JSON=""
NUM_CLASSES=""
TEST_MANIFEST=""
RGB_MEAN=()
RGB_STD=()

# Optional manual overrides after preset is applied. Leave empty to use preset values.
LABEL_MAP_JSON_OVERRIDE=""
NUM_CLASSES_OVERRIDE=""
TEST_MANIFEST_OVERRIDE=""
RGB_MEAN_OVERRIDE=()
RGB_STD_OVERRIDE=()

# Fine-tuned RGB weight root. Change this to test another completed run.
WEIGHT_ROOT="${PROJECT_ROOT}/results/ft_rgb_N_take_put_adamw_22_seed1/weights"

# Output root for summary CSV, tested-weight lists, and console logs.
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${WEIGHT_ROOT}/_batch_test/${TASK_PRESET}_${RUN_STAMP}"
LOG_ROOT="${OUTPUT_BASE}/logs"
SUMMARY_ROOT="${OUTPUT_BASE}/summary"


# ============================================================
# 2) Checkpoint selection and batching
# ============================================================
MATCH_BEST_VAL=1
MATCH_BEST_VAL_BALANCED=1
MATCH_BEST_VAL_MACRO_F1=1
MATCH_LAST=0

# Maximum number of weights passed to one Python invocation.
# Increase if paths are short; decrease if shell reports argument list too long.
MAX_WEIGHTS_PER_RUN=50

# 1: print commands but do not run Python.
DRY_RUN="${DRY_RUN:-0}"

# 1: delete existing summary CSV under this OUTPUT_BASE before running.
# OUTPUT_BASE includes timestamp by default, so this is mostly useful if you set OUTPUT_BASE manually.
CLEAR_PREVIOUS_RESULTS=1

# 1: fail if WEIGHT_ROOT contains no selected checkpoints.
# 0: skip cleanly.
FAIL_IF_NO_WEIGHTS=1


# ============================================================
# 3) HPC environment
# ============================================================
ANACONDA_MODULE="Anaconda3/2022.05"
CUDNN_MODULE="cuDNN/8.9.2.26-CUDA-12.1.1"
CONDA_ENV_NAME="pytorch"
SKIP_ENV_SETUP=0


# ============================================================
# 4) Test dataloader / runtime parameters
# ============================================================
FIXED_SEED=1
BATCH_SIZE=64
NUM_WORKERS_TEST=8
PREFETCH_FACTOR_TEST=2
ENABLE_AMP=0


# ============================================================
# 5) Dataset / label / modality parameters
# ============================================================
USE_MODALITY="rgb"
TIER_MODE="tier1"
N_FRAMES=16


# ============================================================
# 6) Model and RGB preprocessing parameters
# ============================================================
MODEL_DEPTH=18
L2_NORMALIZE_BEFORE_FC=0

RGB_SIZE=224
DEPTH_SIZE=224

RGB_APPLY_SPATIAL_AUG=0
RRC_SCALE_MIN=0.6
RRC_SCALE_MAX=1.0
RRC_RATIO_MIN=0.75
RRC_RATIO_MAX=1.3333333333
RGB_HFLIP_P=0.0
RGB_VFLIP_P=0.0
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


# ============================================================
# 7) RGB normalization presets
# ============================================================
# ---------- take_put: from ft_rgb_N_take_put_adamw_22_seed1.slurm
TAKE_PUT_RGB_MEAN=(0.3725 0.3828 0.3921)
TAKE_PUT_RGB_STD=(0.2923 0.2715 0.2640)

# ---------- except_take_put: from ft_rgb_N_except_take_put_adamw_22_seed1.slurm
EXCEPT_RGB_MEAN=(0.3752 0.3864 0.3960)
EXCEPT_RGB_STD=(0.2934 0.2724 0.2644)

# ---------- custom: edit these when TASK_PRESET=custom
CUSTOM_RGB_MEAN=()
CUSTOM_RGB_STD=()


# ============================================================
# 8) Helper functions
# ============================================================
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

print_command() {
    printf '[Command] '
    printf '%q ' "$@"
    printf '\n'
}

apply_task_preset() {
    case "${TASK_PRESET}" in
        take_put)
            LABEL_MAP_JSON="${DATASET_ROOT}/label_map_take_put.json"
            NUM_CLASSES=2
            TEST_MANIFEST="${DATASET_ROOT}/N_as_test/test_manifest_take_put.jsonl"
            RGB_MEAN=("${TAKE_PUT_RGB_MEAN[@]}")
            RGB_STD=("${TAKE_PUT_RGB_STD[@]}")
            ;;
        except_take_put)
            LABEL_MAP_JSON="${DATASET_ROOT}/label_map_except_take_put.json"
            NUM_CLASSES=15
            TEST_MANIFEST="${DATASET_ROOT}/N_as_test/test_manifest_except_take_put.jsonl"
            RGB_MEAN=("${EXCEPT_RGB_MEAN[@]}")
            RGB_STD=("${EXCEPT_RGB_STD[@]}")
            ;;
        custom)
            RGB_MEAN=("${CUSTOM_RGB_MEAN[@]}")
            RGB_STD=("${CUSTOM_RGB_STD[@]}")
            ;;
        *)
            echo "[Error] Unsupported TASK_PRESET=${TASK_PRESET}. Use take_put, except_take_put, or custom." >&2
            exit 2
            ;;
    esac

    [[ -n "${LABEL_MAP_JSON_OVERRIDE}" ]] && LABEL_MAP_JSON="${LABEL_MAP_JSON_OVERRIDE}"
    [[ -n "${NUM_CLASSES_OVERRIDE}" ]] && NUM_CLASSES="${NUM_CLASSES_OVERRIDE}"
    [[ -n "${TEST_MANIFEST_OVERRIDE}" ]] && TEST_MANIFEST="${TEST_MANIFEST_OVERRIDE}"
    if [[ "${#RGB_MEAN_OVERRIDE[@]}" -gt 0 ]]; then
        RGB_MEAN=("${RGB_MEAN_OVERRIDE[@]}")
    fi
    if [[ "${#RGB_STD_OVERRIDE[@]}" -gt 0 ]]; then
        RGB_STD=("${RGB_STD_OVERRIDE[@]}")
    fi
}

validate_config() {
    if [[ -z "${PROJECT_ROOT}" || ! -d "${PROJECT_ROOT}" ]]; then
        echo "[Error] PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
        exit 1
    fi
    if [[ -z "${PY_SCRIPT}" || ! -f "${PY_SCRIPT}" ]]; then
        echo "[Error] PY_SCRIPT does not exist: ${PY_SCRIPT}" >&2
        exit 1
    fi
    if [[ -z "${DATASET_ROOT}" || ! -d "${DATASET_ROOT}" ]]; then
        echo "[Error] DATASET_ROOT does not exist: ${DATASET_ROOT}" >&2
        exit 1
    fi
    if [[ -z "${LABEL_MAP_JSON}" || ! -f "${LABEL_MAP_JSON}" ]]; then
        echo "[Error] LABEL_MAP_JSON does not exist: ${LABEL_MAP_JSON}" >&2
        exit 1
    fi
    if [[ -z "${TEST_MANIFEST}" || ! -f "${TEST_MANIFEST}" ]]; then
        echo "[Error] TEST_MANIFEST does not exist: ${TEST_MANIFEST}" >&2
        exit 1
    fi
    if [[ -z "${NUM_CLASSES}" ]]; then
        echo "[Error] NUM_CLASSES is empty." >&2
        exit 1
    fi
    if [[ "${#RGB_MEAN[@]}" -ne 3 || "${#RGB_STD[@]}" -ne 3 ]]; then
        echo "[Error] RGB_MEAN and RGB_STD must each contain exactly 3 values." >&2
        echo "        RGB_MEAN=${RGB_MEAN[*]:-EMPTY}" >&2
        echo "        RGB_STD=${RGB_STD[*]:-EMPTY}" >&2
        exit 1
    fi
    if [[ ! -d "${WEIGHT_ROOT}" ]]; then
        echo "[Error] WEIGHT_ROOT does not exist: ${WEIGHT_ROOT}" >&2
        exit 1
    fi
}

build_common_args() {
    COMMON_ARGS=(
        --run_mode test
        --dataset_root "${DATASET_ROOT}"
        --label_map_json "${LABEL_MAP_JSON}"
        --test_manifest "${TEST_MANIFEST}"
        --tier_mode "${TIER_MODE}"
        --num_classes "${NUM_CLASSES}"
        --use_modality "${USE_MODALITY}"
        --model_depth "${MODEL_DEPTH}"
        --n_frames "${N_FRAMES}"
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
        --batch_size "${BATCH_SIZE}"
        --num_workers_test "${NUM_WORKERS_TEST}"
        --prefetch_factor_test "${PREFETCH_FACTOR_TEST}"
        --seed "${FIXED_SEED}"
    )

    append_bool_optional_arg "${RGB_APPLY_SPATIAL_AUG}" "--rgb_apply_spatial_aug" "--no-rgb_apply_spatial_aug"
    append_store_true_arg "${L2_NORMALIZE_BEFORE_FC}" "--l2_normalize_before_fc"
    append_store_true_arg "${ENABLE_AMP}" "--enable_amp"
}

collect_one_name() {
    local ckpt_name="$1"
    local tmp_file="$2"
    find "${WEIGHT_ROOT}" -type f -name "${ckpt_name}" | sort >> "${tmp_file}"
}

collect_weights() {
    local tmp_file="$1"
    : > "${tmp_file}"

    [[ "${MATCH_BEST_VAL}" == "1" ]] && collect_one_name "best_val.pth" "${tmp_file}"
    [[ "${MATCH_BEST_VAL_BALANCED}" == "1" ]] && collect_one_name "best_val_balanced.pth" "${tmp_file}"
    [[ "${MATCH_BEST_VAL_MACRO_F1}" == "1" ]] && collect_one_name "best_val_macro_f1.pth" "${tmp_file}"
    [[ "${MATCH_LAST}" == "1" ]] && collect_one_name "last.pth" "${tmp_file}"

    mapfile -t WEIGHTS < <(sort -u "${tmp_file}")
}

run_python_part() {
    local part_id="$1"
    shift
    local part_weights=("$@")
    local part_count="${#part_weights[@]}"
    local console_log="${LOG_ROOT}/rgb_test_part_${part_id}.log"

    echo ""
    echo "============================================================"
    echo "[Test part]"
    echo "  Part          : ${part_id}"
    echo "  Num weights   : ${part_count}"
    echo "  Test manifest : ${TEST_MANIFEST}"
    echo "  Summary csv   : ${TEST_RESULTS_CSV}"
    echo "  Console log   : ${console_log}"
    echo "============================================================"

    CMD=(
        "${PYTHON_BIN}" -u "${PY_SCRIPT}"
        "${COMMON_ARGS[@]}"
        --save_path "${SAVE_PATH}"
        --test_results_csv "${TEST_RESULTS_CSV}"
        --test_weight_paths "${part_weights[@]}"
    )

    print_command "${CMD[@]}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[DryRun] part=${part_id} command constructed; test skipped."
        return 0
    fi

    "${CMD[@]}" > "${console_log}" 2>&1
    echo "[Done] part=${part_id}"
}

run_batched_tests() {
    local part_id=1
    local start=0
    local total="${#WEIGHTS[@]}"

    while [[ "${start}" -lt "${total}" ]]; do
        local end=$((start + MAX_WEIGHTS_PER_RUN))
        if [[ "${end}" -gt "${total}" ]]; then
            end="${total}"
        fi

        local part_weights=("${WEIGHTS[@]:${start}:$((end - start))}")
        run_python_part "${part_id}" "${part_weights[@]}"

        start="${end}"
        part_id=$((part_id + 1))
    done
}


# ============================================================
# 9) Prepare environment and run
# ============================================================
apply_task_preset

mkdir -p "${LOG_ROOT}" "${SUMMARY_ROOT}" "${PROJECT_ROOT}/ft_and_test/out"
SAVE_PATH="${SUMMARY_ROOT}/rgb"
TEST_RESULTS_CSV="${SUMMARY_ROOT}/rgb_test_results.csv"
WEIGHT_LIST_TXT="${SUMMARY_ROOT}/selected_rgb_weights.txt"
TMP_WEIGHT_LIST="${SUMMARY_ROOT}/selected_rgb_weights.tmp"
mkdir -p "${SAVE_PATH}"

validate_config

if [[ "${SKIP_ENV_SETUP}" != "1" ]]; then
    module load "${ANACONDA_MODULE}"
    module load "${CUDNN_MODULE}"
    source activate "${CONDA_ENV_NAME}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

build_common_args

if [[ "${CLEAR_PREVIOUS_RESULTS}" == "1" && -f "${TEST_RESULTS_CSV}" ]]; then
    rm -f "${TEST_RESULTS_CSV}"
fi

collect_weights "${TMP_WEIGHT_LIST}"
printf "%s\n" "${WEIGHTS[@]}" > "${WEIGHT_LIST_TXT}"
rm -f "${TMP_WEIGHT_LIST}"

echo ""
echo "============================================================"
echo "RGB batch test config"
echo "Project root        : ${PROJECT_ROOT}"
echo "Python script       : ${PY_SCRIPT}"
echo "Dataset root        : ${DATASET_ROOT}"
echo "Task preset         : ${TASK_PRESET}"
echo "Label map           : ${LABEL_MAP_JSON}"
echo "Test manifest       : ${TEST_MANIFEST}"
echo "Num classes         : ${NUM_CLASSES}"
echo "Weight root         : ${WEIGHT_ROOT}"
echo "Output base         : ${OUTPUT_BASE}"
echo "RGB mean            : ${RGB_MEAN[*]}"
echo "RGB std             : ${RGB_STD[*]}"
echo "Selected weights    : ${#WEIGHTS[@]}"
echo "Max weights per run : ${MAX_WEIGHTS_PER_RUN}"
echo "Dry run             : ${DRY_RUN}"
echo "============================================================"

if [[ "${#WEIGHTS[@]}" -eq 0 ]]; then
    if [[ "${FAIL_IF_NO_WEIGHTS}" == "1" ]]; then
        echo "[Error] No selected checkpoints found under: ${WEIGHT_ROOT}" >&2
        exit 1
    fi
    echo "[Skip] No selected checkpoints found under: ${WEIGHT_ROOT}"
    exit 0
fi

echo "Selected checkpoint paths:"
printf '  %s\n' "${WEIGHTS[@]}"
echo "Weight list saved to: ${WEIGHT_LIST_TXT}"

if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" -c "import sys; print(sys.executable); import utils_; import loss; import backbone; print('imports ok')"
fi

run_batched_tests

echo ""
echo "============================================================"
echo "All selected RGB checkpoint tests finished at: $(date)"
echo "Summary CSV:"
echo "  ${TEST_RESULTS_CSV}"
echo "Weight list:"
echo "  ${WEIGHT_LIST_TXT}"
echo "Output base:"
echo "  ${OUTPUT_BASE}"
echo "============================================================"
