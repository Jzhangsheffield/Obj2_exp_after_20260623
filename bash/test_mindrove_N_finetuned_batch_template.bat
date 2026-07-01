@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Generic batch test script for fine-tuned N_as_test MindRove checkpoints
REM
REM What it does:
REM   - Recursively scans EMG and IMU fine-tuned weight roots.
REM   - Matches selected checkpoint file names exactly:
REM       best_val.pth / best_val_balanced.pth / best_val_macro_f1.pth / last.pth
REM   - Tests EMG and IMU separately, using the same signal-specific
REM     target_len and mean/std as fine-tuning.
REM   - Splits weights into batches to avoid Windows command-line length limits.
REM   - Exposes all practical test-time parameters as variables below.
REM
REM Notes:
REM   - TEST_MANIFEST can be absolute, or relative to DATASET_ROOT if your Python
REM     loader supports that pattern.
REM   - Per-sample CSV and metrics JSON are written by the Python script next to
REM     each tested weight.
REM ============================================================


REM ============================================================
REM 1) Top-level task / path config
REM ============================================================
set "PROJECT_ROOT=D:\junxi_data\experiments_after_260623"
set "PY_SCRIPT=%PROJECT_ROOT%\ft_and_test\train_mapstyle_finetune_and_test.py"
set "PYTHON_BIN=python"

set "DATASET_ROOT=C:\MyFolder\mes19jz\Final_Mapstyle_Dataset"

REM Choose one preset: take_put / except_take_put / custom
REM If TASK_PRESET=custom, edit LABEL_MAP_JSON, NUM_CLASSES, TEST_MANIFEST,
REM and the CUSTOM_* normalization values in section 7.
set "TASK_PRESET=take_put"

REM These are filled by :apply_task_preset unless TASK_PRESET=custom.
set "LABEL_MAP_JSON=C:\MyFolder\mes19jz\Final_Mapstyle_Dataset\label_map_take_put.json"
set "NUM_CLASSES=2"
set "TEST_MANIFEST=C:\MyFolder\mes19jz\Final_Mapstyle_Dataset\N_as_test\test_manifest_take_put.jsonl"

REM Optional manual overrides after preset is applied.
REM Leave empty to use preset values.
set "LABEL_MAP_JSON_OVERRIDE="
set "NUM_CLASSES_OVERRIDE="
set "TEST_MANIFEST_OVERRIDE="

REM Weight parent root. Change only this if your output structure is:
REM   WEIGHT_PARENT_ROOT\signal_emg\...
REM   WEIGHT_PARENT_ROOT\signal_imu\...
set "WEIGHT_PARENT_ROOT=D:\junxi_data\experiments_after_260623\results\ft_mindrove_N_take_put_adamw_44_seed1\weights"

REM Optional signal root overrides. Leave empty to use WEIGHT_PARENT_ROOT\signal_emg / signal_imu.
set "EMG_WEIGHT_ROOT="
set "IMU_WEIGHT_ROOT="

REM Output root for summary CSV and console logs.
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_STAMP=%%I"
if not defined RUN_STAMP set "RUN_STAMP=manual_tag"
set "OUTPUT_BASE=%WEIGHT_PARENT_ROOT%\_batch_test\%TASK_PRESET%_%RUN_STAMP%"
set "LOG_ROOT=%OUTPUT_BASE%\logs"
set "SUMMARY_ROOT=%OUTPUT_BASE%\summary"


REM ============================================================
REM 2) Checkpoint selection and batching
REM ============================================================
set "MATCH_BEST_VAL=0"
set "MATCH_BEST_VAL_BALANCED=0"
set "MATCH_BEST_VAL_MACRO_F1=0"
set "MATCH_LAST=1"

REM Maximum number of weights passed to one Python invocation.
REM Increase if paths are short; decrease if cmd reports line too long.
set "MAX_WEIGHTS_PER_RUN=20"

REM 1: print commands but do not run Python.
set "DRY_RUN=0"

REM 1: delete existing signal summary CSV files under this OUTPUT_BASE before running.
REM OUTPUT_BASE includes timestamp by default, so this is mostly useful if you set OUTPUT_BASE manually.
set "CLEAR_PREVIOUS_RESULTS=1"

REM 1: fail if a signal root contains no selected checkpoints.
REM 0: skip that signal.
set "FAIL_IF_NO_WEIGHTS=1"


REM ============================================================
REM 3) Test dataloader / runtime parameters
REM ============================================================
set "FIXED_SEED=1"
set "BATCH_SIZE=64"
set "NUM_WORKERS_TEST=8"
set "PREFETCH_FACTOR_TEST=2"
set "ENABLE_AMP_ARG="
REM Example:
REM set "ENABLE_AMP_ARG=--enable_amp"


REM ============================================================
REM 4) Dataset / label / modality parameters
REM ============================================================
set "USE_MODALITY=mindrove"
set "TIER_MODE=tier1"
set "N_FRAMES=16"


REM ============================================================
REM 5) MindRove signal/data parameters
REM ============================================================
set "MINDROVE_HANDS=left right"
set "MINDROVE_MERGE_HANDS_ARG="
set "MINDROVE_APPLY_NORMALIZATION_ARG=--mindrove_apply_normalization"
set "MINDROVE_APPLY_AUGMENTATION_ARG=--no-mindrove_apply_augmentation"

set "EMG_TARGET_LEN=512"
set "IMU_TARGET_LEN=128"

REM Explicit no-augmentation values for test-time reproducibility.
set "NO_AUG_ARGS=--mindrove_time_warp_prob 0.0 --mindrove_time_warp_sigma 0.2 --mindrove_time_warp_num_knots 3 --mindrove_time_warp_num_splines 4 --mindrove_emg_scaling_prob 0.0 --mindrove_emg_scaling_sigma 0.10 --mindrove_emg_noise_prob 0.0 --mindrove_emg_noise_sigma 0.05 --mindrove_emg_drift_prob 0.0 --mindrove_emg_drift_max 0.2 --mindrove_emg_drift_n_points 4 --mindrove_emg_drift_kind additive --mindrove_emg_drift_per_channel --mindrove_emg_drift_normalize --mindrove_emg_negate_prob 0.0 --mindrove_emg_channel_dropout_prob 0.0 --mindrove_emg_channel_dropout_max_channels 1 --mindrove_imu_scaling_prob 0.0 --mindrove_imu_scaling_sigma 0.05 --mindrove_imu_noise_prob 0.0 --mindrove_imu_noise_sigma 0.03 --mindrove_imu_drift_prob 0.0 --mindrove_imu_drift_max 0.2 --mindrove_imu_drift_n_points 4 --mindrove_imu_drift_kind additive --mindrove_imu_drift_per_channel --mindrove_imu_drift_normalize --mindrove_imu_negate_prob 0.0 --mindrove_imu_channel_dropout_prob 0.0 --mindrove_imu_channel_dropout_max_channels 1"


REM ============================================================
REM 6) Model parameters
REM ============================================================
set "MODEL_DEPTH=10"
set "L2_NORMALIZE_BEFORE_FC_ARG="
set "MINDROVE_ARCH=resnet10_1d"
set "MINDROVE_BASE_CHANNELS=64"
set "MINDROVE_STEM_KERNEL_SIZE=7"
set "MINDROVE_STEM_STRIDE=2"
set "MINDROVE_USE_STEM_POOL_ARG=--mindrove_use_stem_pool"
set "MINDROVE_ZERO_INIT_RESIDUAL_ARG=--no-mindrove_zero_init_residual"


REM ============================================================
REM 7) Normalization presets
REM ============================================================
REM ---------- take_put: from N_as_test\train_normalization_stats\take_put_mindrove_stats.json
set "TAKE_PUT_LEFT_EMG_MEAN=0.0013 -0.0019 0.0012 0.0008 0.0029 0.0029 -0.0004 -0.0002"
set "TAKE_PUT_LEFT_EMG_STD=15.2382 20.2947 22.3931 23.2527 23.0585 19.6927 10.2315 9.3464"
set "TAKE_PUT_RIGHT_EMG_MEAN=-0.0011 -0.0014 0.0036 0.0053 0.0019 0.0010 -0.0016 0.0006"
set "TAKE_PUT_RIGHT_EMG_STD=15.8905 24.9046 28.6934 26.8379 27.4055 19.6011 10.9768 10.0733"
set "TAKE_PUT_LEFT_IMU_MEAN=-0.0351 -0.0078 -0.0514 0.1487 -0.0561 4.4670"
set "TAKE_PUT_LEFT_IMU_STD=0.1947 0.1937 0.0959 44.1654 48.3155 56.6028"
set "TAKE_PUT_RIGHT_IMU_MEAN=0.0487 -0.0318 -0.0380 -2.9322 -2.7989 -11.2903"
set "TAKE_PUT_RIGHT_IMU_STD=0.2340 0.2879 0.1325 65.6138 76.7400 79.8795"

REM ---------- except_take_put: from N_as_test\train_normalization_stats\except_take_put_mindrove_stats.json
set "EXCEPT_LEFT_EMG_MEAN=0.0068 0.0205 0.0096 0.0238 0.0219 0.0168 -0.0024 -0.0058"
set "EXCEPT_LEFT_EMG_STD=30.1656 33.4225 38.0212 36.9723 36.7116 35.9175 22.1645 18.3178"
set "EXCEPT_RIGHT_EMG_MEAN=0.0102 -0.0020 -0.0492 0.0158 0.0006 -0.0251 -0.0112 -0.0135"
set "EXCEPT_RIGHT_EMG_STD=30.6975 41.2925 49.7859 49.2473 45.4501 35.0010 28.3983 22.1492"
set "EXCEPT_LEFT_IMU_MEAN=-0.0482 -0.0028 -0.0314 -0.6035 -0.6008 -2.7832"
set "EXCEPT_LEFT_IMU_STD=0.2216 0.1471 0.0910 33.6979 33.1239 34.2654"
set "EXCEPT_RIGHT_IMU_MEAN=-0.0130 0.0114 -0.0237 0.9942 -0.9100 3.8193"
set "EXCEPT_RIGHT_IMU_STD=0.2280 0.2044 0.1116 39.6820 41.1152 40.6422"

REM ---------- custom: edit these when TASK_PRESET=custom
set "CUSTOM_LEFT_EMG_MEAN="
set "CUSTOM_LEFT_EMG_STD="
set "CUSTOM_RIGHT_EMG_MEAN="
set "CUSTOM_RIGHT_EMG_STD="
set "CUSTOM_LEFT_IMU_MEAN="
set "CUSTOM_LEFT_IMU_STD="
set "CUSTOM_RIGHT_IMU_MEAN="
set "CUSTOM_RIGHT_IMU_STD="


REM ============================================================
REM 8) Optional RGB parameters exposed for argparse validation
REM    They are not used when USE_MODALITY=mindrove, but the Python script still validates them.
REM ============================================================
set "RRC_SCALE_MIN=0.2"
set "RRC_SCALE_MAX=1.0"
set "RRC_RATIO_MIN=0.75"
set "RRC_RATIO_MAX=1.3333333333"
set "RGB_HFLIP_P=0.5"
set "RGB_VFLIP_P=0.0"
set "RGB_JITTER_P=0.5"
set "RGB_JITTER_BRIGHTNESS=0.4"
set "RGB_JITTER_CONTRAST=0.4"
set "RGB_JITTER_SATURATION=0.4"
set "RGB_JITTER_HUE=0.1"
set "RGB_GRAY_P=0.2"
set "RGB_BLUR_P=0.5"
set "RGB_BLUR_KERNEL=7"
set "RGB_BLUR_SIGMA_MIN=0.1"
set "RGB_BLUR_SIGMA_MAX=1.0"


REM ============================================================
REM 9) Validate and prepare
REM ============================================================
call :apply_task_preset
if errorlevel 1 exit /b 1

if not "%LABEL_MAP_JSON_OVERRIDE%"=="" set "LABEL_MAP_JSON=%LABEL_MAP_JSON_OVERRIDE%"
if not "%NUM_CLASSES_OVERRIDE%"=="" set "NUM_CLASSES=%NUM_CLASSES_OVERRIDE%"
if not "%TEST_MANIFEST_OVERRIDE%"=="" set "TEST_MANIFEST=%TEST_MANIFEST_OVERRIDE%"

if "%EMG_WEIGHT_ROOT%"=="" set "EMG_WEIGHT_ROOT=%WEIGHT_PARENT_ROOT%\signal_emg"
if "%IMU_WEIGHT_ROOT%"=="" set "IMU_WEIGHT_ROOT=%WEIGHT_PARENT_ROOT%\signal_imu"

if "%PROJECT_ROOT%"=="" (
  echo [Error] PROJECT_ROOT is empty.
  exit /b 1
)
if "%PY_SCRIPT%"=="" (
  echo [Error] PY_SCRIPT is empty.
  exit /b 1
)
if "%DATASET_ROOT%"=="" (
  echo [Error] DATASET_ROOT is empty.
  exit /b 1
)
if "%LABEL_MAP_JSON%"=="" (
  echo [Error] LABEL_MAP_JSON is empty.
  exit /b 1
)
if "%NUM_CLASSES%"=="" (
  echo [Error] NUM_CLASSES is empty.
  exit /b 1
)
if "%TEST_MANIFEST%"=="" (
  echo [Error] TEST_MANIFEST is empty.
  exit /b 1
)

cd /d "%PROJECT_ROOT%"
if errorlevel 1 (
  echo [Error] Failed to enter PROJECT_ROOT:
  echo   %PROJECT_ROOT%
  exit /b 1
)
set "PYTHONPATH=%PROJECT_ROOT%"

if not exist "%OUTPUT_BASE%" mkdir "%OUTPUT_BASE%"
if not exist "%LOG_ROOT%" mkdir "%LOG_ROOT%"
if not exist "%SUMMARY_ROOT%" mkdir "%SUMMARY_ROOT%"

if /I "%CLEAR_PREVIOUS_RESULTS%"=="1" (
  if exist "%SUMMARY_ROOT%\signal_emg_test_results.csv" del /q "%SUMMARY_ROOT%\signal_emg_test_results.csv" >nul 2>nul
  if exist "%SUMMARY_ROOT%\signal_imu_test_results.csv" del /q "%SUMMARY_ROOT%\signal_imu_test_results.csv" >nul 2>nul
)

echo ============================================================
echo MindRove batch test
echo Project:       %PROJECT_ROOT%
echo Script:        %PY_SCRIPT%
echo Dataset:       %DATASET_ROOT%
echo Task preset:   %TASK_PRESET%
echo Label map:     %LABEL_MAP_JSON%
echo Test manifest: %TEST_MANIFEST%
echo Num classes:   %NUM_CLASSES%
echo EMG root:      %EMG_WEIGHT_ROOT%
echo IMU root:      %IMU_WEIGHT_ROOT%
echo Output base:   %OUTPUT_BASE%
echo Dry run:       %DRY_RUN%
echo ============================================================

call :run_signal_test_block "emg" "%EMG_WEIGHT_ROOT%"
if errorlevel 1 exit /b 1

call :run_signal_test_block "imu" "%IMU_WEIGHT_ROOT%"
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo All selected MindRove checkpoint tests finished successfully.
echo Output root:
echo   %OUTPUT_BASE%
echo ============================================================
pause
exit /b 0


REM ============================================================
REM Subroutine: apply task preset
REM ============================================================
:apply_task_preset
if /I "%TASK_PRESET%"=="take_put" (
  set "LABEL_MAP_JSON=%DATASET_ROOT%\label_map_take_put.json"
  set "NUM_CLASSES=2"
  set "TEST_MANIFEST=N_as_test\test_manifest_take_put.jsonl"
  set "LEFT_EMG_MEAN=%TAKE_PUT_LEFT_EMG_MEAN%"
  set "LEFT_EMG_STD=%TAKE_PUT_LEFT_EMG_STD%"
  set "RIGHT_EMG_MEAN=%TAKE_PUT_RIGHT_EMG_MEAN%"
  set "RIGHT_EMG_STD=%TAKE_PUT_RIGHT_EMG_STD%"
  set "LEFT_IMU_MEAN=%TAKE_PUT_LEFT_IMU_MEAN%"
  set "LEFT_IMU_STD=%TAKE_PUT_LEFT_IMU_STD%"
  set "RIGHT_IMU_MEAN=%TAKE_PUT_RIGHT_IMU_MEAN%"
  set "RIGHT_IMU_STD=%TAKE_PUT_RIGHT_IMU_STD%"
  exit /b 0
)

if /I "%TASK_PRESET%"=="except_take_put" (
  set "LABEL_MAP_JSON=%DATASET_ROOT%\label_map_except_take_put.json"
  set "NUM_CLASSES=15"
  set "TEST_MANIFEST=N_as_test\test_manifest_except_take_put.jsonl"
  set "LEFT_EMG_MEAN=%EXCEPT_LEFT_EMG_MEAN%"
  set "LEFT_EMG_STD=%EXCEPT_LEFT_EMG_STD%"
  set "RIGHT_EMG_MEAN=%EXCEPT_RIGHT_EMG_MEAN%"
  set "RIGHT_EMG_STD=%EXCEPT_RIGHT_EMG_STD%"
  set "LEFT_IMU_MEAN=%EXCEPT_LEFT_IMU_MEAN%"
  set "LEFT_IMU_STD=%EXCEPT_LEFT_IMU_STD%"
  set "RIGHT_IMU_MEAN=%EXCEPT_RIGHT_IMU_MEAN%"
  set "RIGHT_IMU_STD=%EXCEPT_RIGHT_IMU_STD%"
  exit /b 0
)

if /I "%TASK_PRESET%"=="custom" (
  set "LEFT_EMG_MEAN=%CUSTOM_LEFT_EMG_MEAN%"
  set "LEFT_EMG_STD=%CUSTOM_LEFT_EMG_STD%"
  set "RIGHT_EMG_MEAN=%CUSTOM_RIGHT_EMG_MEAN%"
  set "RIGHT_EMG_STD=%CUSTOM_RIGHT_EMG_STD%"
  set "LEFT_IMU_MEAN=%CUSTOM_LEFT_IMU_MEAN%"
  set "LEFT_IMU_STD=%CUSTOM_LEFT_IMU_STD%"
  set "RIGHT_IMU_MEAN=%CUSTOM_RIGHT_IMU_MEAN%"
  set "RIGHT_IMU_STD=%CUSTOM_RIGHT_IMU_STD%"
  exit /b 0
)

echo [Error] Unsupported TASK_PRESET=%TASK_PRESET%
echo         Use take_put, except_take_put, or custom.
exit /b 1


REM ============================================================
REM Subroutine: set signal-specific args
REM Args:
REM   %~1 = emg / imu
REM ============================================================
:set_signal_cfg
set "SIGNAL_TAG=%~1"
if /I "!SIGNAL_TAG!"=="emg" (
  set "SIGNAL_ARG=--mindrove_signals emg"
  set "TARGET_LEN_ARG=--mindrove_target_len %EMG_TARGET_LEN%"
  set "NORM_ARGS=--mindrove_left_emg_mean %LEFT_EMG_MEAN% --mindrove_left_emg_std %LEFT_EMG_STD% --mindrove_right_emg_mean %RIGHT_EMG_MEAN% --mindrove_right_emg_std %RIGHT_EMG_STD%"
  set "SIGNAL_SAVE_PATH=%SUMMARY_ROOT%\signal_emg"
  set "RESULTS_CSV=%SUMMARY_ROOT%\signal_emg_test_results.csv"
  exit /b 0
)

if /I "!SIGNAL_TAG!"=="imu" (
  set "SIGNAL_ARG=--mindrove_signals imu"
  set "TARGET_LEN_ARG=--mindrove_target_len %IMU_TARGET_LEN%"
  set "NORM_ARGS=--mindrove_left_imu_mean %LEFT_IMU_MEAN% --mindrove_left_imu_std %LEFT_IMU_STD% --mindrove_right_imu_mean %RIGHT_IMU_MEAN% --mindrove_right_imu_std %RIGHT_IMU_STD%"
  set "SIGNAL_SAVE_PATH=%SUMMARY_ROOT%\signal_imu"
  set "RESULTS_CSV=%SUMMARY_ROOT%\signal_imu_test_results.csv"
  exit /b 0
)

echo [Error] Unsupported signal: !SIGNAL_TAG!
exit /b 1


REM ============================================================
REM Subroutine: run one signal test block
REM Args:
REM   %~1 = emg / imu
REM   %~2 = fine-tuned weight root
REM ============================================================
:run_signal_test_block
set "SIGNAL_TAG=%~1"
set "WEIGHT_ROOT=%~2"

call :set_signal_cfg "!SIGNAL_TAG!"
if errorlevel 1 exit /b 1

if not exist "!WEIGHT_ROOT!" (
  echo [Error] WEIGHT_ROOT does not exist for signal=!SIGNAL_TAG!:
  echo   !WEIGHT_ROOT!
  exit /b 1
)

if not exist "!SIGNAL_SAVE_PATH!" mkdir "!SIGNAL_SAVE_PATH!"

set /a NUM_WEIGHTS=0
set /a PART_ID=1
set /a PART_COUNT=0
set "PART_WEIGHTS="

set "LIST_FILE=!SIGNAL_SAVE_PATH!\selected_!SIGNAL_TAG!_weights.txt"
if exist "!LIST_FILE!" del /q "!LIST_FILE!" >nul 2>nul

echo.
echo ============================================================
echo [Scan] signal=!SIGNAL_TAG!
echo [Root] !WEIGHT_ROOT!
echo [Checkpoint switches]
echo   best_val.pth          MATCH_BEST_VAL=%MATCH_BEST_VAL%
echo   best_val_balanced.pth MATCH_BEST_VAL_BALANCED=%MATCH_BEST_VAL_BALANCED%
echo   best_val_macro_f1.pth MATCH_BEST_VAL_MACRO_F1=%MATCH_BEST_VAL_MACRO_F1%
echo   last.pth              MATCH_LAST=%MATCH_LAST%
echo ============================================================

if /I "%MATCH_BEST_VAL%"=="1" call :collect_checkpoint_name "!WEIGHT_ROOT!" "best_val.pth"
if errorlevel 1 exit /b 1
if /I "%MATCH_BEST_VAL_BALANCED%"=="1" call :collect_checkpoint_name "!WEIGHT_ROOT!" "best_val_balanced.pth"
if errorlevel 1 exit /b 1
if /I "%MATCH_BEST_VAL_MACRO_F1%"=="1" call :collect_checkpoint_name "!WEIGHT_ROOT!" "best_val_macro_f1.pth"
if errorlevel 1 exit /b 1
if /I "%MATCH_LAST%"=="1" call :collect_checkpoint_name "!WEIGHT_ROOT!" "last.pth"
if errorlevel 1 exit /b 1

if defined PART_WEIGHTS (
  call :run_python_part "!SIGNAL_TAG!" "!PART_ID!" "!PART_COUNT!"
  if errorlevel 1 exit /b 1
)

if !NUM_WEIGHTS! EQU 0 (
  if /I "%FAIL_IF_NO_WEIGHTS%"=="1" (
    echo [Error] No selected checkpoints found under:
    echo   !WEIGHT_ROOT!
    exit /b 1
  ) else (
    echo [Skip] No selected checkpoints found for signal=!SIGNAL_TAG!
    exit /b 0
  )
)

echo.
echo Finished signal=!SIGNAL_TAG!
echo Selected weights: !NUM_WEIGHTS!
echo List file:
echo   !LIST_FILE!
echo Summary csv:
echo   !RESULTS_CSV!
exit /b 0


REM ============================================================
REM Subroutine: collect exact checkpoint file name recursively
REM Args:
REM   %~1 = root
REM   %~2 = exact checkpoint file name
REM ============================================================
:collect_checkpoint_name
set "SCAN_ROOT=%~1"
set "CKPT_NAME=%~2"

for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "Get-ChildItem -LiteralPath '!SCAN_ROOT!' -Recurse -Filter '!CKPT_NAME!' -File ^| Sort-Object FullName ^| ForEach-Object { $_.FullName }"`) do (
  if exist "%%~fF" (
    set /a NUM_WEIGHTS+=1
    set /a PART_COUNT+=1
    echo %%~fF>>"!LIST_FILE!"
    echo [FOUND][!SIGNAL_TAG!][!NUM_WEIGHTS!] %%~fF
    if defined PART_WEIGHTS (
      set "PART_WEIGHTS=!PART_WEIGHTS! "%%~fF""
    ) else (
      set "PART_WEIGHTS="%%~fF""
    )
    if !PART_COUNT! GEQ %MAX_WEIGHTS_PER_RUN% (
      call :run_python_part "!SIGNAL_TAG!" "!PART_ID!" "!PART_COUNT!"
      if errorlevel 1 exit /b 1
      set /a PART_ID+=1
      set /a PART_COUNT=0
      set "PART_WEIGHTS="
    )
  )
)
exit /b 0


REM ============================================================
REM Subroutine: run one Python test part
REM Args:
REM   %~1 = signal
REM   %~2 = part id
REM   %~3 = number of weights in this part
REM Uses:
REM   PART_WEIGHTS = quoted weight args string
REM ============================================================
:run_python_part
set "RUN_SIGNAL=%~1"
set "RUN_PART_ID=%~2"
set "RUN_PART_COUNT=%~3"
set "RUN_WEIGHTS=!PART_WEIGHTS!"

set "CONSOLE_LOG=%LOG_ROOT%\!RUN_SIGNAL!_test_part_!RUN_PART_ID!.log"

echo.
echo ============================================================
echo [Test part]
echo   Signal        : !RUN_SIGNAL!
echo   Part          : !RUN_PART_ID!
echo   Num weights   : !RUN_PART_COUNT!
echo   Test manifest : %TEST_MANIFEST%
echo   Summary csv   : !RESULTS_CSV!
echo   Console log   : !CONSOLE_LOG!
echo ============================================================

if /I "%DRY_RUN%"=="1" (
  echo [DryRun] "%PYTHON_BIN%" "%PY_SCRIPT%" --run_mode test --test_weight_paths !RUN_WEIGHTS!
  exit /b 0
)

"%PYTHON_BIN%" "%PY_SCRIPT%" ^
  --run_mode test ^
  --dataset_root "%DATASET_ROOT%" ^
  --label_map_json "%LABEL_MAP_JSON%" ^
  --test_manifest "%TEST_MANIFEST%" ^
  --test_weight_paths !RUN_WEIGHTS! ^
  --test_results_csv "!RESULTS_CSV!" ^
  --save_path "!SIGNAL_SAVE_PATH!" ^
  --use_modality %USE_MODALITY% ^
  --tier_mode %TIER_MODE% ^
  --num_classes %NUM_CLASSES% ^
  --n_frames %N_FRAMES% ^
  --batch_size %BATCH_SIZE% ^
  --seed %FIXED_SEED% ^
  --num_workers_test %NUM_WORKERS_TEST% ^
  --prefetch_factor_test %PREFETCH_FACTOR_TEST% ^
  --model_depth %MODEL_DEPTH% ^
  %L2_NORMALIZE_BEFORE_FC_ARG% ^
  --mindrove_hands %MINDROVE_HANDS% ^
  !SIGNAL_ARG! ^
  !TARGET_LEN_ARG! ^
  %MINDROVE_MERGE_HANDS_ARG% ^
  %MINDROVE_APPLY_NORMALIZATION_ARG% ^
  %MINDROVE_APPLY_AUGMENTATION_ARG% ^
  !NORM_ARGS! ^
  %NO_AUG_ARGS% ^
  --mindrove_arch %MINDROVE_ARCH% ^
  --mindrove_base_channels %MINDROVE_BASE_CHANNELS% ^
  --mindrove_stem_kernel_size %MINDROVE_STEM_KERNEL_SIZE% ^
  --mindrove_stem_stride %MINDROVE_STEM_STRIDE% ^
  %MINDROVE_USE_STEM_POOL_ARG% ^
  %MINDROVE_ZERO_INIT_RESIDUAL_ARG% ^
  --rrc_scale_min %RRC_SCALE_MIN% ^
  --rrc_scale_max %RRC_SCALE_MAX% ^
  --rrc_ratio_min %RRC_RATIO_MIN% ^
  --rrc_ratio_max %RRC_RATIO_MAX% ^
  --rgb_hflip_p %RGB_HFLIP_P% ^
  --rgb_vflip_p %RGB_VFLIP_P% ^
  --rgb_jitter_p %RGB_JITTER_P% ^
  --rgb_jitter_brightness %RGB_JITTER_BRIGHTNESS% ^
  --rgb_jitter_contrast %RGB_JITTER_CONTRAST% ^
  --rgb_jitter_saturation %RGB_JITTER_SATURATION% ^
  --rgb_jitter_hue %RGB_JITTER_HUE% ^
  --rgb_gray_p %RGB_GRAY_P% ^
  --rgb_blur_p %RGB_BLUR_P% ^
  --rgb_blur_kernel %RGB_BLUR_KERNEL% ^
  --rgb_blur_sigma_min %RGB_BLUR_SIGMA_MIN% ^
  --rgb_blur_sigma_max %RGB_BLUR_SIGMA_MAX% ^
  %ENABLE_AMP_ARG% ^
  > "!CONSOLE_LOG!" 2^>^&1

if errorlevel 1 (
  echo [Error] Test failed for signal=!RUN_SIGNAL! part=!RUN_PART_ID!
  echo See console log:
  echo   !CONSOLE_LOG!
  exit /b 1
)

echo [Done] signal=!RUN_SIGNAL! part=!RUN_PART_ID!
exit /b 0
