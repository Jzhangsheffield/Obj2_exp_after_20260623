@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM Fine-tune MindRove EMG + IMU contrastive checkpoints for mindrove_N_except_take_put_adamw_random_kqueue_12
REM - seed=1 only
REM - scratch baseline: full training from random initialization
REM - pretrained: recursively find checkpoint_0200.pth under each signal pretrain root
REM - pretrained modes: head_only and full
REM - Windows cmd line length guard: pretrained weights are split into part_1 / part_2
REM ============================================================

REM ============================================================
REM 1) Paths to fill in
REM ============================================================
set "PROJECT_ROOT=D:\junxi_data\experiments_after_260623"
set "PY_SCRIPT=%PROJECT_ROOT%\ft_and_test\train_mapstyle_finetune_and_test.py"
set "DATASET_ROOT=C:\MyFolder\mes19jz\Final_Mapstyle_Dataset"
set "LABEL_MAP_JSON=%DATASET_ROOT%\label_map_except_take_put.json"
set "PRETRAIN_PARENT=%PROJECT_ROOT%\results\mindrove_N_except_take_put_adamw_random_kqueue_12"
set "OUTPUT_ROOT=%PROJECT_ROOT%\results\ft_mindrove_N_except_take_put_adamw_random_kqueue_12_seed1"

REM Optional: set these if your signal roots are not PRETRAIN_PARENT\signal_emg and PRETRAIN_PARENT\signal_imu.
set "EMG_PRETRAIN_ROOT="
set "IMU_PRETRAIN_ROOT="

REM Recommended examples:
REM set "PROJECT_ROOT=D:\Junxi_data\Objective2_thermal_crimp\thermal_crimp\experiments_after_260623"
REM set "PY_SCRIPT=%PROJECT_ROOT%\ft_and_test\train_mapstyle_finetune_and_test.py"
REM set "DATASET_ROOT=C:\Junxi_data_for_training_speedup\Final_Mapstyle_Dataset"
REM set "LABEL_MAP_JSON=%DATASET_ROOT%\label_map_except_take_put.json"
REM set "PRETRAIN_PARENT=%PROJECT_ROOT%\results\mindrove_N_except_take_put_adamw_random_kqueue_12"
REM set "OUTPUT_ROOT=%PROJECT_ROOT%\results\ft_mindrove_N_except_take_put_adamw_random_kqueue_12_seed1"

set "PYTHON_BIN=python"
set "CHECKPOINT_NAME=checkpoint_0200.pth"
set "EXPECTED_NUM_CKPTS_PER_SIGNAL=6"
set "ALLOW_CKPT_COUNT_MISMATCH=0"
set "DRY_RUN=0"

REM ============================================================
REM 2) Dataset / label config
REM ============================================================
set "TRAIN_MANIFEST=N_as_test\train_manifest_except_take_put.jsonl"
set "VAL_MANIFEST=N_as_test\val_manifest_except_take_put.jsonl"
set "TIER_MODE=tier1"
set "NUM_CLASSES=15"
set "USE_MODALITY=mindrove"

REM ============================================================
REM 3) MindRove signal config
REM ============================================================
set "MINDROVE_HANDS=left right"
set "MINDROVE_MERGE_HANDS_ARG="
set "MINDROVE_APPLY_NORMALIZATION_ARG=--mindrove_apply_normalization"
set "MINDROVE_APPLY_AUGMENTATION_ARG=--mindrove_apply_augmentation"
set "DISABLE_TRAIN_AUGMENTATION_ARG="
set "EMG_TARGET_LEN=512"
set "IMU_TARGET_LEN=128"

REM Normalization parameters from N_as_test\train_normalization_stats\except_take_put_mindrove_stats.json; whitespace-separated for train_mapstyle_finetune_and_test.py.
set "LEFT_EMG_MEAN=0.0068 0.0205 0.0096 0.0238 0.0219 0.0168 -0.0024 -0.0058"
set "LEFT_EMG_STD=30.1656 33.4225 38.0212 36.9723 36.7116 35.9175 22.1645 18.3178"
set "RIGHT_EMG_MEAN=0.0102 -0.0020 -0.0492 0.0158 0.0006 -0.0251 -0.0112 -0.0135"
set "RIGHT_EMG_STD=30.6975 41.2925 49.7859 49.2473 45.4501 35.0010 28.3983 22.1492"
set "LEFT_IMU_MEAN=-0.0482 -0.0028 -0.0314 -0.6035 -0.6008 -2.7832"
set "LEFT_IMU_STD=0.2216 0.1471 0.0910 33.6979 33.1239 34.2654"
set "RIGHT_IMU_MEAN=-0.0130 0.0114 -0.0237 0.9942 -0.9100 3.8193"
set "RIGHT_IMU_STD=0.2280 0.2044 0.1116 39.6820 41.1152 40.6422"

REM ============================================================
REM 4) Fine-tuning augmentation policy
REM ============================================================
set "EMG_AUG_POLICY=tw_scale_noise"
set "EMG_AUG_ARGS=--mindrove_time_warp_prob 0.5 --mindrove_time_warp_sigma 0.2 --mindrove_time_warp_num_knots 3 --mindrove_time_warp_num_splines 4 --mindrove_emg_scaling_prob 0.5 --mindrove_emg_scaling_sigma 0.10 --mindrove_emg_noise_prob 0.5 --mindrove_emg_noise_sigma 0.05 --mindrove_emg_drift_prob 0.0 --mindrove_emg_drift_max 0.2 --mindrove_emg_drift_n_points 4 --mindrove_emg_drift_kind additive --mindrove_emg_drift_per_channel --mindrove_emg_drift_normalize --mindrove_emg_negate_prob 0.0 --mindrove_emg_channel_dropout_prob 0.0 --mindrove_emg_channel_dropout_max_channels 1 --mindrove_imu_scaling_prob 0.0 --mindrove_imu_noise_prob 0.0 --mindrove_imu_drift_prob 0.0 --mindrove_imu_negate_prob 0.0 --mindrove_imu_channel_dropout_prob 0.0"

set "IMU_AUG_POLICY=tw_scale_noise"
set "IMU_AUG_ARGS=--mindrove_time_warp_prob 0.5 --mindrove_time_warp_sigma 0.2 --mindrove_time_warp_num_knots 3 --mindrove_time_warp_num_splines 4 --mindrove_emg_scaling_prob 0.0 --mindrove_emg_noise_prob 0.0 --mindrove_emg_drift_prob 0.0 --mindrove_emg_negate_prob 0.0 --mindrove_emg_channel_dropout_prob 0.0 --mindrove_imu_scaling_prob 0.5 --mindrove_imu_scaling_sigma 0.05 --mindrove_imu_noise_prob 0.5 --mindrove_imu_noise_sigma 0.03 --mindrove_imu_drift_prob 0.0 --mindrove_imu_drift_max 0.2 --mindrove_imu_drift_n_points 4 --mindrove_imu_drift_kind additive --mindrove_imu_drift_per_channel --mindrove_imu_drift_normalize --mindrove_imu_negate_prob 0.0 --mindrove_imu_channel_dropout_prob 0.0 --mindrove_imu_channel_dropout_max_channels 1"

REM ============================================================
REM 5) DataLoader parameters
REM ============================================================
set "BATCH_SIZE=64"
set "NUM_WORKERS_TRAIN=8"
set "NUM_WORKERS_VAL=6"
set "PREFETCH_FACTOR_TRAIN=2"
set "PREFETCH_FACTOR_VAL=2"
set "DISABLE_VAL_ARG="

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
REM 7) Optimizer / scheduler / checkpoint parameters
REM ============================================================
set "SEED=1"
set "EPOCHS=100"
set "LEARNING_RATE=1e-3"
set "MOMENTUM=0.9"
set "WEIGHT_DECAY=1e-4"
set "OPTIMIZER=adamw"
set "ADAMW_BETA1=0.9"
set "ADAMW_BETA2=0.999"
set "ADAMW_EPS=1e-8"
set "USE_COSINE_LR_ARG="
set "SCHEDULES=50 75"
set "ENABLE_AMP_ARG="
set "SAVE_PERIOD=20"
set "BEST_AFTER_EPOCH=0"

set "HEAD_ONLY_HEAD_LR=1e-3"
set "FULL_BACKBONE_LR=1e-4"
set "FULL_HEAD_LR=1e-3"
set "USE_DISCRIMINATIVE_LR_FOR_FULL_ARG=--use_discriminative_lr"

REM ============================================================
REM 8) Pretrained loading / output naming
REM ============================================================
set "KEEP_PRETRAINED_HEAD_ARG="
set "PRETRAINED_STRICT_ARG="
set "PRETRAINED_TAG_MODE=relative_to_anchor"
set "PRETRAINED_TAG_LAST_K=3"
set "PRETRAINED_TAG_ANCHOR=mindrove_N_except_take_put_adamw_random_kqueue_12"

REM ============================================================
REM 9) Class imbalance / loss options
REM ============================================================
set "USE_WEIGHTED_SAMPLER_ARG="
set "SAMPLER_TIER=%TIER_MODE%"
set "SAMPLER_MODE=sqrt_inv"
set "USE_WEIGHTED_CE_ARG="
set "WEIGHT_METHOD=class_balanced"
set "CB_BETA=0.999"
set "WEIGHT_NORMALIZE_MEAN_ARG="
set "USE_FOCAL_ARG="
set "FOCAL_GAMMA=2.0"
set "FOCAL_USE_ALPHA_ARG="

REM ============================================================
REM 10) Basic validation
REM ============================================================
if "%PROJECT_ROOT%"=="" (
    echo [Error] Please fill PROJECT_ROOT.
    exit /b 1
)
if "%PY_SCRIPT%"=="" set "PY_SCRIPT=%PROJECT_ROOT%\ft_and_test\train_mapstyle_finetune_and_test.py"
if "%DATASET_ROOT%"=="" (
    echo [Error] Please fill DATASET_ROOT.
    exit /b 1
)
if "%LABEL_MAP_JSON%"=="" (
    echo [Error] Please fill LABEL_MAP_JSON.
    exit /b 1
)
if "%PRETRAIN_PARENT%"=="" (
    echo [Error] Please fill PRETRAIN_PARENT, or fill EMG_PRETRAIN_ROOT and IMU_PRETRAIN_ROOT and still set PRETRAIN_PARENT for naming anchor consistency.
    exit /b 1
)
if "%OUTPUT_ROOT%"=="" (
    echo [Error] Please fill OUTPUT_ROOT.
    exit /b 1
)

cd /d "%PROJECT_ROOT%"
if errorlevel 1 (
    echo [Error] Failed to enter PROJECT_ROOT: %PROJECT_ROOT%
    exit /b 1
)
set "PYTHONPATH=%PROJECT_ROOT%"

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
if not exist "%OUTPUT_ROOT%\checkpoint_lists" mkdir "%OUTPUT_ROOT%\checkpoint_lists"

if "%EMG_PRETRAIN_ROOT%"=="" set "EMG_PRETRAIN_ROOT=%PRETRAIN_PARENT%\signal_emg"
if "%IMU_PRETRAIN_ROOT%"=="" set "IMU_PRETRAIN_ROOT=%PRETRAIN_PARENT%\signal_imu"

if /I "%DRY_RUN%"=="1" echo [DryRun] Commands will be printed but not executed.

echo ============================================================
echo MindRove finetuning: train_manifest_except_take_put.jsonl
echo Project:        %PROJECT_ROOT%
echo Script:         %PY_SCRIPT%
echo Dataset:        %DATASET_ROOT%
echo Label map:      %LABEL_MAP_JSON%
echo Pretrain root:  %PRETRAIN_PARENT%
echo Output root:    %OUTPUT_ROOT%
echo Train manifest: %TRAIN_MANIFEST%
echo Val manifest:   %VAL_MANIFEST%
echo Num classes:    %NUM_CLASSES%
echo ============================================================

call :run_signal "emg" "%EMG_PRETRAIN_ROOT%"
if errorlevel 1 exit /b 1

call :run_signal "imu" "%IMU_PRETRAIN_ROOT%"
if errorlevel 1 exit /b 1

echo.
echo All MindRove finetuning runs finished successfully.
echo Output root: %OUTPUT_ROOT%
exit /b 0

:run_signal
set "RG_SIGNAL=%~1"
set "ROOT_DIR=%~2"
call :set_signal_cfg "!RG_SIGNAL!"
if errorlevel 1 exit /b 1
if "!ROOT_DIR!"=="" (
    echo [Error] Empty pretrain root for signal=!RG_SIGNAL!
    exit /b 1
)
if not exist "!ROOT_DIR!" (
    echo [Error] Pretrain root does not exist for signal=!RG_SIGNAL!:
    echo   !ROOT_DIR!
    exit /b 1
)
set "WEIGHTS_PART1="
set "WEIGHTS_PART2="
set /a FOUND_COUNT=0
set /a PART1_COUNT=0
set /a PART2_COUNT=0
set "LIST_FILE=%OUTPUT_ROOT%\checkpoint_lists\signal_!RG_SIGNAL!_%CHECKPOINT_NAME%.txt"
if exist "!LIST_FILE!" del /q "!LIST_FILE!" >nul 2>nul
for /f "usebackq delims=" %%W in (`powershell -NoProfile -Command "Get-ChildItem -LiteralPath '!ROOT_DIR!' -Recurse -Filter '%CHECKPOINT_NAME%' -File ^| Sort-Object FullName ^| ForEach-Object { $_.FullName }"`) do (
    if exist "%%~fW" (
        set /a FOUND_COUNT+=1
        set /a PART_ID=FOUND_COUNT %% 2
        echo %%~fW>>"!LIST_FILE!"
        echo [FOUND][!RG_SIGNAL!][!FOUND_COUNT!] %%~fW
        if !PART_ID! EQU 1 (
            set /a PART1_COUNT+=1
            if defined WEIGHTS_PART1 (set "WEIGHTS_PART1=!WEIGHTS_PART1! "%%~fW"") else (set "WEIGHTS_PART1="%%~fW"")
        ) else (
            set /a PART2_COUNT+=1
            if defined WEIGHTS_PART2 (set "WEIGHTS_PART2=!WEIGHTS_PART2! "%%~fW"") else (set "WEIGHTS_PART2="%%~fW"")
        )
    )
)
echo.
echo ============================================================
echo [Scan] signal=!RG_SIGNAL!
echo [Root] !ROOT_DIR!
echo [Found %CHECKPOINT_NAME%] !FOUND_COUNT!
echo [part_1 count] !PART1_COUNT!
echo [part_2 count] !PART2_COUNT!
echo [List file] !LIST_FILE!
echo ============================================================
if !FOUND_COUNT! EQU 0 (
    echo [Error] No %CHECKPOINT_NAME% found for signal=!RG_SIGNAL!.
    exit /b 1
)
if not "!FOUND_COUNT!"=="%EXPECTED_NUM_CKPTS_PER_SIGNAL%" if not "%ALLOW_CKPT_COUNT_MISMATCH%"=="1" (
    echo [Error] Expected %EXPECTED_NUM_CKPTS_PER_SIGNAL% checkpoints for signal=!RG_SIGNAL!, but found !FOUND_COUNT!.
    echo         Check !LIST_FILE!, or set ALLOW_CKPT_COUNT_MISMATCH=1 intentionally.
    exit /b 1
)
call :run_scratch "!RG_SIGNAL!"
if errorlevel 1 exit /b 1
call :run_one_part "!RG_SIGNAL!" "part_1"
if errorlevel 1 exit /b 1
call :run_one_part "!RG_SIGNAL!" "part_2"
if errorlevel 1 exit /b 1
exit /b 0

:set_signal_cfg
set "RG_SIGNAL=%~1"
if /I "!RG_SIGNAL!"=="emg" (
    set "TARGET_LEN=%EMG_TARGET_LEN%"
    set "AUG_POLICY=%EMG_AUG_POLICY%"
    set "AUG_ARGS=%EMG_AUG_ARGS%"
    set "SIGNAL_NORM_ARGS=--mindrove_left_emg_mean %LEFT_EMG_MEAN% --mindrove_left_emg_std %LEFT_EMG_STD% --mindrove_right_emg_mean %RIGHT_EMG_MEAN% --mindrove_right_emg_std %RIGHT_EMG_STD%"
    exit /b 0
)
if /I "!RG_SIGNAL!"=="imu" (
    set "TARGET_LEN=%IMU_TARGET_LEN%"
    set "AUG_POLICY=%IMU_AUG_POLICY%"
    set "AUG_ARGS=%IMU_AUG_ARGS%"
    set "SIGNAL_NORM_ARGS=--mindrove_left_imu_mean %LEFT_IMU_MEAN% --mindrove_left_imu_std %LEFT_IMU_STD% --mindrove_right_imu_mean %RIGHT_IMU_MEAN% --mindrove_right_imu_std %RIGHT_IMU_STD%"
    exit /b 0
)
echo [Error] Unsupported signal: !RG_SIGNAL!
exit /b 1

:run_scratch
set "RG_SIGNAL=%~1"
call :set_signal_cfg "!RG_SIGNAL!"
if errorlevel 1 exit /b 1
set "SAVE_DIR=%OUTPUT_ROOT%\weights\signal_!RG_SIGNAL!\scratch_full"
set "DATAMAP_DIR=%OUTPUT_ROOT%\datamaps\signal_!RG_SIGNAL!\scratch_full"
if not exist "!SAVE_DIR!" mkdir "!SAVE_DIR!"
if not exist "!DATAMAP_DIR!" mkdir "!DATAMAP_DIR!"
echo.
echo ============================================================
echo [Run] scratch_full signal=!RG_SIGNAL!
echo [Save root]    !SAVE_DIR!
echo [Datamap root] !DATAMAP_DIR!
echo ============================================================
call :run_python "scratch_full" "!SAVE_DIR!" "!DATAMAP_DIR!" "full" "" "%USE_DISCRIMINATIVE_LR_FOR_FULL_ARG% --backbone_learning_rate %FULL_BACKBONE_LR% --head_learning_rate %FULL_HEAD_LR%"
exit /b !ERRORLEVEL!

:run_one_part
set "RG_SIGNAL=%~1"
set "PART_NAME=%~2"
call :set_signal_cfg "!RG_SIGNAL!"
if errorlevel 1 exit /b 1
if /I "!PART_NAME!"=="part_1" (
    set "PART_WEIGHTS=!WEIGHTS_PART1!"
    set "PART_COUNT=!PART1_COUNT!"
) else if /I "!PART_NAME!"=="part_2" (
    set "PART_WEIGHTS=!WEIGHTS_PART2!"
    set "PART_COUNT=!PART2_COUNT!"
) else (
    echo [Error] Unsupported part: !PART_NAME!
    exit /b 1
)
if "!PART_WEIGHTS!"=="" (
    echo [Skip] signal=!RG_SIGNAL! !PART_NAME!: no weights.
    exit /b 0
)
set "SAVE_DIR=%OUTPUT_ROOT%\weights\signal_!RG_SIGNAL!\head_only\!PART_NAME!"
set "DATAMAP_DIR=%OUTPUT_ROOT%\datamaps\signal_!RG_SIGNAL!\head_only\!PART_NAME!"
if not exist "!SAVE_DIR!" mkdir "!SAVE_DIR!"
if not exist "!DATAMAP_DIR!" mkdir "!DATAMAP_DIR!"
echo.
echo ============================================================
echo [Run] pretrained_head_only signal=!RG_SIGNAL! !PART_NAME!
echo [Num weights] !PART_COUNT!
echo [Save root]   !SAVE_DIR!
echo ============================================================
call :run_python "pretrained_head_only_!PART_NAME!" "!SAVE_DIR!" "!DATAMAP_DIR!" "head_only" "--pretrained_weight_paths !PART_WEIGHTS!" "--head_learning_rate %HEAD_ONLY_HEAD_LR%"
if errorlevel 1 exit /b 1
set "SAVE_DIR=%OUTPUT_ROOT%\weights\signal_!RG_SIGNAL!\full\!PART_NAME!"
set "DATAMAP_DIR=%OUTPUT_ROOT%\datamaps\signal_!RG_SIGNAL!\full\!PART_NAME!"
if not exist "!SAVE_DIR!" mkdir "!SAVE_DIR!"
if not exist "!DATAMAP_DIR!" mkdir "!DATAMAP_DIR!"
echo.
echo ============================================================
echo [Run] pretrained_full signal=!RG_SIGNAL! !PART_NAME!
echo [Num weights] !PART_COUNT!
echo [Save root]   !SAVE_DIR!
echo ============================================================
call :run_python "pretrained_full_!PART_NAME!" "!SAVE_DIR!" "!DATAMAP_DIR!" "full" "--pretrained_weight_paths !PART_WEIGHTS!" "%USE_DISCRIMINATIVE_LR_FOR_FULL_ARG% --backbone_learning_rate %FULL_BACKBONE_LR% --head_learning_rate %FULL_HEAD_LR%"
exit /b !ERRORLEVEL!

:run_python
set "RUN_LABEL=%~1"
set "SAVE_DIR=%~2"
set "DATAMAP_DIR=%~3"
set "FT_MODE=%~4"
set "PRETRAINED_ARGS=%~5"
set "LR_MODE_ARGS=%~6"
echo [Command label] !RUN_LABEL!
if /I "%DRY_RUN%"=="1" (
    echo "%PYTHON_BIN%" "%PY_SCRIPT%" --run_mode train ... --finetune_mode !FT_MODE! !PRETRAINED_ARGS! !LR_MODE_ARGS!
    exit /b 0
)
"%PYTHON_BIN%" "%PY_SCRIPT%" ^
  --run_mode train ^
  --dataset_root "%DATASET_ROOT%" ^
  --label_map_json "%LABEL_MAP_JSON%" ^
  --train_manifest "%TRAIN_MANIFEST%" ^
  --val_manifest "%VAL_MANIFEST%" ^
  --save_path "!SAVE_DIR!" ^
  --datamap_csv_path "!DATAMAP_DIR!" ^
  --use_modality %USE_MODALITY% ^
  --tier_mode %TIER_MODE% ^
  --num_classes %NUM_CLASSES% ^
  --mindrove_hands %MINDROVE_HANDS% ^
  --mindrove_signals !RG_SIGNAL! ^
  --mindrove_target_len !TARGET_LEN! ^
  %MINDROVE_MERGE_HANDS_ARG% ^
  %MINDROVE_APPLY_NORMALIZATION_ARG% ^
  %MINDROVE_APPLY_AUGMENTATION_ARG% ^
  %DISABLE_TRAIN_AUGMENTATION_ARG% ^
  !SIGNAL_NORM_ARGS! ^
  !AUG_ARGS! ^
  --num_workers_train %NUM_WORKERS_TRAIN% ^
  --num_workers_val %NUM_WORKERS_VAL% ^
  --prefetch_factor_train %PREFETCH_FACTOR_TRAIN% ^
  --prefetch_factor_val %PREFETCH_FACTOR_VAL% ^
  %DISABLE_VAL_ARG% ^
  --model_depth %MODEL_DEPTH% ^
  %L2_NORMALIZE_BEFORE_FC_ARG% ^
  --mindrove_arch %MINDROVE_ARCH% ^
  --mindrove_base_channels %MINDROVE_BASE_CHANNELS% ^
  --mindrove_stem_kernel_size %MINDROVE_STEM_KERNEL_SIZE% ^
  --mindrove_stem_stride %MINDROVE_STEM_STRIDE% ^
  %MINDROVE_USE_STEM_POOL_ARG% ^
  %MINDROVE_ZERO_INIT_RESIDUAL_ARG% ^
  --epochs %EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --learning_rate %LEARNING_RATE% ^
  --momentum %MOMENTUM% ^
  --weight_decay %WEIGHT_DECAY% ^
  --optimizer %OPTIMIZER% ^
  --adamw_beta1 %ADAMW_BETA1% ^
  --adamw_beta2 %ADAMW_BETA2% ^
  --adamw_eps %ADAMW_EPS% ^
  %USE_COSINE_LR_ARG% ^
  --schedules %SCHEDULES% ^
  --seed %SEED% ^
  --pretrained_tag_mode %PRETRAINED_TAG_MODE% ^
  --pretrained_tag_last_k %PRETRAINED_TAG_LAST_K% ^
  --pretrained_tag_anchor %PRETRAINED_TAG_ANCHOR% ^
  %KEEP_PRETRAINED_HEAD_ARG% ^
  %PRETRAINED_STRICT_ARG% ^
  --finetune_mode !FT_MODE! ^
  !LR_MODE_ARGS! ^
  %USE_WEIGHTED_SAMPLER_ARG% ^
  --sampler_tier %SAMPLER_TIER% ^
  --sampler_mode %SAMPLER_MODE% ^
  %USE_WEIGHTED_CE_ARG% ^
  --weight_method %WEIGHT_METHOD% ^
  --cb_beta %CB_BETA% ^
  %WEIGHT_NORMALIZE_MEAN_ARG% ^
  %USE_FOCAL_ARG% ^
  --focal_gamma %FOCAL_GAMMA% ^
  %FOCAL_USE_ALPHA_ARG% ^
  %ENABLE_AMP_ARG% ^
  --save_period %SAVE_PERIOD% ^
  --best_after_epoch %BEST_AFTER_EPOCH% ^
  !PRETRAINED_ARGS!
if errorlevel 1 (
    echo [Error] Failed run label=!RUN_LABEL! signal=!RG_SIGNAL! mode=!FT_MODE!
    exit /b 1
)
exit /b 0
