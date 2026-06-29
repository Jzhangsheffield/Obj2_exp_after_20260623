@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM MindRove EMG + IMU SupLoss ablation for N_as_test / train_manifest_except_take_put.jsonl
REM 2 signals x 22 configs = 44 runs.
REM Normalization is copied from N_as_test\train_normalization_stats\except_take_put_mindrove_stats.json and rounded to 4 decimals.
REM ============================================================

REM ------------------------------
REM 1) Project / data paths
REM ------------------------------
set "PROJECT_ROOT=D:\junxi_data\experiments_after_260623"
set "PY_SCRIPT=%PROJECT_ROOT%\train\MoCo_main_supcon_mapstyle_varproto_debug_mindrove_modified_varlen_topk_adamw.py"
set "DATASET_ROOT=C:\MyFolder\mes19jz\Final_Mapstyle_Dataset"
set "TRAIN_MANIFEST_NAME=N_as_test\train_manifest_except_take_put.jsonl"
set "LABEL_MAP_JSON=%DATASET_ROOT%\label_map.json"
set "NORMALIZATION_STATS_JSON=%DATASET_ROOT%\N_as_test\train_normalization_stats\except_take_put_mindrove_stats.json"
set "OUT_ROOT=%PROJECT_ROOT%\results\mindrove_N_except_take_put_adamw_44"
set "PYTHON_BIN=python"

cd /d "%PROJECT_ROOT%"
if errorlevel 1 (
  echo [Error] Failed to enter PROJECT_ROOT: %PROJECT_ROOT%
  exit /b 1
)
set "PYTHONPATH=%PROJECT_ROOT%"

REM ------------------------------
REM 2) Dataset / MindRove parameters
REM ------------------------------
set "TIER_MODE=tier1"
set "MINDROVE_HANDS=left right"
set "MINDROVE_MERGE_HANDS_ARG="
set "MINDROVE_APPLY_AUGMENTATION_ARG=--mindrove_apply_augmentation"
set "MINDROVE_APPLY_NORMALIZATION_ARG=--mindrove_apply_normalization"
set "MINDROVE_TARGET_LEN=256"
set "EMG_TARGET_LEN=512"
set "IMU_TARGET_LEN=128"
set "MINDROVE_PACK_LENGTH_POLICY=max"
set "MINDROVE_PACK_TARGET_LEN="

REM ------------------------------
REM 3) Normalization parameters, IMU order = [acc0, acc1, acc2, gyro0, gyro1, gyro2]
REM ------------------------------
set "LEFT_EMG_MEAN=[0.0068, 0.0205, 0.0096, 0.0238, 0.0219, 0.0168, -0.0024, -0.0058]"
set "LEFT_EMG_STD=[30.1656, 33.4225, 38.0212, 36.9723, 36.7116, 35.9175, 22.1645, 18.3178]"
set "RIGHT_EMG_MEAN=[0.0102, -0.0020, -0.0492, 0.0158, 0.0006, -0.0251, -0.0112, -0.0135]"
set "RIGHT_EMG_STD=[30.6975, 41.2925, 49.7859, 49.2473, 45.4501, 35.0010, 28.3983, 22.1492]"
set "LEFT_IMU_MEAN=[-0.0482, -0.0028, -0.0314, -0.6035, -0.6008, -2.7832]"
set "LEFT_IMU_STD=[0.2216, 0.1471, 0.0910, 33.6979, 33.1239, 34.2654]"
set "RIGHT_IMU_MEAN=[-0.0130, 0.0114, -0.0237, 0.9942, -0.9100, 3.8193]"
set "RIGHT_IMU_STD=[0.2280, 0.2044, 0.1116, 39.6820, 41.1152, 40.6422]"

REM ------------------------------
REM 4) DataLoader parameters
REM ------------------------------
set "BATCH_SIZE=64"
set "NUM_WORKERS=8"
set "PREFETCH_FACTOR="
set "PIN_MEMORY_ARG="
set "VERIFY_PATHS_ON_INIT_ARG=--verify_paths_on_init"
set "PROTO_REFRESH_BATCH_SIZE=64"
set "PROTO_REFRESH_NUM_WORKERS=8"
set "PROTO_REFRESH_PREFETCH_FACTOR="
set "PROTO_REFRESH_PIN_MEMORY_ARG="
set "PROTO_REFRESH_VERIFY_PATHS_ON_INIT_ARG=--proto_refresh_verify_paths_on_init"

REM ------------------------------
REM 5) Time-series model / contrastive parameters
REM ------------------------------
set "TS_ARCH=resnet10_1d"
set "TS_BASE_CHANNELS=64"
set "TS_STEM_KERNEL_SIZE=7"
set "TS_STEM_STRIDE=2"
set "TS_USE_STEM_POOL_ARG=--ts_use_stem_pool"
set "TS_ZERO_INIT_RESIDUAL_ARG=--no-ts_zero_init_residual"
set "PROJ_DIM=128"
set "K_QUEUE=1088"
set "TEMPERATURE=0.07"
set "USE_MLP_ARG=--mlp"
set "CONTRASTIVE_LOSS=suploss"
set "NUM_POSITIVE=6"
set "EXCLUDE_INVALID_QUEUE_ARG="

REM ------------------------------
REM 6) Prototype / relative loss parameters
REM ------------------------------
set "WARMUP_EPOCHS=50"
set "RECLUSTER_INTERVAL=10"
set "NUM_PROTOTYPES_PER_CLASS="
set "LAMBDA_PROTO=1.0"
set "PROTO_TEMPERATURE=0.07"
set "ENABLE_PROTOTYPE_TEMPERATURE_SCALING_ARG="
set "PROTO_TEMPERATURE_EPS=1e-6"
set "PROTO_KMEANS_RANDOM_STATE=42"
set "PROTO_KMEANS_N_INIT=10"
set "PROTO_KMEANS_MAX_ITER=300"
set "LAMBDA_REL=1.0"
set "PROTO_EMA_MOMENTUM=0.99"
set "REL_SAME_MARGIN=0.01"
set "REL_DIFF_MARGIN=0.01"
set "REL_SAME_WEIGHT=1.0"
set "REL_DIFF_WEIGHT=1.0"
set "REL_TOPK_DIFF_CLASSES=3"

REM ------------------------------
REM 7) Loss-stage schedule parameters
REM ------------------------------
set "ENABLE_LOSS_STAGE_SCHEDULE_ARG=--enable_loss_stage_schedule"
set "PROTO_LOSS_START_EPOCH=50"
set "REL_LOSS_START_EPOCH=50"
set "REL_LOSS_END_EPOCH=200"
set "REL_LAMBDA_SCHEDULE=cosine"

REM ------------------------------
REM 8) Optimizer / training control
REM ------------------------------
set "EPOCHS=200"
set "START_EPOCH=0"
set "LEARNING_RATE=1e-3"
set "USE_COSINE_LR_ARG="
set "SCHEDULE=50 100 150"
set "WEIGHT_DECAY=1e-4"
set "MOMENTUM=0.9"
set "OPTIMIZER=adamw"
set "ADAMW_BETAS=0.9 0.999"
set "ADAMW_EPS=1e-8"
set "ADAMW_AMSGRAD_ARG=--no-adamw_amsgrad"
set "NO_DDP_ARG=--no_ddp"
set "SEED_ARG="
set "PRINT_FREQ=10"
set "SAVE_INTERVAL=50"
set "USE_SYNCBN_ARG=--use_syncbn"
set "FIND_UNUSED_PARAMETERS_ARG=--no-find_unused_parameters"

REM ------------------------------
REM 9) Debug parameters
REM ------------------------------
set "DEBUG_MODE_ARG="
set "DEBUG_LOG_INTERVAL=20"
set "DEBUG_GRAD_STATS_ARG=--debug_grad_stats"
set "DEBUG_PARAM_UPDATE_STATS_ARG=--debug_param_update_stats"
set "DEBUG_BATCH_LABEL_STATS_ARG=--debug_batch_label_stats"
set "DEBUG_PROTO_STATS_ARG=--debug_proto_stats"
set "DEBUG_FEATURE_STATS_ARG=--debug_feature_stats"
set "DEBUG_NONFINITE_CHECK_ARG=--debug_nonfinite_check"
set "DEBUG_ABORT_ON_NONFINITE_ARG=--no-debug_abort_on_nonfinite"
set "DEBUG_GRAD_TOPK=8"
set "DEBUG_PARAM_PATTERNS=module.encoder_q.fc,encoder_q.fc,module.encoder_q.layer4,encoder_q.layer4,module.encoder_q.conv1,encoder_q.conv1"
set "DEBUG_PARAM_FALLBACK_LAST_N=4"
set "DEBUG_WRITE_JSONL_ARG=--no-debug_write_jsonl"
set "DEBUG_JSONL_NAME=debug_train_log.jsonl"

REM ------------------------------
REM 10) Augmentation policies copied from prior best-augmentation BAT
REM ------------------------------
set "EMG_AUG_POLICY=tw_scale_noise_dropout_negate"
set "EMG_AUG_ARGS=--mindrove_time_warp_prob 0.5 --mindrove_time_warp_sigma 0.2 --mindrove_time_warp_num_knots 3 --mindrove_emg_scaling_prob 0.8 --mindrove_emg_scaling_sigma 0.10 --mindrove_emg_noise_prob 0.8 --mindrove_emg_noise_sigma 0.05 --mindrove_emg_drift_prob 0.0 --mindrove_emg_negate_prob 0.1 --mindrove_emg_channel_dropout_prob 0.1 --mindrove_emg_channel_dropout_max_channels 1 --mindrove_imu_scaling_prob 0.8 --mindrove_imu_scaling_sigma 0.05 --mindrove_imu_noise_prob 0.8 --mindrove_imu_noise_sigma 0.03 --mindrove_imu_drift_prob 0.0 --mindrove_imu_negate_prob 0.1 --mindrove_imu_channel_dropout_prob 0.1 --mindrove_imu_channel_dropout_max_channels 1"

set "IMU_AUG_POLICY=tw_scale_noise_negate"
set "IMU_AUG_ARGS=--mindrove_time_warp_prob 0.5 --mindrove_time_warp_sigma 0.2 --mindrove_time_warp_num_knots 3 --mindrove_emg_scaling_prob 0.8 --mindrove_emg_scaling_sigma 0.10 --mindrove_emg_noise_prob 0.8 --mindrove_emg_noise_sigma 0.05 --mindrove_emg_drift_prob 0.0 --mindrove_emg_negate_prob 0.1 --mindrove_emg_channel_dropout_prob 0.0 --mindrove_imu_scaling_prob 0.8 --mindrove_imu_scaling_sigma 0.05 --mindrove_imu_noise_prob 0.8 --mindrove_imu_noise_sigma 0.03 --mindrove_imu_drift_prob 0.0 --mindrove_imu_negate_prob 0.1 --mindrove_imu_channel_dropout_prob 0.0"

if not exist "%OUT_ROOT%" mkdir "%OUT_ROOT%"

echo ============================================================
echo MindRove N_as_test except_take_put ablation
echo Manifest: %TRAIN_MANIFEST_NAME%
echo Stats:    %NORMALIZATION_STATS_JSON%
echo Output:   %OUT_ROOT%
echo ============================================================

for %%S in (emg imu) do (
    call :run_signal "%%S"
    if errorlevel 1 exit /b 1
)

echo.
echo All MindRove runs finished successfully.
echo Output root: %OUT_ROOT%
exit /b 0

REM ============================================================
REM Run one signal: emg or imu
REM ============================================================
:run_signal
set "CUR_SIGNAL=%~1"
if /I "!CUR_SIGNAL!"=="emg" (
    set "CUR_AUG_POLICY=!EMG_AUG_POLICY!"
    set "CUR_AUG_ARGS=!EMG_AUG_ARGS!"
    set "CUR_TARGET_LEN=!EMG_TARGET_LEN!"
) else if /I "!CUR_SIGNAL!"=="imu" (
    set "CUR_AUG_POLICY=!IMU_AUG_POLICY!"
    set "CUR_AUG_ARGS=!IMU_AUG_ARGS!"
    set "CUR_TARGET_LEN=!IMU_TARGET_LEN!"
) else (
    echo [Error] Unsupported signal: !CUR_SIGNAL!
    exit /b 1
)

echo.
echo ============================================================
echo Running signal=!CUR_SIGNAL! aug=!CUR_AUG_POLICY! target_len=!CUR_TARGET_LEN!
echo ============================================================

call :run_one_cfg "!CUR_SIGNAL!" "contrastive_only" "" "1" "suploss_only"
if errorlevel 1 exit /b 1

for %%D in (1 2 3) do (
    call :run_one_cfg "!CUR_SIGNAL!" "contrastive_proto" "" "%%D" "suploss_proto_p%%D"
    if errorlevel 1 exit /b 1
)

for %%D in (1 2 3) do (
    for %%P in (0.3 0.5 0.8) do (
        call :run_one_cfg "!CUR_SIGNAL!" "contrastive_rel" "%%P" "%%D" "suploss_rel_p%%D_prem%%P"
        if errorlevel 1 exit /b 1
    )
)

for %%D in (1 2 3) do (
    for %%P in (0.3 0.5 0.8) do (
        call :run_one_cfg "!CUR_SIGNAL!" "contrastive_proto_rel" "%%P" "%%D" "suploss_proto_rel_p%%D_prem%%P"
        if errorlevel 1 exit /b 1
    )
)

exit /b 0

REM ============================================================
REM Run one ablation config
REM ============================================================
:run_one_cfg
set "RG_SIGNAL=%~1"
set "RG_ABLATION=%~2"
set "RG_PREM=%~3"
set "RG_PROTO=%~4"
set "RG_RUN_NAME=%~5"

set "PREVIEW_ARG="
if /I "!RG_ABLATION!"=="contrastive_rel" set "PREVIEW_ARG=--preview_ema_momentum !RG_PREM!"
if /I "!RG_ABLATION!"=="contrastive_proto_rel" set "PREVIEW_ARG=--preview_ema_momentum !RG_PREM!"

set "RUN_DIR=%OUT_ROOT%\signal_!RG_SIGNAL!\!RG_RUN_NAME!"

echo.
echo [Run] signal=!RG_SIGNAL! ablation=!RG_ABLATION! proto=!RG_PROTO! preview_ema_momentum=!RG_PREM! save=!RUN_DIR!

if /I "!RG_SIGNAL!"=="emg" (
    "%PYTHON_BIN%" "%PY_SCRIPT%" ^
      %NO_DDP_ARG% ^
      --dataset_root "%DATASET_ROOT%" ^
      --train_manifest_name "%TRAIN_MANIFEST_NAME%" ^
      --label_map_json "%LABEL_MAP_JSON%" ^
      --weight_save_path "!RUN_DIR!" ^
      --tier_mode "%TIER_MODE%" ^
      --mindrove_hands %MINDROVE_HANDS% ^
      --mindrove_signals !RG_SIGNAL! ^
      --mindrove_target_len !CUR_TARGET_LEN! ^
      --mindrove_emg_target_len %EMG_TARGET_LEN% ^
      --mindrove_pack_length_policy %MINDROVE_PACK_LENGTH_POLICY% ^
      %MINDROVE_MERGE_HANDS_ARG% ^
      %MINDROVE_APPLY_AUGMENTATION_ARG% ^
      %MINDROVE_APPLY_NORMALIZATION_ARG% ^
      --mindrove_left_emg_mean "%LEFT_EMG_MEAN%" ^
      --mindrove_left_emg_std "%LEFT_EMG_STD%" ^
      --mindrove_right_emg_mean "%RIGHT_EMG_MEAN%" ^
      --mindrove_right_emg_std "%RIGHT_EMG_STD%" ^
      !CUR_AUG_ARGS! ^
      --batch_size %BATCH_SIZE% ^
      --num_workers %NUM_WORKERS% ^
      %PIN_MEMORY_ARG% ^
      %VERIFY_PATHS_ON_INIT_ARG% ^
      --ts_arch %TS_ARCH% ^
      --ts_base_channels %TS_BASE_CHANNELS% ^
      --ts_stem_kernel_size %TS_STEM_KERNEL_SIZE% ^
      --ts_stem_stride %TS_STEM_STRIDE% ^
      %TS_USE_STEM_POOL_ARG% ^
      %TS_ZERO_INIT_RESIDUAL_ARG% ^
      --proj_dim %PROJ_DIM% ^
      --K_queue %K_QUEUE% ^
      --temperature %TEMPERATURE% ^
      %USE_MLP_ARG% ^
      --contrastive_loss %CONTRASTIVE_LOSS% ^
      --num_positive %NUM_POSITIVE% ^
      %EXCLUDE_INVALID_QUEUE_ARG% ^
      --ablation_mode !RG_ABLATION! ^
      --warmup_epochs %WARMUP_EPOCHS% ^
      --recluster_interval %RECLUSTER_INTERVAL% ^
      --default_num_prototypes !RG_PROTO! ^
      --lambda_proto %LAMBDA_PROTO% ^
      --proto_temperature %PROTO_TEMPERATURE% ^
      --proto_temperature_eps %PROTO_TEMPERATURE_EPS% ^
      --proto_kmeans_random_state %PROTO_KMEANS_RANDOM_STATE% ^
      --proto_kmeans_n_init %PROTO_KMEANS_N_INIT% ^
      --proto_kmeans_max_iter %PROTO_KMEANS_MAX_ITER% ^
      --proto_refresh_batch_size %PROTO_REFRESH_BATCH_SIZE% ^
      --proto_refresh_num_workers %PROTO_REFRESH_NUM_WORKERS% ^
      %PROTO_REFRESH_VERIFY_PATHS_ON_INIT_ARG% ^
      --lambda_rel %LAMBDA_REL% ^
      --proto_ema_momentum %PROTO_EMA_MOMENTUM% ^
      --rel_same_margin %REL_SAME_MARGIN% ^
      --rel_diff_margin %REL_DIFF_MARGIN% ^
      --rel_same_weight %REL_SAME_WEIGHT% ^
      --rel_diff_weight %REL_DIFF_WEIGHT% ^
      --rel_topk_diff_classes %REL_TOPK_DIFF_CLASSES% ^
      %ENABLE_LOSS_STAGE_SCHEDULE_ARG% ^
      --proto_loss_start_epoch %PROTO_LOSS_START_EPOCH% ^
      --rel_loss_start_epoch %REL_LOSS_START_EPOCH% ^
      --rel_loss_end_epoch %REL_LOSS_END_EPOCH% ^
      --rel_lambda_schedule %REL_LAMBDA_SCHEDULE% ^
      --epochs %EPOCHS% ^
      --start_epoch %START_EPOCH% ^
      --learning_rate %LEARNING_RATE% ^
      %USE_COSINE_LR_ARG% ^
      --schedule %SCHEDULE% ^
      --weight_decay %WEIGHT_DECAY% ^
      --momentum %MOMENTUM% ^
      --optimizer %OPTIMIZER% ^
      --adamw_betas %ADAMW_BETAS% ^
      --adamw_eps %ADAMW_EPS% ^
      %ADAMW_AMSGRAD_ARG% ^
      %SEED_ARG% ^
      --print_freq %PRINT_FREQ% ^
      --save_interval %SAVE_INTERVAL% ^
      %USE_SYNCBN_ARG% ^
      %FIND_UNUSED_PARAMETERS_ARG% ^
      %DEBUG_MODE_ARG% ^
      --debug_log_interval %DEBUG_LOG_INTERVAL% ^
      %DEBUG_GRAD_STATS_ARG% ^
      %DEBUG_PARAM_UPDATE_STATS_ARG% ^
      %DEBUG_BATCH_LABEL_STATS_ARG% ^
      %DEBUG_PROTO_STATS_ARG% ^
      %DEBUG_FEATURE_STATS_ARG% ^
      %DEBUG_NONFINITE_CHECK_ARG% ^
      %DEBUG_ABORT_ON_NONFINITE_ARG% ^
      --debug_grad_topk %DEBUG_GRAD_TOPK% ^
      --debug_param_patterns "%DEBUG_PARAM_PATTERNS%" ^
      --debug_param_fallback_last_n %DEBUG_PARAM_FALLBACK_LAST_N% ^
      %DEBUG_WRITE_JSONL_ARG% ^
      --debug_jsonl_name %DEBUG_JSONL_NAME% ^
      !PREVIEW_ARG!
) else if /I "!RG_SIGNAL!"=="imu" (
    "%PYTHON_BIN%" "%PY_SCRIPT%" ^
      %NO_DDP_ARG% ^
      --dataset_root "%DATASET_ROOT%" ^
      --train_manifest_name "%TRAIN_MANIFEST_NAME%" ^
      --label_map_json "%LABEL_MAP_JSON%" ^
      --weight_save_path "!RUN_DIR!" ^
      --tier_mode "%TIER_MODE%" ^
      --mindrove_hands %MINDROVE_HANDS% ^
      --mindrove_signals !RG_SIGNAL! ^
      --mindrove_target_len !CUR_TARGET_LEN! ^
      --mindrove_imu_target_len %IMU_TARGET_LEN% ^
      --mindrove_pack_length_policy %MINDROVE_PACK_LENGTH_POLICY% ^
      %MINDROVE_MERGE_HANDS_ARG% ^
      %MINDROVE_APPLY_AUGMENTATION_ARG% ^
      %MINDROVE_APPLY_NORMALIZATION_ARG% ^
      --mindrove_left_imu_mean "%LEFT_IMU_MEAN%" ^
      --mindrove_left_imu_std "%LEFT_IMU_STD%" ^
      --mindrove_right_imu_mean "%RIGHT_IMU_MEAN%" ^
      --mindrove_right_imu_std "%RIGHT_IMU_STD%" ^
      !CUR_AUG_ARGS! ^
      --batch_size %BATCH_SIZE% ^
      --num_workers %NUM_WORKERS% ^
      %PIN_MEMORY_ARG% ^
      %VERIFY_PATHS_ON_INIT_ARG% ^
      --ts_arch %TS_ARCH% ^
      --ts_base_channels %TS_BASE_CHANNELS% ^
      --ts_stem_kernel_size %TS_STEM_KERNEL_SIZE% ^
      --ts_stem_stride %TS_STEM_STRIDE% ^
      %TS_USE_STEM_POOL_ARG% ^
      %TS_ZERO_INIT_RESIDUAL_ARG% ^
      --proj_dim %PROJ_DIM% ^
      --K_queue %K_QUEUE% ^
      --temperature %TEMPERATURE% ^
      %USE_MLP_ARG% ^
      --contrastive_loss %CONTRASTIVE_LOSS% ^
      --num_positive %NUM_POSITIVE% ^
      %EXCLUDE_INVALID_QUEUE_ARG% ^
      --ablation_mode !RG_ABLATION! ^
      --warmup_epochs %WARMUP_EPOCHS% ^
      --recluster_interval %RECLUSTER_INTERVAL% ^
      --default_num_prototypes !RG_PROTO! ^
      --lambda_proto %LAMBDA_PROTO% ^
      --proto_temperature %PROTO_TEMPERATURE% ^
      --proto_temperature_eps %PROTO_TEMPERATURE_EPS% ^
      --proto_kmeans_random_state %PROTO_KMEANS_RANDOM_STATE% ^
      --proto_kmeans_n_init %PROTO_KMEANS_N_INIT% ^
      --proto_kmeans_max_iter %PROTO_KMEANS_MAX_ITER% ^
      --proto_refresh_batch_size %PROTO_REFRESH_BATCH_SIZE% ^
      --proto_refresh_num_workers %PROTO_REFRESH_NUM_WORKERS% ^
      %PROTO_REFRESH_VERIFY_PATHS_ON_INIT_ARG% ^
      --lambda_rel %LAMBDA_REL% ^
      --proto_ema_momentum %PROTO_EMA_MOMENTUM% ^
      --rel_same_margin %REL_SAME_MARGIN% ^
      --rel_diff_margin %REL_DIFF_MARGIN% ^
      --rel_same_weight %REL_SAME_WEIGHT% ^
      --rel_diff_weight %REL_DIFF_WEIGHT% ^
      --rel_topk_diff_classes %REL_TOPK_DIFF_CLASSES% ^
      %ENABLE_LOSS_STAGE_SCHEDULE_ARG% ^
      --proto_loss_start_epoch %PROTO_LOSS_START_EPOCH% ^
      --rel_loss_start_epoch %REL_LOSS_START_EPOCH% ^
      --rel_loss_end_epoch %REL_LOSS_END_EPOCH% ^
      --rel_lambda_schedule %REL_LAMBDA_SCHEDULE% ^
      --epochs %EPOCHS% ^
      --start_epoch %START_EPOCH% ^
      --learning_rate %LEARNING_RATE% ^
      %USE_COSINE_LR_ARG% ^
      --schedule %SCHEDULE% ^
      --weight_decay %WEIGHT_DECAY% ^
      --momentum %MOMENTUM% ^
      --optimizer %OPTIMIZER% ^
      --adamw_betas %ADAMW_BETAS% ^
      --adamw_eps %ADAMW_EPS% ^
      %ADAMW_AMSGRAD_ARG% ^
      %SEED_ARG% ^
      --print_freq %PRINT_FREQ% ^
      --save_interval %SAVE_INTERVAL% ^
      %USE_SYNCBN_ARG% ^
      %FIND_UNUSED_PARAMETERS_ARG% ^
      %DEBUG_MODE_ARG% ^
      --debug_log_interval %DEBUG_LOG_INTERVAL% ^
      %DEBUG_GRAD_STATS_ARG% ^
      %DEBUG_PARAM_UPDATE_STATS_ARG% ^
      %DEBUG_BATCH_LABEL_STATS_ARG% ^
      %DEBUG_PROTO_STATS_ARG% ^
      %DEBUG_FEATURE_STATS_ARG% ^
      %DEBUG_NONFINITE_CHECK_ARG% ^
      %DEBUG_ABORT_ON_NONFINITE_ARG% ^
      --debug_grad_topk %DEBUG_GRAD_TOPK% ^
      --debug_param_patterns "%DEBUG_PARAM_PATTERNS%" ^
      --debug_param_fallback_last_n %DEBUG_PARAM_FALLBACK_LAST_N% ^
      %DEBUG_WRITE_JSONL_ARG% ^
      --debug_jsonl_name %DEBUG_JSONL_NAME% ^
      !PREVIEW_ARG!
) else (
    echo [Error] Unsupported signal: !RG_SIGNAL!
    exit /b 1
)

if errorlevel 1 (
    echo [Error] Failed: signal=!RG_SIGNAL! ablation=!RG_ABLATION! proto=!RG_PROTO! preview_ema_momentum=!RG_PREM!
    exit /b 1
)

exit /b 0