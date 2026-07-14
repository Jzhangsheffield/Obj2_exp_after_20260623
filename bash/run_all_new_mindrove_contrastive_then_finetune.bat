@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM MindRove-only launcher.
REM
REM Run all newly generated MindRove contrastive scripts first,
REM then run their corresponding fine-tuning scripts.
REM
REM Order:
REM   1) random_kqueue_12 contrastive
REM   2) stage5_reltopk_random_10 contrastive
REM   3) reltopk_random_12 contrastive
REM   4) random_kqueue_12 fine-tune
REM   5) stage5_reltopk_random_10 fine-tune
REM   6) reltopk_random_12 fine-tune
REM
REM Set DRY_RUN=1 before running this script to print commands only.
REM ============================================================

set "BASH_DIR=%~dp0"
if not defined DRY_RUN set "DRY_RUN=0"

set "MIND_TRAIN_RANDOM_KQUEUE=%BASH_DIR%run_mindrove_N_except_take_put_adamw_random_kqueue_12.bat"
set "MIND_TRAIN_STAGE5=%BASH_DIR%run_mindrove_N_except_take_put_adamw_stage5_random_10.bat"
set "MIND_TRAIN_RELTOPK=%BASH_DIR%run_mindrove_N_except_take_put_adamw_reltopk_random_12.bat"

set "MIND_FT_RANDOM_KQUEUE=%BASH_DIR%finetune_mindrove_N_except_take_put_adamw_random_kqueue_12_seed1.bat"
set "MIND_FT_STAGE5=%BASH_DIR%finetune_mindrove_N_except_take_put_adamw_stage5_random_10_seed1.bat"
set "MIND_FT_RELTOPK=%BASH_DIR%finetune_mindrove_N_except_take_put_adamw_reltopk_random_12_seed1.bat"

echo ============================================================
echo MindRove contrastive + fine-tune launcher
echo Bash dir: %BASH_DIR%
echo DRY_RUN=%DRY_RUN%
echo ============================================================

call :require_file "%MIND_TRAIN_RANDOM_KQUEUE%"
if errorlevel 1 exit /b 1
call :require_file "%MIND_TRAIN_STAGE5%"
if errorlevel 1 exit /b 1
call :require_file "%MIND_TRAIN_RELTOPK%"
if errorlevel 1 exit /b 1
call :require_file "%MIND_FT_RANDOM_KQUEUE%"
if errorlevel 1 exit /b 1
call :require_file "%MIND_FT_STAGE5%"
if errorlevel 1 exit /b 1
call :require_file "%MIND_FT_RELTOPK%"
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo Phase 1: MindRove contrastive training
echo ============================================================

call :run_bat "mindrove_random_kqueue_12" "%MIND_TRAIN_RANDOM_KQUEUE%"
if errorlevel 1 exit /b 1
call :run_bat "mindrove_stage5_reltopk_random_10" "%MIND_TRAIN_STAGE5%"
if errorlevel 1 exit /b 1
call :run_bat "mindrove_reltopk_random_12" "%MIND_TRAIN_RELTOPK%"
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo Phase 2: MindRove fine-tuning
echo ============================================================

call :run_bat "ft_mindrove_random_kqueue_12" "%MIND_FT_RANDOM_KQUEUE%"
if errorlevel 1 exit /b 1
call :run_bat "ft_mindrove_stage5_reltopk_random_10" "%MIND_FT_STAGE5%"
if errorlevel 1 exit /b 1
call :run_bat "ft_mindrove_reltopk_random_12" "%MIND_FT_RELTOPK%"
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo All MindRove contrastive and fine-tuning scripts finished.
echo ============================================================
exit /b 0

:require_file
set "REQ_FILE=%~1"
if not exist "%REQ_FILE%" (
    echo [Error] Required script not found:
    echo   %REQ_FILE%
    exit /b 1
)
exit /b 0

:run_bat
set "JOB_LABEL=%~1"
set "JOB_SCRIPT=%~2"
echo.
echo [Run] %JOB_LABEL%
echo   %JOB_SCRIPT%
if "%DRY_RUN%"=="1" (
    echo [DryRun] call "%JOB_SCRIPT%"
    exit /b 0
)
call "%JOB_SCRIPT%"
if errorlevel 1 (
    echo [Error] Failed: %JOB_LABEL%
    exit /b 1
)
echo [OK] %JOB_LABEL%
exit /b 0
