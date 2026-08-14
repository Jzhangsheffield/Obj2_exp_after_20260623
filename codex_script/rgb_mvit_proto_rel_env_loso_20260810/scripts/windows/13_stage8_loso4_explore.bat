@echo off
setlocal
call "%~dp0_env.bat"

set "TARGET_FOLD=%~1"
if not defined TARGET_FOLD set "TARGET_FOLD=all"

if not exist "%PROJECT_ROOT%\results\rgb_mvit_pr_env_loso_20260810\runtime\splits\protocol_audit.json" (
    echo ERROR: protocol audit is missing. Run 01_prepare.bat first.
    exit /b 1
)

echo Stage 8 exploratory LOSO; held-out fold=%TARGET_FOLD%
for /L %%I in (0,1,7) do (
    "%PYTHON_BIN%" "%PACKAGE_ROOT%\run.py" pipeline --stage stage8 --index %%I --fold %TARGET_FOLD% --outer-test --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
    if errorlevel 1 exit /b 1
)

"%PYTHON_BIN%" "%PACKAGE_ROOT%\run.py" summarize --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1

echo Stage 8 completed for fold=%TARGET_FOLD%.
endlocal

