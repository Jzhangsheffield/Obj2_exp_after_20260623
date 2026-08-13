@echo off
if "%~2"=="" (
  echo Usage: u04_run_manifest.bat MANIFEST.csv pretrain,finetune,evaluate_or_test
  exit /b 2
)
call "%~dp0_env.bat"
"%PYTHON_BIN%" -B "%CONFIRM_ROOT%\run_unified.py" run-all --manifest "%~1" --phases "%~2" --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1
