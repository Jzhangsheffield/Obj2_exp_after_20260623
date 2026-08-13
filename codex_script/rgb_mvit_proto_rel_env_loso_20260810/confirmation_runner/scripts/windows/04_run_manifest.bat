@echo off
if "%~1"=="" (
  echo Usage: 04_run_manifest.bat MANIFEST.csv [phases]
  exit /b 2
)
call "%~dp0_env.bat"
set PHASES=%~2
if "%PHASES%"=="" set PHASES=pretrain,proto-env,finetune,test
"%PYTHON_BIN%" "%CONFIRM_ROOT%\run_confirmation.py" run-all --manifest "%~1" --phases "%PHASES%" --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1
