@echo off
if "%~1"=="" (
  echo Usage: 06_inspect_missing.bat MANIFEST.csv [pretrain^|finetune^|test]
  exit /b 2
)
call "%~dp0_env.bat"
set PHASE=%~2
if "%PHASE%"=="" set PHASE=test
"%PYTHON_BIN%" "%CONFIRM_ROOT%\tools\inspect_missing_runs.py" --manifest "%~1" --phase "%PHASE%"
if errorlevel 1 exit /b 1
