@echo off
if "%~2"=="" (
  echo Usage: u06_inspect_missing.bat MANIFEST.csv pretrain^|finetune^|evaluate^|test
  exit /b 2
)
call "%~dp0_env.bat"
"%PYTHON_BIN%" -B "%CONFIRM_ROOT%\tools\inspect_unified_missing.py" --manifest "%~1" --phase "%~2" --results-root "%PROJECT_ROOT%\results\rgb_mvit_pr_unified_followup_20260813"
if errorlevel 1 exit /b 1
