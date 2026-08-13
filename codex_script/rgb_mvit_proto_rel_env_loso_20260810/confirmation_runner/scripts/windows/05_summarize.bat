@echo off
if "%~1"=="" (
  echo Usage: 05_summarize.bat MANIFEST.csv
  exit /b 2
)
call "%~dp0_env.bat"
"%PYTHON_BIN%" "%CONFIRM_ROOT%\tools\analyze_confirmation.py" --results-root "%PROJECT_ROOT%\results\rgb_mvit_pr_confirm_20260812" --manifest "%~1"
if errorlevel 1 exit /b 1
