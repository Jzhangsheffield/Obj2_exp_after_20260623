@echo off
if "%~1"=="" (
  echo Usage: u05_summarize.bat MANIFEST.csv
  exit /b 2
)
call "%~dp0_env.bat"
"%PYTHON_BIN%" -B "%CONFIRM_ROOT%\tools\analyze_unified.py" --results-root "%PROJECT_ROOT%\results\rgb_mvit_pr_unified_followup_20260813" --manifest "%~1"
if errorlevel 1 exit /b 1
