@echo off
call "%~dp0_env.bat"
"%PYTHON_BIN%" -B "%CONFIRM_ROOT%\run_unified.py" prepare --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%" %*
if errorlevel 1 exit /b 1
