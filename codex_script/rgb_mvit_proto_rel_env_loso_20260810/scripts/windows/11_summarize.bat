@echo off
call "%~dp0_env.bat"
"%PYTHON_BIN%" "%PACKAGE_ROOT%\run.py" summarize --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1
