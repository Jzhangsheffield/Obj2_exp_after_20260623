@echo off
if "%~3"=="" (
  echo Usage: run_one.bat STAGE INDEX FOLD [outer]
  exit /b 2
)
call "%~dp0_env.bat"
set EXTRA=
if /I "%~4"=="outer" set EXTRA=--outer-test
"%PYTHON_BIN%" "%PACKAGE_ROOT%\run.py" pipeline --stage %~1 --index %~2 --fold %~3 %EXTRA% --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1
