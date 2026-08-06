@echo off
call "%~dp0..\..\scripts\windows\common_env.bat"
"%PYTHON_BIN%" "%LAUNCHER%" prepare --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%" %DRY_ARG%
if errorlevel 1 exit /b %errorlevel%
