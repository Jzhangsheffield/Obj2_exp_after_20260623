@echo off
call "%~dp0..\..\scripts\windows\common_env.bat"
"%PYTHON_BIN%" "%LAUNCHER%" summarize --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
