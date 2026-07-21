@echo off
setlocal EnableExtensions
call "%~dp0common_env.bat"
"%PYTHON_BIN%" "%LAUNCHER%" summarize --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%" --python-bin "%PYTHON_BIN%" %DRY_ARG%
exit /b %ERRORLEVEL%
