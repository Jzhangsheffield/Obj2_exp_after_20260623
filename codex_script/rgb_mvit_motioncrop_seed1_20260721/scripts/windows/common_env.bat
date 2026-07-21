@echo off
REM Optional overrides may be set before calling any stage script.
if not defined PYTHON_BIN set "PYTHON_BIN=python"
if not defined DATASET_ROOT set "DATASET_ROOT=C:\Junxi_data_for_training_speedup\Final_Mapstyle_Dataset"
for %%I in ("%~dp0..\..") do set "PACKAGE_ROOT=%%~fI"
for %%I in ("%PACKAGE_ROOT%\..\..") do set "PROJECT_ROOT=%%~fI"
set "PYTHONHASHSEED=1"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"
set "LAUNCHER=%PACKAGE_ROOT%\run_experiment.py"
set "DRY_ARG="
if "%DRY_RUN%"=="1" set "DRY_ARG=--dry-run"
