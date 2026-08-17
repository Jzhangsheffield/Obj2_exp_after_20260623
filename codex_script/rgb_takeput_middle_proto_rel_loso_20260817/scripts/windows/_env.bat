@echo off
set "PACKAGE_ROOT=%~dp0..\.."
if not defined PYTHON_BOOTSTRAP set "PYTHON_BOOTSTRAP=python"
set "RUNNER=%PACKAGE_ROOT%\run.py"
set "EXP_CONFIG=%PACKAGE_ROOT%\config\experiment_config.json"

