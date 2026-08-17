@echo off
setlocal
call "%~dp0_env.bat"
"%PYTHON_BOOTSTRAP%" "%RUNNER%" validate --platform windows --config "%EXP_CONFIG%" || exit /b 1
"%PYTHON_BOOTSTRAP%" "%RUNNER%" prepare --platform windows --config "%EXP_CONFIG%" || exit /b 1
echo Manifest generation and validation completed.

