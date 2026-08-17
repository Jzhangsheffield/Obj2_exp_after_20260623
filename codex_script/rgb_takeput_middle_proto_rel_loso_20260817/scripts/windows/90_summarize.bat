@echo off
setlocal
call "%~dp0_env.bat"
"%PYTHON_BOOTSTRAP%" "%RUNNER%" summarize --platform windows --config "%EXP_CONFIG%"

