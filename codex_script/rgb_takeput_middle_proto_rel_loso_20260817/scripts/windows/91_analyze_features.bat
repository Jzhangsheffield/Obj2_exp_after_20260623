@echo off
setlocal
call "%~dp0_env.bat"
if "%~4"=="" goto usage
"%PYTHON_BOOTSTRAP%" "%RUNNER%" features --platform windows --config "%EXP_CONFIG%" --stage %~1 --experiment-id %~2 --fold %~3 --checkpoint-kind classifier --policy %~4
exit /b %errorlevel%
:usage
echo Usage: 91_analyze_features.bat STAGE EXPERIMENT_ID FOLD full_or_head_only
exit /b 2

