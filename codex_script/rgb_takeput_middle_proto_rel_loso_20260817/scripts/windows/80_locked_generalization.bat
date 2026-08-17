@echo off
setlocal
call "%~dp0_env.bat"
if "%~1"=="" goto usage
if "%~2"=="" goto usage
for %%F in (test_M test_J test_MR) do (
  "%PYTHON_BOOTSTRAP%" "%RUNNER%" pipeline --platform windows --config "%EXP_CONFIG%" --stage %~1 --experiment-id %~2 --fold %%F || exit /b 1
)
exit /b 0
:usage
echo Usage: 80_locked_generalization.bat STAGE EXPERIMENT_ID
exit /b 2

