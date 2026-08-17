@echo off
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
for /L %%I in (0,1,7) do (
  "%PYTHON_BOOTSTRAP%" "%RUNNER%" pipeline --platform windows --config "%EXP_CONFIG%" --stage middle_loss_screen --index %%I --fold dev_N || exit /b 1
)

