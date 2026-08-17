@echo off
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
for /L %%I in (0,1,2) do (
  "%PYTHON_BOOTSTRAP%" "%RUNNER%" pipeline --platform windows --config "%EXP_CONFIG%" --stage middle_rel_topk --index %%I --fold dev_N || exit /b 1
)

