@echo off
setlocal EnableDelayedExpansion
call "%~dp0_env.bat"
for %%S in (middle_rel_start middle_combined middle_p2_sentinel) do (
  "%PYTHON_BOOTSTRAP%" "%RUNNER%" list --platform windows --config "%EXP_CONFIG%" --stage %%S
)
echo Run only the approved follow-up rows with: run.py pipeline --stage STAGE --index INDEX --fold dev_N

