@echo off
setlocal EnableExtensions
call "%~dp0common_env.bat"
for %%E in (mvit_v2_s_original_supcon resnet3d18_motioncrop_supcon) do (
  echo [Pretrain] %%E
  "%PYTHON_BIN%" "%LAUNCHER%" pretrain --experiment %%E --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%" --python-bin "%PYTHON_BIN%" %DRY_ARG%
  if errorlevel 1 exit /b 1
)
exit /b 0
