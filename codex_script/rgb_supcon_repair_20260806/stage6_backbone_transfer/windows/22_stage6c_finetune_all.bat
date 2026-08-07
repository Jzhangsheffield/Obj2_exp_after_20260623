@echo off
call "%~dp0..\..\scripts\windows\common_env.bat"
for /L %%I in (0,1,2) do (
  "%PYTHON_BIN%" "%LAUNCHER%" stage6c_finetune --index %%I --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%" %DRY_ARG%
  if errorlevel 1 exit /b 1
)
