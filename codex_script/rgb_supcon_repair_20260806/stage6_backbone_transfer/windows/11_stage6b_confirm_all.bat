@echo off
call "%~dp0..\..\scripts\windows\common_env.bat"
for /L %%I in (0,1,1) do (
  "%PYTHON_BIN%" "%LAUNCHER%" stage6b_confirm --index %%I --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%" %DRY_ARG%
  if errorlevel 1 exit /b 1
)
