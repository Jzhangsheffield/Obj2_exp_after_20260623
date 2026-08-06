@echo off
call "%~dp0..\..\scripts\windows\common_env.bat"
if not "%ALLOW_LOCKED_TEST%"=="YES" (
  echo Locked N test. Set ALLOW_LOCKED_TEST=YES only after candidate selection and all validation runs finish.
  exit /b 2
)
for /L %%I in (0,1,8) do (
  "%PYTHON_BIN%" "%LAUNCHER%" test --index %%I --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%" %DRY_ARG%
  if errorlevel 1 exit /b 1
)
"%PYTHON_BIN%" "%LAUNCHER%" summarize --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
