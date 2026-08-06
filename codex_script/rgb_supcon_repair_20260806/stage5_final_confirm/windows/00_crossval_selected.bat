@echo off
call "%~dp0..\..\scripts\windows\common_env.bat"
if not defined CV_STAGE set "CV_STAGE=stage2"
if not defined CV_EXPERIMENT set "CV_EXPERIMENT=s2_m3_dual"
for %%F in (fold_M fold_J fold_MR) do (
  if "%CV_STAGE%"=="stage4" "%PYTHON_BIN%" "%LAUNCHER%" cache_teacher --split-profile %%F --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
  if errorlevel 1 exit /b 1
  "%PYTHON_BIN%" "%LAUNCHER%" pretrain --stage %CV_STAGE% --experiment %CV_EXPERIMENT% --split-profile %%F --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
  if errorlevel 1 exit /b 1
  "%PYTHON_BIN%" "%LAUNCHER%" diagnose --stage %CV_STAGE% --experiment %CV_EXPERIMENT% --split-profile %%F --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
  if errorlevel 1 exit /b 1
)
"%PYTHON_BIN%" "%LAUNCHER%" summarize --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
