@echo off
call "%~dp0_env.bat"
for /L %%I in (0,1,1) do "%PYTHON_BIN%" "%PACKAGE_ROOT%\run.py" pipeline --stage stage1 --index %%I --fold %SCREEN_FOLD% --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%" || exit /b 1
