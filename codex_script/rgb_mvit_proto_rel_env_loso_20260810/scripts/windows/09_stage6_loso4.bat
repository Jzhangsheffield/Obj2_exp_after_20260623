@echo off
call "%~dp0_env.bat"
for /L %%I in (0,1,5) do "%PYTHON_BIN%" "%PACKAGE_ROOT%\run.py" pipeline --stage stage6 --index %%I --fold all --outer-test --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%" || exit /b 1
