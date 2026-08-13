@echo off
setlocal EnableExtensions
call "%~dp0_env.bat"

for /L %%I in (0,1,1) do call :run_test stage1 %%I || exit /b 1
for /L %%I in (0,1,10) do call :run_test stage2a %%I || exit /b 1
for /L %%I in (0,1,5) do call :run_test stage3a %%I || exit /b 1
for /L %%I in (0,1,10) do call :run_test stage4 %%I || exit /b 1

"%PYTHON_BIN%" "%PACKAGE_ROOT%\run.py" summarize --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
if errorlevel 1 exit /b 1

echo Completed 30 fold_MR outer-test evaluations.
echo Summary: %PROJECT_ROOT%\results\rgb_mvit_pr_env_loso_20260810\summary\SUMMARY.md
exit /b 0

:run_test
echo ===== TEST fold_MR %~1 index=%~2 =====
"%PYTHON_BIN%" -u "%PACKAGE_ROOT%\run.py" test --stage %~1 --index %~2 --fold MR --platform windows --project-root "%PROJECT_ROOT%" --dataset-root "%DATASET_ROOT%"
exit /b %errorlevel%
