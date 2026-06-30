@echo off
setlocal EnableExtensions

REM ============================================================
REM Run two MindRove .bat scripts sequentially
REM The second script runs only if the first one succeeds.
REM ============================================================

set "SCRIPT_DIR=%~dp0"

set "BAT_1=%SCRIPT_DIR%finetune_mindrove_N_except_take_put_adamw_44_seed1.bat"
set "BAT_2=%SCRIPT_DIR%finetune_mindrove_N_take_put_adamw_44_seed1.bat"

if not exist "%BAT_1%" (
    echo [ERROR] Cannot find first script:
    echo         "%BAT_1%"
    exit /b 1
)

if not exist "%BAT_2%" (
    echo [ERROR] Cannot find second script:
    echo         "%BAT_2%"
    exit /b 1
)

echo ============================================================
echo [1/2] Running:
echo "%BAT_1%"
echo ============================================================
call "%BAT_1%"

if errorlevel 1 (
    echo.
    echo [ERROR] First script failed. Stop here.
    echo Failed script: "%BAT_1%"
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo [2/2] Running:
echo "%BAT_2%"
echo ============================================================
call "%BAT_2%"

if errorlevel 1 (
    echo.
    echo [ERROR] Second script failed.
    echo Failed script: "%BAT_2%"
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo [DONE] Both scripts finished successfully.
echo ============================================================

endlocal
exit /b 0
