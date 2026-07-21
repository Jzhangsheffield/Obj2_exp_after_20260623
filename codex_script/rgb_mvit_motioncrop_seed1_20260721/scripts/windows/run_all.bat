@echo off
setlocal EnableExtensions
for %%S in (00_validate.bat 01_compute_crop_stats.bat 02_pretrain_all.bat 03_classifier_all.bat 04_test_all.bat 05_features_all.bat 06_summarize.bat) do (
  echo ============================================================
  echo Running %%S
  call "%~dp0%%S"
  if errorlevel 1 exit /b 1
)
echo All experiment stages completed.
exit /b 0
