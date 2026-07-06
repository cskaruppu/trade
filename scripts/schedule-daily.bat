@echo off
REM ===========================================================================
REM  EdgeForge - register a DAILY auto-refresh with Windows Task Scheduler.
REM
REM  Double-click this ONCE. It schedules scripts\precompute.bat to run every
REM  day at 7:00 PM, which refreshes the cached opportunity/pattern scans and
REM  updates the honest Track Record. Your laptop must be on (and awake) at that
REM  time for it to run; missed runs execute at the next login.
REM
REM  Change the time by editing /ST below (24h HH:MM). To remove the schedule:
REM      schtasks /Delete /TN "EdgeForge Daily Refresh" /F
REM ===========================================================================
setlocal

set "TASK=EdgeForge Daily Refresh"
set "RUN=%~dp0precompute.bat"
set "WHEN=19:00"

echo [EdgeForge] Registering a daily refresh at %WHEN% ...
schtasks /Create /SC DAILY /TN "%TASK%" /TR "\"%RUN%\"" /ST %WHEN% /F
if errorlevel 1 (
    echo.
    echo ERROR: could not create the scheduled task. Try running this file as
    echo        Administrator ^(right-click -^> Run as administrator^).
    pause
    exit /b 1
)

echo.
echo [EdgeForge] Done. A daily task "%TASK%" now runs precompute.bat at %WHEN%.
echo            Verify/adjust it anytime in Task Scheduler (taskschd.msc).
echo            To run it right now once:  schtasks /Run /TN "%TASK%"
pause
