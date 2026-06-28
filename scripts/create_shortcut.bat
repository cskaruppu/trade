@echo off
REM ===========================================================================
REM  nsetrade - create a Desktop shortcut to launch the dashboard with one
REM  double-click (no PowerShell, no typing). Run this ONCE.
REM
REM  Optional auto-start on login: copy the created "nsetrade Dashboard"
REM  shortcut into your Startup folder. Press Win+R, type  shell:startup ,
REM  press Enter, and paste the shortcut there.
REM ===========================================================================
setlocal
cd /d "%~dp0\.."
set "TARGET=%CD%\scripts\run_dashboard.bat"
set "ICONDIR=%CD%"
set "LINK=%USERPROFILE%\Desktop\nsetrade Dashboard.lnk"

powershell -NoProfile -Command ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%LINK%');" ^
  "$s.TargetPath='%TARGET%';" ^
  "$s.WorkingDirectory='%ICONDIR%';" ^
  "$s.Description='Launch the private nsetrade dashboard';" ^
  "$s.Save()"

if exist "%LINK%" (
    echo.
    echo [nsetrade] Created Desktop shortcut: "nsetrade Dashboard"
    echo            Double-click it any time to open the dashboard.
    echo.
    echo            To launch automatically when Windows starts, press Win+R,
    echo            type  shell:startup  , Enter, and copy that shortcut there.
) else (
    echo ERROR: could not create the shortcut.
)
echo.
pause
