@echo off
REM ===========================================================================
REM  nsetrade - pull the latest code from GitHub and refresh dependencies.
REM  Double-click this any time you want the newest features on this laptop.
REM  (The app runs locally; this is how new changes "deploy" to your machine.)
REM ===========================================================================
setlocal
cd /d "%~dp0\.."

echo [nsetrade] Fetching the latest code from GitHub...
git pull
if errorlevel 1 (
    echo.
    echo ERROR: git pull failed. If you have local edits, commit or discard them
    echo        first ^(git status^). If git isn't installed, get it from
    echo        https://git-scm.com/download/win
    pause
    exit /b 1
)

if not exist .venv\Scripts\activate.bat (
    echo ERROR: virtual environment not found. Run scripts\setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo [nsetrade] Refreshing dependencies (fast if nothing changed)...
pip install -e ".[yfinance,charts,dashboard,kite,ai]" >nul

echo.
echo [nsetrade] Up to date. Launch the dashboard with scripts\run_dashboard.bat
echo.
pause
