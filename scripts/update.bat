@echo off
REM ===========================================================================
REM  nsetrade - force-sync the latest code from GitHub and refresh deps.
REM  Double-click this any time to get the newest features on this laptop.
REM
REM  This does a HARD reset to match GitHub exactly, so a partial/blocked
REM  'git pull' can never leave you on a half-updated copy. It discards local
REM  edits to TRACKED files only; your config.yaml and watchlist.txt are
REM  git-ignored and are never touched.
REM ===========================================================================
setlocal
cd /d "%~dp0\.."

echo [nsetrade] Fetching the latest code from GitHub...
git fetch origin
if errorlevel 1 (
    echo.
    echo ERROR: git fetch failed. Check your internet connection. If git isn't
    echo        installed, get it from https://git-scm.com/download/win
    pause
    exit /b 1
)

REM determine the current branch and hard-reset it to the remote
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%b"
echo [nsetrade] Force-syncing branch "%BRANCH%" to origin/%BRANCH%...
git reset --hard origin/%BRANCH%
if errorlevel 1 (
    echo.
    echo ERROR: could not reset to origin/%BRANCH%. Run 'git status' to inspect.
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
echo [nsetrade] Up to date with GitHub. Verifying the package imports...
python -c "import nsetrade.universe as u; assert hasattr(u, 'refresh_nse_equity_list'); import nsetrade.bhavcopy; print('  OK - full NSE + bhavcopy available')"
echo.
echo [nsetrade] Done. Launch the dashboard with scripts\run_dashboard.bat
echo.
pause
