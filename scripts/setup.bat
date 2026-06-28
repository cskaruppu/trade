@echo off
REM ===========================================================================
REM  nsetrade - one-time setup for a private local install on Windows.
REM  Double-click this file, or run it from a terminal. It creates a virtual
REM  environment and installs everything (dashboard, charts, Kite, yfinance).
REM ===========================================================================
setlocal
cd /d "%~dp0\.."

echo.
echo [nsetrade] Creating virtual environment (.venv)...
py -3 -m venv .venv
if errorlevel 1 (
    echo ERROR: could not create venv. Is Python 3 installed and on PATH?
    echo Install it from https://www.python.org/downloads/ ^(tick "Add to PATH"^).
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo [nsetrade] Upgrading pip...
python -m pip install --upgrade pip >nul

echo [nsetrade] Installing nsetrade and all extras (this can take a minute)...
pip install -e ".[yfinance,charts,dashboard,kite,ai]"
if errorlevel 1 (
    echo ERROR: installation failed. See the messages above.
    pause
    exit /b 1
)

if not exist config.yaml (
    copy config.example.yaml config.yaml >nul
    echo [nsetrade] Created config.yaml from the template.
    echo            Edit it to add your Zerodha Kite api_key / access_token.
)

echo.
echo [nsetrade] Setup complete. Run the dashboard with:  scripts\run_dashboard.bat
echo.
pause
