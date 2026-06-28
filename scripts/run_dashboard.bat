@echo off
REM ===========================================================================
REM  nsetrade - launch the PRIVATE local dashboard on Windows.
REM  Binds to localhost only (see .streamlit\config.toml) so nothing is exposed
REM  to your network. Open http://localhost:8501 in your browser.
REM  Press Ctrl+C in this window to stop the server.
REM ===========================================================================
setlocal
cd /d "%~dp0\.."

if not exist .venv\Scripts\activate.bat (
    echo ERROR: virtual environment not found. Run scripts\setup.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo [nsetrade] Starting dashboard at http://localhost:8501  (Ctrl+C to stop)
REM Open the browser a few seconds after the server starts (headless mode is on).
start "" /b cmd /c "timeout /t 4 >nul & start http://localhost:8501"
streamlit run dashboard\app.py
pause
