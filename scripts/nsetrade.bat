@echo off
REM ===========================================================================
REM  nsetrade - run any CLI command inside the local venv on Windows.
REM  Examples:
REM     scripts\nsetrade.bat analyse RELIANCE
REM     scripts\nsetrade.bat screen --universe nifty50 --top 15
REM     scripts\nsetrade.bat backtest RELIANCE --strategy rsi_ma --years 3
REM     scripts\nsetrade.bat chart INFY --out infy.png
REM ===========================================================================
setlocal
cd /d "%~dp0\.."
if not exist .venv\Scripts\activate.bat (
    echo ERROR: virtual environment not found. Run scripts\setup.bat first.
    exit /b 1
)
call .venv\Scripts\activate.bat
nsetrade %*
