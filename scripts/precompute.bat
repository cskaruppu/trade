@echo off
REM ===========================================================================
REM  nsetrade - precompute the opportunity ranking into the local cache.
REM  Run this nightly so the dashboard opens with fresh rankings instantly.
REM
REM  Schedule it: open Task Scheduler -> Create Basic Task -> Daily (e.g. 7pm)
REM  -> Start a program -> point to this file. The dashboard's Opportunities
REM  tab can then read the cached scan with one click.
REM
REM  Edit the universe/side below to taste. Use nse_all for the full NSE list
REM  (run scripts\nsetrade.bat refresh-universe once first).
REM ===========================================================================
setlocal
cd /d "%~dp0\.."
if not exist .venv\Scripts\activate.bat (
    echo ERROR: virtual environment not found. Run scripts\setup.bat first.
    exit /b 1
)
call .venv\Scripts\activate.bat

echo [nsetrade] Precomputing opportunities + pattern picks for nifty100...
nsetrade precompute --universe nifty100 --side long --with-patterns

echo [nsetrade] Updating the track record (log new breakouts, resolve matured)...
nsetrade track evaluate
nsetrade track log --universe nifty100

echo [nsetrade] Done. Open the dashboard - cached scans load instantly, and the
echo            Track Record page fills in as signals mature.
