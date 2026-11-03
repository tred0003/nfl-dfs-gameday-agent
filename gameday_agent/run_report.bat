@echo off
REM Wrapper for Windows Task Scheduler.
REM
REM Exists for two reasons the .py cannot handle itself:
REM   1. Task Scheduler captures no output. Without the redirect below, an
REM      unattended failure leaves nothing behind to read on Monday.
REM   2. It pins the ffml conda env explicitly. Task Scheduler does not inherit
REM      your shell's PATH, so "python" alone may resolve to base -- which has
REM      no google-adk installed and would fail at import.
REM
REM Any arguments are passed through:  run_report.bat --dry-run

if not defined NFL_DFS_PYTHON set "NFL_DFS_PYTHON=python"
set "PY=%NFL_DFS_PYTHON%"
set "PROJ=%~dp0.."
set "LOG=%PROJ%\data\gameday_report.log"

cd /d "%PROJ%"
echo. >> "%LOG%"
echo ================ %DATE% %TIME% ================ >> "%LOG%"
"%PY%" "%PROJ%\gameday_agent\run_report.py" %* >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"
exit /b %ERRORLEVEL%
