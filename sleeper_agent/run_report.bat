@echo off
REM Wrapper for Windows Task Scheduler. Same shape as gameday_agent's, and for
REM the same two reasons: Task Scheduler captures no output, and it does not
REM inherit your shell's PATH (so plain "python" resolves to base, which has no
REM google-adk installed).
REM
REM Any arguments pass through:  run_report.bat --dry-run

set "PY=C:\Users\tred0\miniconda3\envs\ffml\python.exe"
set "PROJ=C:\new_career2026\nfl-dfs"
set "LOG=%PROJ%\data\sleeper_report.log"

cd /d "%PROJ%"
echo. >> "%LOG%"
echo ================ %DATE% %TIME% ================ >> "%LOG%"
"%PY%" "%PROJ%\sleeper_agent\run_report.py" %* >> "%LOG%" 2>&1
echo exit code: %ERRORLEVEL% >> "%LOG%"
exit /b %ERRORLEVEL%
