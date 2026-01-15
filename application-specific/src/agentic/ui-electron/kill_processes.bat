@echo off
REM Kill any running MCP servers and Python processes related to the UI

echo Killing MCP server processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do (
    echo Killing process %%a
    taskkill /F /PID %%a 2>nul
)

echo.
echo Killing any chatbot_bridge.py processes...
for /f "tokens=2" %%a in ('tasklist ^| findstr /i "python.exe"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr /i "chatbot_bridge" >nul
    if not errorlevel 1 (
        echo Killing chatbot_bridge process %%a
        taskkill /F /PID %%a 2>nul
    )
)

echo.
echo All processes killed.
pause
