@echo off
REM Launch Agentic NN Generator Electron UI
REM This script starts ONLY the MCP server and Electron app
REM The Electron app will start the chatbot bridge directly (no Flask server needed)

echo ========================================
echo Agentic NN Generator - Electron UI
echo ========================================
echo.

REM Kill any existing MCP servers on port 8000
echo Checking for existing MCP servers...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing existing MCP server process %%a
    taskkill /F /PID %%a 2>nul
)
timeout /t 2 /nobreak >nul
echo.

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

REM Check if npm dependencies are installed
if not exist "node_modules" (
    echo Installing Node.js dependencies...
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install Node.js dependencies
        pause
        exit /b 1
    )
) else (
    REM Check if electron is installed
    where electron >nul 2>&1
    if errorlevel 1 (
        echo Electron not found. Installing dependencies...
        call npm install
        if errorlevel 1 (
            echo ERROR: Failed to install Node.js dependencies
            pause
            exit /b 1
        )
    )
)

echo.
echo Starting services...
echo.

REM Set the project root path
set PROJECT_ROOT=C:\Users\Ahmed hosam\Desktop\grad\Application-Specific-Deep-Learning-Accelerator-Designer
set VENV_PYTHON=%PROJECT_ROOT%\trial\Scripts\python.exe

REM Check if virtual environment exists
if not exist "%VENV_PYTHON%" (
    echo ERROR: Virtual environment not found at: %VENV_PYTHON%
    echo Please ensure venv exists in the project root
    pause
    exit /b 1
)

echo Using virtual environment at: %PROJECT_ROOT%\trial
echo.

REM Start MCP Server in a new window
echo Starting MCP Server...
start "MCP Server" cmd /k "%~dp0start_mcp.bat"
echo Waiting for MCP Server to start (10 seconds)...
timeout /t 10 /nobreak >nul

REM NOTE: No Flask server needed! 
REM The Electron app will start chatbot_bridge.py directly via stdio
echo.
echo Chatbot bridge will be started by Electron (no Flask server needed)
echo.

REM Start Electron app
echo Starting Electron UI...
call npm start

REM This will wait until Electron closes

echo.
echo Electron UI closed.
echo Note: MCP Server is still running in its window.
echo Close the MCP Server window manually if needed.
pause
