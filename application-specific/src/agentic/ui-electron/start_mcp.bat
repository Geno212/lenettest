@echo off
REM Helper script to start MCP Server

set PROJECT_ROOT=C:\Users\Ahmed hosam\Desktop\grad\Application-Specific-Deep-Learning-Accelerator-Designer
set VENV_PYTHON=%PROJECT_ROOT%\trial\Scripts\python.exe
set PYTHONPATH=%PROJECT_ROOT%

REM Activate the virtual environment so the server runs in that environment
call "%PROJECT_ROOT%\trial\Scripts\activate.bat"

REM Run from project root so relative paths work
cd /d "%PROJECT_ROOT%"
python -m src.cli.mcp_server.main
pause
