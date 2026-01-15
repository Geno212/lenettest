@echo off
REM Helper script to start Flask Backend

set PROJECT_ROOT=d:\Siemens GP\Application-Specific-Deep-Learning-Accelerator-Designer
set VENV_PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe
set PYTHONPATH=%PROJECT_ROOT%

cd /d "%~dp0backend"
"%VENV_PYTHON%" server.py
pause
