@echo off
REM Quick Setup Script for Agentic NN Generator UI
REM This script helps you set up the .env file

echo ========================================
echo Agentic NN Generator - Quick Setup
echo ========================================
echo.

REM Check if .env already exists
if exist "backend\.env" (
    echo .env file already exists!
    echo.
    choice /C YN /M "Do you want to overwrite it"
    if errorlevel 2 goto :skip_env
)

REM Copy .env.example to .env
echo Creating backend\.env file...
copy "backend\.env.example" "backend\.env" >nul
echo.

echo ========================================
echo IMPORTANT: Configure Your API Key
echo ========================================
echo.
echo The .env file has been created at:
echo   backend\.env
echo.
echo You MUST edit this file and add your OpenAI API key!
echo.
echo Opening .env file in notepad...
timeout /t 2 /nobreak >nul
notepad "backend\.env"

:skip_env
echo.
echo ========================================
echo Installing Dependencies
echo ========================================
echo.

echo Installing Node.js dependencies...
call npm install
if errorlevel 1 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)

echo.
echo Installing Python dependencies...
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Make sure you added your OPENAI_API_KEY to backend\.env
echo   2. Run: .\launch.bat
echo.
pause
