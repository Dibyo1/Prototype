@echo off
echo.
echo ================================================================
echo                    TruthGuard - Fake News Detector
echo ================================================================
echo.
echo Starting Flask server...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

REM Check if in correct directory
if not exist "app.py" (
    echo ERROR: app.py not found in current directory
    echo Please make sure you're in the correct project directory
    pause
    exit /b 1
)

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import flask, flask_cors, requests, nltk, numpy, PIL, PyPDF2" >nul 2>&1
if errorlevel 1 (
    echo Installing missing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo.
echo ✅ All dependencies are installed
echo.
echo 🚀 Starting TruthGuard server...
echo.
echo After the server starts:
echo   • Open your browser
echo   • Go to: http://localhost:5000
echo   • Start fact-checking!
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask application
python app.py

pause