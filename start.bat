@echo off
TITLE RAG Chatbot

echo ============================================
echo  RAG AI Chatbot - Starting...
echo ============================================
echo.
set KMP_DUPLICATE_LIB_OK=TRUE
REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Create virtual environment if missing
if not exist "venv\" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install/upgrade deps
echo [SETUP] Installing dependencies (first run may take a few minutes)...
python -m pip install -r requirements.txt --quiet

REM Load HOST / PORT from .env (only those two lines — avoids feeding
REM comment lines with parentheses, like "(runs fully on CPU)", into the
REM FOR loop, which confuses cmd.exe's block parser and throws
REM "The syntax of the command is incorrect.")
set "HOST_VAL=127.0.0.1"
set "PORT_VAL=8000"
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b /i /r "HOST= PORT=" ".env"`) do (
        if /i "%%A"=="HOST" set "HOST_VAL=%%B"
        if /i "%%A"=="PORT" set "PORT_VAL=%%B"
    )
)

echo.
echo [INFO] Starting server at http://%HOST_VAL%:%PORT_VAL%
echo [INFO] Open your browser and go to: http://%HOST_VAL%:%PORT_VAL%
echo [INFO] Press Ctrl+C to stop.
echo.

cd backend
python -m uvicorn main:app --host %HOST_VAL% --port %PORT_VAL% --reload

pause
