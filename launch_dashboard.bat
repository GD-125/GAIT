@echo off
REM Launch Gait Detection Dashboard

echo ====================================
echo   Gait Detection Dashboard Launcher
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing dependencies...
    pip install streamlit plotly openpyxl scikit-learn
) else (
    echo Dependencies OK
)

echo.
echo [2/3] Checking model...

REM Check if model exists
if not exist "checkpoints\best_model.pt" (
    echo.
    echo ========================================
    echo   WARNING: Trained model not found!
    echo ========================================
    echo.
    echo Please train the model first by running:
    echo   python efficient_run.py
    echo.
    echo Or use the original training script:
    echo   python main.py
    echo.
    pause
    exit /b 1
)

echo Model found: checkpoints\best_model.pt

echo.
echo [3/3] Launching dashboard...
echo.
echo ========================================
echo   Dashboard will open in your browser
echo   Press Ctrl+C to stop the server
echo ========================================
echo.

REM Launch Streamlit
streamlit run dashboard_app.py

pause
