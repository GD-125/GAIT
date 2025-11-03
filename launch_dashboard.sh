#!/bin/bash

# Launch Gait Detection Dashboard

echo "===================================="
echo "  Gait Detection Dashboard Launcher"
echo "===================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/3] Checking dependencies..."

# Check if streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    pip3 install streamlit plotly openpyxl scikit-learn
else
    echo "Dependencies OK"
fi

echo ""
echo "[2/3] Checking model..."

# Check if model exists
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo ""
    echo "========================================"
    echo "  WARNING: Trained model not found!"
    echo "========================================"
    echo ""
    echo "Please train the model first by running:"
    echo "  python3 efficient_run.py"
    echo ""
    echo "Or use the original training script:"
    echo "  python3 main.py"
    echo ""
    exit 1
fi

echo "Model found: checkpoints/best_model.pt"

echo ""
echo "[3/3] Launching dashboard..."
echo ""
echo "========================================"
echo "  Dashboard will open in your browser"
echo "  Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Launch Streamlit
streamlit run dashboard_app.py
