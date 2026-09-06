#!/usr/bin/env bash
set -e

echo "============================================"
echo " RAG AI Chatbot - Starting..."
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.10+."
    exit 1
fi

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install deps
echo "[SETUP] Installing dependencies..."
pip install -r requirements.txt --quiet

# Load .env defaults
HOST_VAL="127.0.0.1"
PORT_VAL="8000"
if [ -f ".env" ]; then
    while IFS='=' read -r key val; do
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        case "$key" in
            HOST) HOST_VAL="$val" ;;
            PORT) PORT_VAL="$val" ;;
        esac
    done < .env
fi

echo ""
echo "[INFO] Starting server at http://${HOST_VAL}:${PORT_VAL}"
echo "[INFO] Open your browser and go to: http://${HOST_VAL}:${PORT_VAL}"
echo "[INFO] Press Ctrl+C to stop."
echo ""

cd backend
python -m uvicorn main:app --host "${HOST_VAL}" --port "${PORT_VAL}" --reload
