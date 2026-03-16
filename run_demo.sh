#!/bin/bash
# Run the IoT Device Classification Demo

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Set library path for XGBoost
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

# Run Streamlit app
echo "🚀 Starting IoT Device Classifier Demo..."
echo "   Open http://localhost:8501 in your browser"
echo ""
streamlit run demo/app.py --server.headless true
