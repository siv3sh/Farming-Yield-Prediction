#!/bin/bash

# Quick start script for Streamlit Dashboard

echo "🌾 Starting Farming Yield Prediction Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Get port from command line argument or use default
PORT=${1:-8501}

# Run the dashboard
echo "🚀 Launching dashboard on port $PORT..."
echo "📊 Dashboard will be available at: http://localhost:$PORT"
streamlit run streamlit_dashboard.py --server.port $PORT

