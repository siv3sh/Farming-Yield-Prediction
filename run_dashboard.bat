@echo off
REM Quick start script for Streamlit Dashboard (Windows)

echo 🌾 Starting Farming Yield Prediction Dashboard...
echo.

REM Get port from command line argument or use default
if "%1"=="" (
    set PORT=8501
) else (
    set PORT=%1
)

REM Run the dashboard
echo 🚀 Launching dashboard on port %PORT%...
echo 📊 Dashboard will be available at: http://localhost:%PORT%
streamlit run streamlit_dashboard.py --server.port %PORT%

pause

