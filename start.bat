@echo off
REM MarketPulse — one-command launcher for Windows
REM Usage: start.bat

cd /d "%~dp0"

echo [1/2] Starting financial-mcp server...
start /B python -m financial_mcp.server

timeout /t 3 /nobreak >nul

echo [2/2] Starting MarketPulse dashboard...
python -m streamlit run app/MarketPulse.py --server.port 8501
