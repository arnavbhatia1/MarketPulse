@echo off
REM MarketPulse — one-command launcher for Windows
REM Usage: start.bat

cd /d "%~dp0"

echo [1/2] Starting financial-mcp server (SSE on :8520)...
REM financial-mcp defaults to stdio; the dashboard connects over SSE, so force it.
set FINANCIAL_MCP_TRANSPORT=sse
start /B financial-mcp --transport sse

timeout /t 5 /nobreak >nul

echo [2/2] Starting MarketPulse dashboard...
python -m streamlit run app/MarketPulse.py --server.port 8501
