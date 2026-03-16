#!/usr/bin/env bash
# MarketPulse — one-command launcher
# Usage: bash start.sh

set -e
cd "$(dirname "$0")"

# Load env vars
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Start financial-mcp server in background (needed for Trading Bot)
echo "[1/2] Starting financial-mcp server..."
python -m financial_mcp.server &
MCP_PID=$!
sleep 3

# Start Streamlit
echo "[2/2] Starting MarketPulse dashboard..."
python -m streamlit run app/MarketPulse.py --server.port 8501

# Cleanup MCP server on exit
kill $MCP_PID 2>/dev/null
