@echo off
REM MarketPulse — one-command launcher for Windows (opens in Chrome)
REM Usage: start.bat
REM NOTE: For a clean Ctrl-C with no "Terminate batch job (Y/N)?" prompt,
REM       run  .\start.ps1  instead (that prompt is a cmd.exe limitation).

cd /d "%~dp0"

REM --- locate Chrome so the dashboard opens there, not the default browser ---
set "CHROME="
for %%P in (
  "%ProgramFiles%\Google\Chrome\Application\chrome.exe"
  "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
  "%LocalAppData%\Google\Chrome\Application\chrome.exe"
) do if exist "%%~P" set "CHROME=%%~P"

echo [1/3] Starting financial-mcp server (SSE on :8520)...
set FINANCIAL_MCP_TRANSPORT=sse
start /B financial-mcp --transport sse

timeout /t 5 /nobreak >nul

echo [2/3] Opening MarketPulse in Chrome...
if defined CHROME (
  start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process '%CHROME%' 'http://localhost:8501'"
) else (
  echo   Chrome not found in the usual spots - open http://localhost:8501 manually.
)

echo [3/3] Starting MarketPulse dashboard (Ctrl-C to stop)...
python -m streamlit run app/MarketPulse.py --server.port 8501 --server.headless true
