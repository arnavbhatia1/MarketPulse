# MarketPulse launcher (PowerShell) — opens the dashboard in Chrome.
# Ctrl-C stops everything cleanly with NO "Terminate batch job (Y/N)?" prompt.
# Usage:  .\start.ps1

Set-Location $PSScriptRoot

# --- locate Chrome so the dashboard opens there, not the default browser ---
$chrome = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LocalAppData\Google\Chrome\Application\chrome.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

Write-Host "[1/3] Starting financial-mcp server (SSE on :8520)..."
$env:FINANCIAL_MCP_TRANSPORT = "sse"
$server = Start-Process -FilePath "financial-mcp" -ArgumentList "--transport", "sse" `
    -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 5

Write-Host "[2/3] Opening MarketPulse in Chrome..."
if ($chrome) {
    # Detached opener: waits for Streamlit to bind, then launches Chrome.
    Start-Process powershell -WindowStyle Hidden -ArgumentList `
        "-NoProfile", "-Command", "Start-Sleep -Seconds 5; Start-Process '$chrome' 'http://localhost:8501'"
} else {
    Write-Host "  Chrome not found - open http://localhost:8501 manually." -ForegroundColor Yellow
}

Write-Host "[3/3] Starting MarketPulse dashboard (Ctrl-C to stop)..."
try {
    python -m streamlit run app/MarketPulse.py --server.port 8501 --server.headless true
} finally {
    # Clean up the MCP server on exit so no orphan keeps port 8520.
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Stopped MarketPulse and the financial-mcp server."
}
