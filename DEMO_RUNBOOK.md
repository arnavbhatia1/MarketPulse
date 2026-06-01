# Demo Runbook — showing off `financial-mcp-server` in MarketPulse

**The point of the video:** your published MCP server (`financial-mcp-server`) gives
any AI app live market data + paper-trading tools. MarketPulse is a real app that
uses it — and an autonomous bot runs *on* your server, scanning the market and
placing trades by itself. The server is the engine; the bot is the proof.

---

## Setup

```powershell
cd C:\Users\abhat\Personal\MarketPulse
python scripts/seed_demo.py --reset   # start the bot cold so everything shown is real
.\start.ps1                           # launches MCP server (SSE :8520) + dashboard, opens Chrome
```

`.\start.ps1` opens the dashboard in **Chrome** and stops cleanly on Ctrl-C (no Y/N prompt).
Record **locally** (home IP) — yfinance is reliable there; on cloud it can be throttled.

---

## Recording (3 scenes)

**1. Sentiment page (~15s)** — lands first. "Free news RSS → sentiment, no API keys."
Click a ticker tile to show the breakdown popup.

**2. Trading Bot page (~45s)** — left sidebar. This is where the server shines:
- **Regime banner + VIX** at the top → "live from my MCP server."
- Type `AAPL` → chart, score gauge, Fundamentals / Momentum / Smart-Money cards →
  "each card is a different tool call to the server."

**3. The bot (~90s, the star)** — scroll to **Bot Control**, click **Start Bot**.
- Narrate the loop: scans the market through the MCP server → scores each stock →
  sizes with Kelly math → places paper trades.
- Within ~40–60s the cycle counter ticks, **Open Positions** fills, and the
  **Activity Log** streams BUY lines. Let it run a couple of cycles to show SELLs
  (hard stop, trailing stop, profit taking) and the Edge Statistics filling in live.

**Close (~15s):** "All of this ran on one `pip install financial-mcp-server`, over
standard MCP — so Claude Desktop, Cursor, or any agent can use the same 33 tools."

---

## Facts to get right
- The server exposes **33 MCP tools**; **no API keys required** (yfinance, SEC, CFTC, Treasury).
- A cycle takes ~40–60s — mostly live scoring of ~25 candidates.
- Everything is **paper trading** on a local SQLite DB.

## If something's off
| Symptom | Fix |
|---------|-----|
| `start.ps1` won't run | `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| "MCP server not running" | Confirm port 8520 is listening; the server must run with `--transport sse` (the launchers do this). |
| Bot opens no positions | Market may be closed, or scores are below the threshold — let a cycle or two run, ideally during US market hours. |
| Prices/scores show N/A on the deployed app | Cloud-host yfinance throttling — works locally. Record locally; use the cloud link as a "try it" CTA. |
