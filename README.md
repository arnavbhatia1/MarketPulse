# MarketPulse

Financial sentiment hub that tracks market mood across Reddit, Stocktwits, and financial news. Search any ticker to get a structured briefing card — per-source sentiment breakdown, 7-day trend, and a 2-3 sentence AI verdict synthesized by Claude.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill in your API keys
cp .env.example .env
# edit .env with your keys (see API Keys section below)

# 3. Launch the dashboard
streamlit run app/MarketPulse.py
```

Opens at **http://localhost:8501**. On first load the market grid is empty — click **Refresh Data** in the sidebar to ingest and analyze posts (~30 seconds).

## API Keys

Everything is optional. The app works without any keys using synthetic data, and without `ANTHROPIC_API_KEY` using a static fallback message.

| Key | What it enables | Where to get it |
|-----|----------------|-----------------|
| `ANTHROPIC_API_KEY` | AI Verdict on briefing cards | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | Reddit/WSB posts | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) → create script app |
| `STOCKTWITS_ACCESS_TOKEN` | Stocktwits messages | [api.stocktwits.com/developers](https://api.stocktwits.com/developers) |

**News is scraped for free** via Google News RSS and Yahoo Finance RSS — no API key needed.

Set these in your `.env` file (copied from `.env.example`). The app loads it automatically on startup.

## How It Works

```
Ingest (Reddit + Stocktwits + NewsAPI, or synthetic fallback)
        ↓
Keyword majority vote → training labels
        ↓
TF-IDF + LogReg classifies all posts (auto-trains when enough data)
        ↓
Posts + per-ticker summaries saved to SQLite (data/marketpulse.db)
        ↓
Home page grid reads from SQLite (instant)
User searches ticker → Claude writes 2-3 sentence verdict → briefing card
```

## Using the App

**Search a ticker:** Enter a symbol (`TSLA`) or company name (`Tesla`) and click **Research**. The briefing card renders inline with sentiment by source, a 7-day trend chart, and top post snippets.

**Refresh data:** Click **Refresh Data** in the sidebar. Use the date range picker to control the lookback window (up to 30 days).

**Market grid:** Shows all tracked tickers color-coded by dominant sentiment. Updates after each refresh.

## Project Structure

```
MarketPulse/
├── app/
│   ├── MarketPulse.py          # Home page: search bar + market grid
│   ├── pipeline_runner.py      # refresh_pipeline(), get_ticker_cache(), load_model()
│   ├── pages/
│   │   └── 1_Ticker_Detail.py  # Deep-dive page for a single ticker
│   └── components/             # Charts, metrics, CSS styles
├── src/
│   ├── ingestion/              # Reddit, Stocktwits, NewsAPI, Synthetic
│   ├── labeling/               # 16 keyword/emoji/options labeling functions
│   ├── models/                 # TF-IDF + LogReg training pipeline
│   ├── extraction/             # Ticker entity extraction + normalization
│   ├── analysis/               # Per-ticker sentiment aggregation
│   ├── storage/db.py           # SQLite read/write (data/marketpulse.db)
│   └── agent/briefing.py       # Claude synthesis — one API call per search
├── scripts/run_pipeline.py     # CLI: full pipeline end-to-end
├── config/default.yaml         # Data sources, model hyperparameters
└── tests/                      # 164 tests
```

## CLI

```bash
# Full pipeline (ingest → label → train → analyze → SQLite)
python scripts/run_pipeline.py

# Synthetic data only (no API keys needed)
python scripts/run_pipeline.py --synthetic

# Tests
pytest tests/ -v
```

## Tech Stack

Python 3.9+ · Streamlit · scikit-learn · Plotly · Anthropic SDK · SQLite · PRAW · pandas
