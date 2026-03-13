# CLAUDE.md — MarketPulse

Financial sentiment hub that tracks market mood across Reddit, Stocktwits, and financial news RSS. Users search any ticker to get a structured briefing card with per-source sentiment breakdown, 7-day trend, and a Claude-generated verdict.

---

## Architecture

```
Ingest (Reddit + Stocktwits + free News RSS)
        ↓
Keyword/emoji labeling functions (16 LFs) → confidence-weighted vote
        ↓
TF-IDF + LogReg classifies all posts (auto-trains when ≥200 labeled)
        ↓
Posts + per-ticker summaries saved to SQLite (data/marketpulse.db)
        ↓
Home page grid reads from SQLite (instant)
User searches ticker → Claude writes 2-3 sentence verdict → briefing card
        ↓
MCP Client (src/investor/mcp_client.py) connects to financial-mcp-server via SSE
        ↓
Trading Bot page: market regime, ticker analysis, portfolio management
  (portfolio state lives in data/financial_mcp.db, managed by MCP server)
```

---

## Project Structure

```
MarketPulse/
├── app/
│   ├── MarketPulse.py              # Home page: search bar + market grid + market intel teaser
│   ├── pipeline_runner.py          # refresh_pipeline(), get_ticker_cache(), load_model()
│   ├── pages/
│   │   ├── 1_Ticker_Detail.py      # Deep-dive page for a single ticker
│   │   └── 2_Trading_Bot.py        # Full trading terminal (market intel + ticker analysis + portfolio)
│   └── components/
│       ├── charts.py               # Plotly chart components (pie, bar, trend, probability)
│       ├── styles.py               # Dark theme colors, animations, and CSS
│       └── trading_charts.py       # Candlestick, score gauge, stress gauge, CFTC bars, sector bars
├── src/
│   ├── ingestion/
│   │   ├── base.py                 # Abstract base ingester + REQUIRED_COLUMNS schema
│   │   ├── reddit.py               # PRAW-based Reddit ingester
│   │   ├── stocktwits.py           # Stocktwits API ingester
│   │   ├── news.py                 # Free RSS ingester (Google News + Yahoo Finance + CNBC + MarketWatch)
│   │   └── manager.py              # Orchestrates all sources; raises if all fail
│   ├── labeling/
│   │   ├── functions.py            # 16 keyword/emoji/structural labeling functions
│   │   └── aggregator.py           # Confidence-weighted vote aggregation
│   ├── models/
│   │   └── pipeline.py             # TF-IDF + LogReg training and inference
│   ├── extraction/
│   │   ├── ticker_extractor.py     # Cashtag, bare ticker, company name extraction
│   │   └── normalizer.py           # Canonical company name normalization
│   ├── analysis/
│   │   └── ticker_sentiment.py     # Per-ticker aggregation from labeled posts
│   ├── investor/
│   │   └── mcp_client.py           # MCP client — SSE connection to financial-mcp-server (21 tools)
│   ├── storage/
│   │   └── db.py                   # SQLite read/write (data/marketpulse.db) — sentiment tables only
│   ├── agent/
│   │   └── briefing.py             # Claude synthesis — one API call per search
│   └── utils/
│       ├── config.py               # YAML config loader
│       ├── logger.py               # Structured logging
│       └── cache.py                # API response cache
├── scripts/
│   ├── run_pipeline.py             # CLI: full pipeline end-to-end
│   ├── ingest.py                   # CLI: ingest only
│   ├── label.py                    # CLI: label only
│   └── train.py                    # CLI: train only
├── config/
│   └── default.yaml                # Data sources, model hyperparameters, MCP server config
├── tests/                          # pytest test suite
├── data/
│   ├── marketpulse.db              # SQLite — sentiment data (gitignored)
│   ├── financial_mcp.db            # SQLite — portfolio state, managed by MCP server (gitignored)
│   ├── raw/                        # Raw ingested CSVs (gitignored)
│   ├── labeled/                    # Labeled CSVs (gitignored)
│   └── models/                     # Trained model artifacts (gitignored)
└── requirements.txt
```

---

## Key Modules

### `src/ingestion/base.py`
Defines `REQUIRED_COLUMNS` that every ingester must return:
`post_id, text, source, timestamp, author, score, url, metadata`

### `src/ingestion/news.py`
Free RSS ingester — **no API key needed**. Pulls from Google News RSS, Yahoo Finance RSS (20 tickers), CNBC Markets, and MarketWatch using `feedparser`. `is_available()` always returns `True`. Posts are deduplicated by normalized URL (tracking params stripped), filtered by date range. Entries with unparseable dates are skipped. Per-entry error handling ensures one bad feed entry doesn't kill the whole feed.

### `src/ingestion/manager.py`
Tries all three sources. Skips sources where `is_available()` is False. Raises `RuntimeError` if no data is collected (news RSS should always succeed). No synthetic fallback.

### `src/labeling/functions.py`
16 labeling functions, each voting `bullish | bearish | neutral | meme | ABSTAIN`:
- Keyword-based: `lf_keyword_bullish/bearish/neutral/meme`
- Emoji-based: `lf_emoji_bullish/bearish/meme`
- Structural: `lf_question_structure`, `lf_short_post`, `lf_all_caps_ratio`
- Financial: `lf_options_directional`, `lf_price_target_mention`, `lf_loss_reporting`, `lf_news_language`
- Sarcasm: `lf_sarcasm_indicators`, `lf_self_deprecating`
- Metadata-aware (used when metadata present): `lf_stocktwits_user_sentiment`, `lf_reddit_flair`

### `src/labeling/aggregator.py`
`LabelAggregator` runs all functions then applies confidence-weighted voting. Posts below `confidence_threshold` (default 0.35) get `programmatic_label=None`. Config: `labeling.confidence_threshold`, `labeling.min_votes`.

### `src/models/pipeline.py`
`SentimentPipeline`: TF-IDF (500 features, 1-2 ngrams) + balanced LogReg. `train(texts, labels)` returns a metrics report. `predict(texts)` returns list of `{label, confidence, probabilities}`. `save(dir)` / `load(dir)` persist to `data/models/`.

### `src/investor/mcp_client.py`
MCP client that connects to `financial-mcp-server` over SSE. Maintains a persistent connection in a background daemon thread. Exposes 21 sync wrapper functions grouped into three categories:
- **Scoring & Analysis**: `score_ticker`, `scan_universe`, `analyze_ticker`, `get_fundamentals`, `get_momentum`, `get_price`
- **Portfolio & Trading**: `create_portfolio`, `analyze_portfolio`, `get_holdings`, `get_trades`, `execute_buy`, `execute_sell`, `run_rebalance`, `check_risk`
- **Market Intelligence**: `detect_market_regime`, `get_vix_analysis`, `scan_anomalies`, `scan_volume_leaders`, `scan_gap_movers`, `get_smart_money_signal`, `get_futures_positioning`

`call_tool(name, timeout, **kwargs)` is the core dispatcher. `is_connected()` checks if the background thread is alive and the SSE session is established.

### `src/storage/db.py`
Single SQLite file at `data/marketpulse.db`. Three tables (sentiment data only):
- `posts` — all ingested posts with sentiment and tickers
- `ticker_cache` — per-ticker aggregated sentiment (the market grid source)
- `model_training_log` — model training history

Portfolio state (portfolios, holdings, trades, snapshots) lives in `data/financial_mcp.db`, managed entirely by the `financial-mcp-server`.

### `src/agent/briefing.py`
`generate_briefing(company, ticker, ticker_data)` calls `claude-sonnet-4-6` with `max_tokens=150` to write a 2-3 sentence verdict. Returns a static fallback string if `ANTHROPIC_API_KEY` is unset or the call fails.

### `app/pipeline_runner.py`
- `refresh_pipeline(start_date_str, end_date_str, progress_callback)` — full pipeline run with optional stage-level progress reporting
- `get_ticker_cache()` — reads SQLite, cached 60s via `@st.cache_data`
- `load_model()` — loads trained model, cached via `@st.cache_resource`
- `_maybe_train_model(df, config)` — auto-trains if ≥200 labeled posts and no model exists yet

---

## SQLite Schema

### `data/marketpulse.db` (sentiment data)

```sql
posts (
    post_id TEXT PRIMARY KEY, text TEXT, source TEXT, timestamp TEXT,
    author TEXT, score INTEGER, tickers TEXT,   -- JSON array
    sentiment TEXT, confidence REAL, url TEXT
)

ticker_cache (
    ticker TEXT PRIMARY KEY,    -- company name key
    symbol TEXT,                -- e.g. "TSLA"
    last_updated TEXT,
    dominant_sentiment TEXT,
    mention_count INTEGER,
    avg_confidence REAL,
    reddit_sentiment TEXT,
    news_sentiment TEXT,
    stocktwits_sentiment TEXT,
    sentiment_by_day TEXT,      -- JSON: {"2026-03-01": "bullish", ...}
    top_posts TEXT              -- JSON: {source: [post snippets]}
)

model_training_log (
    run_id TEXT PRIMARY KEY, trained_at TEXT,
    num_samples INTEGER, weighted_f1 REAL, label_source TEXT
)
```

### `data/financial_mcp.db` (portfolio state)

Managed entirely by the `financial-mcp-server`. Contains portfolios, holdings, trades, snapshots, and ETF universe data. MarketPulse does not read or write this database directly -- all access goes through MCP tool calls via `src/investor/mcp_client.py`.

---

## Configuration (`config/default.yaml`)

```yaml
data:
  mode: "auto"          # "live" | "auto" — no synthetic mode

labeling:
  aggregation_strategy: "confidence_weighted"
  confidence_threshold: 0.35
  min_votes: 1

model:
  max_features: 500
  ngram_range: [1, 2]
  C: 1.0
  class_weight: "balanced"

mcp_server:
  url: "http://localhost:8520/sse"
  timeout: 30
  rebalance_timeout: 120
```

---

## Environment Variables (`.env`)

```
ANTHROPIC_API_KEY=...       # Required for AI verdict; falls back to static text if absent
REDDIT_CLIENT_ID=...        # Optional — Reddit posts
REDDIT_CLIENT_SECRET=...    # Optional — Reddit posts
STOCKTWITS_ACCESS_TOKEN=... # Optional — Stocktwits messages
```

News is free via RSS — no key needed.

The `financial-mcp-server` must be running on `localhost:8520` (configurable in `config/default.yaml`) for the Trading Bot page to function. The sentiment pipeline and home page work independently of the MCP server.

---

## Running Locally

```bash
pip install -r requirements.txt
cp .env.example .env   # add your keys
financial-mcp &        # start MCP server (required for Trading Bot page)
python3 -m streamlit run app/MarketPulse.py
```

## Running the CLI Pipeline

```bash
python3 scripts/run_pipeline.py           # 7-day lookback
python3 scripts/run_pipeline.py --days 30 # 30-day lookback
python3 -m pytest tests/ -v               # run all 166 tests
```

---

## Deployment Notes

- **No external database** — SQLite files at `data/marketpulse.db` (sentiment) and `data/financial_mcp.db` (portfolio). Mount a persistent volume at `/app/data/` in production.
- **No required API keys** — news RSS always provides data. Reddit and Stocktwits are optional.
- **Model auto-trains** on first Refresh if ≥200 labeled posts are collected. Model lives at `data/models/`.
- **MCP server** — the `financial-mcp-server` process must be running for the Trading Bot page. Start with `financial-mcp`. The sentiment pipeline and home page work without it.
- **Port** — Streamlit defaults to 8501. Set with `--server.port` or `STREAMLIT_SERVER_PORT` env var. MCP server runs on port 8520 by default.
- **Memory** — peak usage during ingestion is modest (feedparser + sklearn). 512MB RAM is sufficient.

---

## Agent Team

4 project-specific AI agents at `.claude/agents/`. All share persistent memory at `.claude/agent-memory/shared/`.

| Agent | Role | Scope | Color |
|-------|------|-------|-------|
| `mp-ingestion` | Data acquisition specialist | `src/ingestion/`, `src/extraction/`, `src/storage/`, `src/utils/`, `scripts/ingest.py`, `scripts/run_pipeline.py` | blue |
| `mp-sentiment` | ML & NLP specialist | `src/labeling/`, `src/models/`, `src/analysis/`, `scripts/label.py`, `scripts/train.py` | magenta |
| `mp-dashboard` | UI/UX specialist | `app/`, `src/agent/briefing.py` | green |
| `mp-qa` | Quality gatekeeper | `tests/` (owns), all source (read-only review) | yellow |

**Behavioral rules**: All agents propose before executing, check shared memory at task start, stay within their scope, and verify work with evidence before claiming done. Cross-boundary tasks are flagged for the appropriate agent; the user orchestrates handoffs.
