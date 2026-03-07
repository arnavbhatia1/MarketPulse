# MarketPulse: Sentiment Hub + Research Agent — Design Doc
**Date:** 2026-03-07
**Status:** Approved

---

## Overview

Pivot MarketPulse from an academic weak-supervision ML pipeline into a focused financial sentiment hub with an AI research agent. Anyone can come to the app, see overall market sentiment across tracked tickers, and research any ticker to get a structured briefing card synthesized by Claude.

**Core thesis (updated):** Real ingestion + ML classification + AI synthesis = a genuinely useful financial intelligence tool.

---

## What Changes

### Remove
- Snorkel/weak supervision aggregation machinery (`aggregator.py` confidence-weighted voting)
- ML training evaluation framework (`src/evaluation/`)
- Gold standard CSV and synthetic data complexity
- Model versioning system (`src/models/versioning.py`)
- Dashboard pages: Labeling Studio, Under the Hood, Live Inference

### Keep
- All 3 ingestion sources: `reddit.py`, `stocktwits.py`, `news.py`
- `ticker_extractor.py` + `normalizer.py`
- Labeling `functions.py` — simplified to plain majority vote (used only for generating training labels)
- `src/models/pipeline.py` — TF-IDF + Logistic Regression sentiment model

### Add
- `src/storage/db.py` — SQLite read/write layer
- `src/agent/briefing.py` — Claude API synthesis for briefing card narrative
- Background refresh on app startup + manual Refresh button

### Modify
- `pipeline_runner.py` — writes to SQLite on ingest, reads from SQLite for display
- Home page — search bar as primary UI, market overview grid as secondary

---

## Architecture

```
Ingest (Reddit + Stocktwits + NewsAPI)
        ↓
Keyword majority vote → weak labels
        ↓
Train TF-IDF + LogReg (once, on startup or manually)
        ↓
Trained model classifies all posts → SQLite
        ↓
ticker_cache rebuilt from posts
        ↓
Home page grid ← SQLite (instant)
User searches ticker → Claude reads ticker_cache → writes AI Verdict → briefing card renders
```

**Fallback:** if model not yet trained, keyword majority vote classifies live posts directly.

---

## Data Model (SQLite)

### `posts`
| Column | Type | Notes |
|--------|------|-------|
| post_id | TEXT PK | |
| text | TEXT | |
| source | TEXT | `reddit` \| `stocktwits` \| `news` |
| timestamp | DATETIME | |
| author | TEXT | |
| score | INTEGER | upvotes/likes |
| tickers | TEXT | JSON array of canonical company names |
| sentiment | TEXT | `bullish` \| `bearish` \| `neutral` \| `meme` |
| confidence | REAL | model probability score |
| url | TEXT | |

### `ticker_cache`
| Column | Type | Notes |
|--------|------|-------|
| ticker | TEXT PK | canonical company name e.g. "Tesla" |
| symbol | TEXT | "TSLA" |
| last_updated | DATETIME | |
| dominant_sentiment | TEXT | |
| mention_count | INTEGER | |
| reddit_sentiment | TEXT | dominant sentiment from Reddit posts only |
| news_sentiment | TEXT | dominant sentiment from news posts only |
| stocktwits_sentiment | TEXT | dominant sentiment from Stocktwits only |
| sentiment_by_day | TEXT | JSON: `{"2026-03-01": "bullish", ...}` |
| top_posts | TEXT | JSON: best 3 posts per source (Claude reads these) |

### `model_training_log`
| Column | Type | Notes |
|--------|------|-------|
| run_id | TEXT PK | |
| trained_at | DATETIME | |
| num_samples | INTEGER | |
| weighted_f1 | REAL | |
| label_source | TEXT | `keyword_majority` |

---

## Briefing Card

Triggered when user searches a ticker on the home page. Renders inline below the search bar.

```
┌─────────────────────────────────────────────────────┐
│  TSLA — Tesla                          BEARISH  🔴   │
│  847 mentions · last 7 days · updated 12 min ago     │
└─────────────────────────────────────────────────────┘

┌─── AI Verdict ──────────────────────────────────────┐
│  "Tesla sentiment has shifted bearish this week,     │
│   driven by delivery miss concerns on Reddit and     │
│   cautious analyst coverage. Stocktwits traders      │
│   remain split."                    — MarketPulse AI │
└─────────────────────────────────────────────────────┘

┌─── Sentiment Trend (7 days) ────────────────────────┐
│  [line chart: bullish% vs bearish% by day]           │
└─────────────────────────────────────────────────────┘

┌─── By Source ───────────────────────────────────────┐
│  Reddit/WSB          Stocktwits        News          │
│  😐 NEUTRAL          🔴 BEARISH        🔴 BEARISH    │
│  312 posts           289 posts         246 posts     │
│  [top 3 posts]       [top 3 posts]     [headlines]   │
└─────────────────────────────────────────────────────┘
```

**What Claude receives (single API call):**
- Ticker symbol + company name
- Total mention count + date range
- Sentiment breakdown per source (counts + percentages)
- `sentiment_by_day` JSON for trend context
- Top 3 posts per source (raw text)

**What Claude writes:** AI Verdict paragraph only (2-3 sentences). All charts, source cards, and post samples are rendered directly from SQLite — Claude is the narrative layer, not the data layer.

---

## Home Page

```
┌─────────────────────────────────────────────────────┐
│  MarketPulse                          [Refresh Data] │
│  Sentiment intelligence for financial markets        │
├─────────────────────────────────────────────────────┤
│                                                      │
│         🔍  Research a ticker...                     │
│         [ TSLA, NVDA, AAPL...          ] [Research]  │
│                                                      │
├─────────────────────────────────────────────────────┤
│  MARKET OVERVIEW          Last updated: 4 min ago    │
│                                                      │
│  [TSLA 🔴]  [NVDA 🟢]  [AAPL 🟢]  [GME 🟡]        │
│  [MSFT 🟢]  [AMD  🔴]  [META 🟢]  [SPY  ⚪]        │
│                                                      │
│  📈 Bullish: 5    📉 Bearish: 2    ⚪ Neutral: 1    │
└─────────────────────────────────────────────────────┘
```

**Sidebar:**
- Date range picker (max 7 days back)
- Model status: `Trained (F1: 0.74)` or `Keyword fallback`
- Data source status: Reddit ✅ Stocktwits ✅ News ✅

**App flow:**
1. Startup → ingest → classify → write SQLite → render market grid
2. User searches ticker → briefing card renders inline
3. Refresh → re-ingest → update SQLite → grid re-renders

**Single page app** — no sidebar navigation to other pages.

---

## Key Design Decisions

- **SQLite over in-memory:** free, persistent across restarts, zero cloud cost, ~5-20MB for a week of data
- **ML model over LLM classification:** fast, no per-post API cost, improves as data grows
- **Claude for narrative only:** one API call per user search, not per post — keeps costs near zero
- **Keyword fallback:** app is useful immediately on fresh install before model is trained
- **Inline briefing:** no page navigation — search and result stay on home page for snappy UX
