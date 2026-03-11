# MarketPulse Feature Changes — Design Spec

**Date**: 2026-03-11
**Status**: Approved
**Workstreams**: RSS Optimization, Code Minimization, UI Overhaul

---

## 1. RSS Optimization

### 1.1 New Feed Sources

Add 2 confirmed-free RSS sources alongside existing Google News + Yahoo Finance:

| Source | URL Pattern | Coverage |
|--------|-------------|----------|
| CNBC Markets | `https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664` | Market-wide news |
| MarketWatch | `https://feeds.marketwatch.com/marketwatch/topstories/` | Top financial stories |

Note: Seeking Alpha and Reuters feeds were considered but are likely gated/blocked. The implementation should attempt these feeds gracefully — if they return no entries, skip silently. They can be added to config as optional feeds that are validated at runtime.

Total feeds: 5 Google News + 20 Yahoo Finance + 2 new static = **27 feeds**.

### 1.2 Expanded Yahoo Finance Tickers

Current 7 → 20 high-volume tickers:

```
AAPL, TSLA, NVDA, GME, AMC, SPY, MSFT, AMZN,
META, NFLX, AMD, INTC, GOOG, DIS, COIN, HOOD,
PLTR, NIO, SOFI, QQQ
```

These match tickers the extractor already supports.

### 1.3 Date Handling Fixes

- Replace `datetime.now()` fallback with `None` — skip entries with unparseable dates
- Log a warning when date parsing fails (currently silent)
- Add per-entry try/except inside the feed loop (currently one bad entry kills the whole feed)

### 1.4 URL Normalization

Strip known tracking parameters before dedup (not all query params — some sites use query params as article identifiers):

```python
import re
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

TRACKING_PARAMS = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_term',
                   'utm_content', 'ref', 'fbclid', 'gclid', 'source'}

def _normalize_url(url):
    parsed = urlparse(url)
    params = {k: v for k, v in parse_qs(parsed.query).items() if k not in TRACKING_PARAMS}
    cleaned_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=cleaned_query, fragment=''))
```

Apply before `seen_urls` check to catch same article from different feeds with different tracking params.

### 1.5 Config Fix

Give news its own config section in `config/default.yaml`:

```yaml
ingestion:
  news:
    query_terms: ["stock market", "earnings", "IPO", "SEC", "Fed"]
    symbols: ["AAPL", "TSLA", "NVDA", "GME", "AMC", "SPY", "MSFT", "AMZN",
              "META", "NFLX", "AMD", "INTC", "GOOG", "DIS", "COIN", "HOOD",
              "PLTR", "NIO", "SOFI", "QQQ"]
    additional_feeds:
      - url: "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"
        name: "CNBC Markets"
      - url: "https://feeds.marketwatch.com/marketwatch/topstories/"
        name: "MarketWatch"
    language: "en"
```

News no longer borrows from stocktwits config.

### Files Modified
- `src/ingestion/news.py` — new feeds, expanded tickers, date fix, URL normalization, per-entry error handling
- `config/default.yaml` — new `news.symbols` list, new `news.additional_feeds` list

### Tests Needed
- Test `_normalize_url()` strips tracking params but preserves article ID params
- Test that `None` date entries are skipped (not defaulting to now)
- Test that per-entry exceptions don't kill the entire feed
- Test new feed parsing with mock feedparser data

---

## 2. Code Minimization

### 2.1 Deletions (zero behavior change)

| File | Action | Lines | Reason |
|------|--------|-------|--------|
| `app/components/metrics.py` | Delete entire file | ~100 | All 4 functions never called anywhere in app |
| `app/pipeline_runner.py:89-93` | Delete wrapper function | ~5 | `get_market_summary()` wrapper never called. Note: the method on `TickerSentimentAnalyzer` is still used by `scripts/run_pipeline.py` — only the unused wrapper in `pipeline_runner.py` is removed. |
| `src/ingestion/manager.py` | Edit docstring | ~1 line | Remove "7. Fall back to synthetic if needed (in auto mode)" from `ingest()` docstring — synthetic mode no longer exists |

### 2.2 Broken Script Fixes

**`scripts/label.py`**: Remove broken import of `LabelQualityAnalyzer` from non-existent `src.labeling.quality`. Remove lines calling it. Keep aggregation + save logic functional.

**`scripts/train.py`**: Remove broken import of `ModelVersion` from non-existent `src.models.versioning`. Remove lines calling it. Keep training logic functional.

### 2.3 Minor Cleanups

- `scripts/ingest.py:55`: Remove reference to `used_fallback` key that IngestionManager never sets

Note: `_parse_metadata()` in `src/labeling/aggregator.py` was considered for cleanup but the current two-pass approach (try clean JSON first, then quote-replaced fallback) is intentionally defensive and correct. Leave as-is.

### 2.4 NOT Touched

- Docstrings — they provide value
- Normalizer hardcoded mappings — content, not bloat
- CSV writes in `scripts/run_pipeline.py` — debug trail
- No architectural changes

### Files Modified
- `app/components/metrics.py` — delete entire file
- `app/pipeline_runner.py` — delete unused `get_market_summary()` wrapper
- `src/ingestion/manager.py` — edit docstring to remove synthetic reference
- `scripts/label.py` — remove broken import and usage of `LabelQualityAnalyzer`
- `scripts/train.py` — remove broken import and usage of `ModelVersion`
- `scripts/ingest.py` — remove `used_fallback` reference

**Total: ~115-120 lines removed, 0 behavior change to running app.**

---

## 3. UI Overhaul

### 3.1 Clickable Ticker Cards

Current ticker cards are custom HTML via `st.markdown(unsafe_allow_html=True)` — HTML elements cannot trigger Streamlit actions. Implementation approach:

- Replace each HTML card with a Streamlit `st.container` wrapping a `st.page_link` or `st.button` styled to look like the current cards
- On click, use `st.switch_page("pages/1_Ticker_Detail.py")` with `st.session_state["selected_ticker"]` set to the ticker
- Detail page reads from `st.query_params` first (for direct links), falls back to `st.session_state["selected_ticker"]` (for card clicks), then falls back to selectbox
- CSS hover effect on containers: border glow + `transform: translateY(-2px)` transition

### 3.2 Briefing Card Redesign

Replace `st.info()` with structured custom card:
- **Header bar**: Sentiment-colored left border + large symbol + company name + sentiment badge pill
- **AI Verdict**: Styled blockquote with subtle dark background, not blue info box
- **Stats row**: Mentions, confidence, sentiment as compact pills
- **Per-source breakdown**: 3 mini-cards (Reddit/Stocktwits/News) with sentiment color dots
- **7-day trend**: Larger chart (250px vs 160px), line chart with colored markers instead of tiny bars

### 3.3 Dark Theme Enhancements

Add to `styles.py`:
- Card hover animation: `transition: transform 0.2s, border-color 0.2s`
- Sentiment badge pills: Rounded colored chips (e.g., green pill for "bullish")
- Subtle gradient on page header for visual depth
- More padding between grid cards, breathing room in metrics rows

### 3.4 Loading & Feedback

- **Refresh**: Replace spinner with `st.status` showing stages:
  - "Fetching from RSS feeds..."
  - "Labeling posts..."
  - "Extracting tickers..."
  - "Analyzing sentiment..."
  - "Done — X posts from Y sources"
- **Search**: Brief "Searching..." feedback + helpful message on miss ("No results for X — try a ticker symbol like TSLA")

### 3.5 Information Hierarchy Cleanup

- **Home page**: Remove the inline 4-column KPI metrics row at `MarketPulse.py:199-208` (bullish/bearish/neutral counts) — the grid itself communicates this visually. Search bar prominent → briefing card → grid.
- **Detail page**: Simplify metrics to 3 (Mentions, Bullish %, Bearish %) — drop "Avg Confidence" (meaningless to users).
- **Model status**: Replace "F1: 0.82" with "AI-enhanced analysis active" (green) or "Basic analysis mode" (gray). No ML jargon.

### 3.6 Evidence Table Styling

Replace `st.dataframe()` with custom HTML table:
- Sentiment column: colored pills instead of text
- Source column: small badge (Reddit, News, Stocktwits)
- Alternating row backgrounds
- Hover highlight on rows

### 3.7 NOT Doing

- No comparison mode (YAGNI)
- No search history (YAGNI)
- No breadcrumbs (only 2 pages)
- No autocomplete (requires JS/state complexity)
- No custom fonts

### Files Modified
- `app/MarketPulse.py` — clickable cards, briefing redesign, loading states, KPI removal, model status text
- `app/pages/1_Ticker_Detail.py` — query param + session_state navigation, metrics simplification, table styling
- `app/components/styles.py` — hover animations, badge pills, gradient, spacing
- `app/components/charts.py` — larger trend chart, line chart variant
- `app/pipeline_runner.py` — stage-level progress reporting

---

## 4. Post-Implementation

### CLAUDE.md Update

After all changes, update CLAUDE.md to reflect:
- Remove `metrics.py` from project structure
- Update `pipeline_runner.py` description (remove `get_market_summary` mention)
- Note expanded RSS sources and ticker count
- Update any chart/component descriptions that changed

---

## Execution Order

1. **Code Minimization first** — clean slate, remove dead code before adding new code
2. **RSS Optimization second** — backend changes, no UI dependency
3. **UI Overhaul last** — depends on clean codebase and working backend
4. **CLAUDE.md update** — after all changes verified

## Agent Assignments

- **mp-ingestion**: RSS optimization (Section 1)
- **mp-dashboard**: UI overhaul (Section 3)
- **mp-qa**: Verify all changes, run full test suite after each section
- Code minimization (Section 2) is cross-cutting — executed directly, reviewed by mp-qa
