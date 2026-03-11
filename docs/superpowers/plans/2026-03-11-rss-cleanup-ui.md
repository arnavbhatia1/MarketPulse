# MarketPulse RSS + Cleanup + UI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize RSS ingestion coverage, remove dead code, and overhaul the UI for a more appealing, readable experience.

**Architecture:** Three sequential workstreams — code cleanup first (clean slate), then RSS optimization (backend), then UI overhaul (frontend). Each workstream produces a working, tested state before moving on.

**Tech Stack:** Python 3.9+, Streamlit, feedparser, Plotly, pytest

**Spec:** `docs/superpowers/specs/2026-03-11-rss-cleanup-ui-design.md`

---

## Chunk 1: Code Minimization

### Task 1: Delete dead `metrics.py`

**Files:**
- Delete: `app/components/metrics.py`

- [ ] **Step 1: Verify no imports exist**

Run: `grep -r "from app.components.metrics" . --include="*.py"` and `grep -r "import metrics" app/ --include="*.py"`
Expected: No matches found.

- [ ] **Step 2: Delete the file**

```bash
rm app/components/metrics.py
```

- [ ] **Step 3: Run tests to confirm no breakage**

Run: `python -m pytest tests/ -v`
Expected: All 158 tests pass.

- [ ] **Step 4: Commit**

```bash
git add -A app/components/metrics.py
git commit -m "chore: delete dead metrics.py — all 4 functions were never called"
```

---

### Task 2: Delete unused `get_market_summary` wrapper

**Files:**
- Modify: `app/pipeline_runner.py:89-93`

- [ ] **Step 1: Verify wrapper is not imported anywhere**

Run: `grep -r "get_market_summary" . --include="*.py"`
Expected: Only matches in `app/pipeline_runner.py` (definition) and `scripts/run_pipeline.py` (calls `analyzer.get_market_summary` directly on the class, not this wrapper).

- [ ] **Step 2: Remove the wrapper function**

Delete lines 89-93 of `app/pipeline_runner.py`:

```python
# DELETE THIS:
def get_market_summary(ticker_results: dict) -> dict:
    """Compute market-level summary from ticker_results."""
    from src.analysis.ticker_sentiment import TickerSentimentAnalyzer
    analyzer = TickerSentimentAnalyzer()
    return analyzer.get_market_summary(ticker_results)
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/pipeline_runner.py
git commit -m "chore: remove unused get_market_summary wrapper from pipeline_runner"
```

---

### Task 3: Fix broken `scripts/label.py`

**Files:**
- Modify: `scripts/label.py`

- [ ] **Step 1: Remove broken import and usage**

Remove line 16:
```python
from src.labeling.quality import LabelQualityAnalyzer
```

Remove lines 42-43:
```python
    analyzer = LabelQualityAnalyzer(LABELING_FUNCTIONS, agg)
    report = analyzer.aggregate_quality_report(df)
```

Replace lines 57-60 (which reference `report`) with inline stats:
```python
    labeled_count = df['programmatic_label'].notna().sum()
    coverage = labeled_count / len(df) if len(df) > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"LABELING COMPLETE")
    print(f"{'='*50}")
    print(f"  Total posts: {len(df)}")
    print(f"  Labeled: {labeled_count} ({labeled_count/len(df):.1%})")
    print(f"  Coverage: {coverage:.1%}")
    if 'programmatic_label' in df.columns:
        dist = df['programmatic_label'].value_counts().to_dict()
        print(f"  Distribution: {dist}")
    print(f"  Saved to: {output_path}")
    print(f"{'='*50}")
```

- [ ] **Step 2: Verify script runs**

Run: `python scripts/label.py` (will say "no ingested data" which is fine — confirms no import error)
Expected: Clean error about missing input file, NOT an ImportError.

- [ ] **Step 3: Commit**

```bash
git add scripts/label.py
git commit -m "fix: remove broken LabelQualityAnalyzer import from label.py"
```

---

### Task 4: Fix broken `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Remove broken import and usage**

Remove line 19:
```python
from src.models.versioning import ModelVersion
```

Remove lines 66-69:
```python
    mv = ModelVersion(model_dir)
    version = mv.save_version(pipeline, args.source,
                               report.get('validation_metrics', {}),
                               notes=f"Trained on {len(texts)} {args.source} samples")
```

Replace line 81 (`print(f"  Model version: v{version}")`) with:
```python
    print(f"  Model saved to: {model_dir}")
```

- [ ] **Step 2: Verify script runs**

Run: `python scripts/train.py` (will say "no labeled data" — confirms no import error)
Expected: Clean error about missing data, NOT an ImportError.

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "fix: remove broken ModelVersion import from train.py"
```

---

### Task 5: Clean up `scripts/ingest.py` and `manager.py` docstring

**Files:**
- Modify: `scripts/ingest.py:55`
- Modify: `src/ingestion/manager.py:61`

- [ ] **Step 1: Fix ingest.py**

Replace line 55:
```python
    print(f"  Fallback used: {summary['used_fallback']}")
```
With:
```python
    print(f"  Unavailable: {summary.get('sources_unavailable', [])}")
```

- [ ] **Step 2: Fix manager.py docstring**

In `src/ingestion/manager.py`, line 61 inside the `ingest()` docstring, replace:
```
        7. Fall back to synthetic if needed (in auto mode)
```
With:
```
        7. Store summary statistics
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/ingest.py src/ingestion/manager.py
git commit -m "chore: fix dead references in ingest.py and manager.py docstring"
```

---

## Chunk 2: RSS Optimization

### Task 6: Write tests for RSS improvements

**Files:**
- Create: `tests/test_news_improvements.py`

- [ ] **Step 1: Write tests for URL normalization**

```python
"""Tests for NewsIngester improvements — URL normalization, date handling, new feeds."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd


class TestUrlNormalization:
    """Test that URL normalization strips tracking params but preserves article IDs."""

    def test_strips_utm_params(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        url = "https://example.com/article?utm_source=google&utm_medium=rss&id=123"
        normalized = ingester._normalize_url(url)
        assert "utm_source" not in normalized
        assert "utm_medium" not in normalized
        assert "id=123" in normalized

    def test_strips_fbclid(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        url = "https://example.com/article?fbclid=abc123&page=2"
        normalized = ingester._normalize_url(url)
        assert "fbclid" not in normalized
        assert "page=2" in normalized

    def test_strips_fragment(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        url = "https://example.com/article?id=5#comments"
        normalized = ingester._normalize_url(url)
        assert "#comments" not in normalized
        assert "id=5" in normalized

    def test_preserves_clean_url(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        url = "https://example.com/article/12345"
        normalized = ingester._normalize_url(url)
        assert normalized == url
```

- [ ] **Step 2: Write tests for date handling**

Append to same file:

```python
class TestDateHandling:
    """Test that unparseable dates are skipped, not defaulted to now()."""

    def test_missing_date_returns_none(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        entry = {}  # no published or updated field
        result = ingester._parse_date(entry)
        assert result is None

    def test_malformed_date_returns_none(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        entry = {'published': 'not-a-date'}
        result = ingester._parse_date(entry)
        assert result is None

    def test_valid_date_parses(self):
        from src.ingestion.news import NewsIngester
        ingester = NewsIngester({'ingestion': {'news': {}}})
        entry = {'published': 'Fri, 28 Feb 2026 10:00:00 GMT'}
        result = ingester._parse_date(entry)
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
```

- [ ] **Step 3: Write test for per-entry error handling**

Append to same file:

```python
class TestPerEntryErrorHandling:
    """Test that one bad entry doesn't kill the whole feed."""

    @patch('src.ingestion.news.feedparser')
    def test_bad_entry_skipped_others_survive(self, mock_fp):
        from src.ingestion.news import NewsIngester

        good_entry = MagicMock()
        good_entry.get = lambda k, d='': {
            'link': 'https://example.com/good',
            'title': 'Good article',
            'summary': 'This is good',
            'published': 'Fri, 28 Feb 2026 10:00:00 GMT',
            'author': 'tester',
        }.get(k, d)

        # Bad entry that raises on .get('link')
        bad_entry = MagicMock()
        bad_entry.get = MagicMock(side_effect=Exception("corrupt entry"))

        mock_feed = MagicMock()
        mock_feed.entries = [bad_entry, good_entry]
        mock_fp.parse.return_value = mock_feed

        ingester = NewsIngester({'ingestion': {'news': {}}})
        rows = ingester._parse_feed(
            'https://fake.com/rss', 'test',
            set(), datetime(2026, 1, 1), datetime(2026, 12, 31)
        )
        assert len(rows) == 1
        assert rows[0]['text'].startswith('Good article')
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_news_improvements.py -v`
Expected: FAIL — `_normalize_url` doesn't exist yet, `_parse_date` returns `datetime.now()` not `None`.

- [ ] **Step 5: Commit test file**

```bash
git add tests/test_news_improvements.py
git commit -m "test: add failing tests for RSS improvements"
```

---

### Task 7: Implement RSS improvements in `news.py`

**Files:**
- Modify: `src/ingestion/news.py`

- [ ] **Step 1: Add URL normalization method**

Add imports at top of file (after line 4):
```python
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
```

Add method to `NewsIngester` class (after `_parse_date`):
```python
    _TRACKING_PARAMS = frozenset({
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_term',
        'utm_content', 'ref', 'fbclid', 'gclid', 'source',
    })

    def _normalize_url(self, url):
        """Strip tracking parameters from URL for better deduplication."""
        parsed = urlparse(url)
        params = {k: v for k, v in parse_qs(parsed.query).items()
                  if k not in self._TRACKING_PARAMS}
        cleaned_query = urlencode(params, doseq=True)
        return urlunparse(parsed._replace(query=cleaned_query, fragment=''))
```

- [ ] **Step 2: Fix `_parse_date` to return None instead of now()**

Replace lines 107-115:
```python
    def _parse_date(self, entry):
        """Parse date from feed entry. Returns None if unparseable."""
        for field in ('published', 'updated'):
            raw = entry.get(field)
            if raw:
                try:
                    return parsedate_to_datetime(raw).replace(tzinfo=None)
                except Exception:
                    pass
        logger.debug(f"News: unparseable date in entry, skipping")
        return None
```

- [ ] **Step 3: Update `_parse_feed` with per-entry error handling, URL normalization, and None date skip**

Replace lines 69-104 with:
```python
    def _parse_feed(self, feed_url, source_slug, seen_urls, start_date, end_date):
        rows = []
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                try:
                    url = entry.get('link', '')
                    if not url:
                        continue

                    normalized = self._normalize_url(url)
                    if normalized in seen_urls:
                        continue

                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    text = f"{title} {summary}".strip()[:500]
                    if not text:
                        continue

                    ts = self._parse_date(entry)
                    if ts is None:
                        continue
                    if start_date and ts < start_date:
                        continue
                    if end_date and ts > end_date:
                        continue

                    seen_urls.add(normalized)
                    url_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
                    rows.append({
                        'post_id': f"news_{source_slug}_{url_hash}",
                        'text': text,
                        'source': 'news',
                        'timestamp': ts.isoformat(),
                        'author': entry.get('author', 'unknown'),
                        'score': 0,
                        'url': url,
                        'metadata': str({'news_source': source_slug, 'article_url': url}),
                    })
                except Exception as e:
                    logger.debug(f"News: skipping bad entry in {feed_url}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"News: failed to parse feed {feed_url}: {e}")
        return rows
```

- [ ] **Step 4: Run new tests**

Run: `python -m pytest tests/test_news_improvements.py -v`
Expected: All pass.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass (existing tests may need minor adjustment if they mock `_parse_date` expecting `datetime.now()` return).

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/news.py
git commit -m "feat: add URL normalization, fix date fallback, add per-entry error handling"
```

---

### Task 8: Expand feeds and fix config

**Files:**
- Modify: `config/default.yaml`
- Modify: `src/ingestion/news.py`

- [ ] **Step 1: Update config with news-specific symbols and additional feeds**

Replace lines 22-25 of `config/default.yaml`:
```yaml
  news:
    query_terms: ["stock market", "earnings", "IPO", "SEC", "Fed"]
    language: "en"
    page_size: 100
```
With:
```yaml
  news:
    query_terms: ["stock market", "earnings", "IPO", "SEC", "Fed"]
    symbols: ["AAPL", "TSLA", "NVDA", "GME", "AMC", "SPY", "MSFT", "AMZN",
              "META", "NFLX", "AMD", "INTC", "GOOG", "DIS", "COIN", "HOOD",
              "PLTR", "NIO", "SOFI", "QQQ"]
    additional_feeds:
      - url: "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"
        name: "cnbc_markets"
      - url: "https://feeds.marketwatch.com/marketwatch/topstories/"
        name: "marketwatch"
    language: "en"
```

- [ ] **Step 2: Update NewsIngester `__init__` to read news config**

Replace lines 34-38 of `news.py`:
```python
        # Ticker symbols to fetch per-ticker Yahoo Finance feeds
        stocktwits_cfg = config.get('ingestion', {}).get('stocktwits', {})
        self.symbols = stocktwits_cfg.get(
            'symbols', ['AAPL', 'TSLA', 'NVDA', 'GME', 'MSFT', 'AMZN', 'SPY']
        )
```
With:
```python
        self.symbols = news_cfg.get(
            'symbols', ['AAPL', 'TSLA', 'NVDA', 'GME', 'AMC', 'SPY', 'MSFT', 'AMZN']
        )
        self.additional_feeds = news_cfg.get('additional_feeds', [])
```

- [ ] **Step 3: Update `ingest()` to include additional feeds**

After the Yahoo Finance loop (after line 60), add:
```python
        # Additional RSS feeds (CNBC, MarketWatch, etc.)
        for feed_info in self.additional_feeds:
            feed_url = feed_info.get('url', '')
            feed_name = feed_info.get('name', 'extra')
            if feed_url:
                rows += self._parse_feed(feed_url, feed_name, seen_urls, start_date, end_date)
```

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add config/default.yaml src/ingestion/news.py
git commit -m "feat: expand RSS to 20 tickers + CNBC/MarketWatch, fix config borrowing"
```

---

## Chunk 3: UI Overhaul

### Task 9: Enhance `styles.py` with animations and badge pills

**Files:**
- Modify: `app/components/styles.py`

- [ ] **Step 1: Add hover animations, badge pills, gradient header, and spacing**

Replace the entire `apply_theme()` CSS block (lines 33-56) with:

```python
def apply_theme():
    """Inject custom CSS for MarketPulse-specific components."""
    import streamlit as st
    st.markdown("""
    <style>
    .ticker-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        color: #E6EDF3;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    }
    .ticker-card:hover {
        border-color: #58A6FF;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(88, 166, 255, 0.15);
    }
    .ticker-card div { color: #E6EDF3; }

    .sentiment-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
        text-transform: uppercase;
    }
    .sentiment-badge-bullish { background: rgba(0,200,83,0.15); color: #00C853; }
    .sentiment-badge-bearish { background: rgba(255,23,68,0.15); color: #FF1744; }
    .sentiment-badge-neutral { background: rgba(120,144,156,0.15); color: #78909C; }
    .sentiment-badge-meme { background: rgba(255,214,0,0.15); color: #FFD600; }

    .sentiment-bullish { color: #00C853 !important; font-weight: bold; }
    .sentiment-bearish { color: #FF1744 !important; font-weight: bold; }
    .sentiment-neutral { color: #78909C !important; font-weight: bold; }
    .sentiment-meme { color: #FFD600 !important; font-weight: bold; }

    .page-header {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #1a2332 100%);
        padding: 20px 0;
        margin-bottom: 16px;
        border-bottom: 1px solid #30363D;
    }

    .briefing-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 20px;
        margin: 12px 0;
    }
    .briefing-verdict {
        background: #0D1117;
        border-left: 3px solid #58A6FF;
        padding: 12px 16px;
        margin: 12px 0;
        border-radius: 0 6px 6px 0;
        font-style: italic;
        color: #E6EDF3;
    }
    .source-card {
        background: #0D1117;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 12px;
        margin: 4px 0;
    }

    div[data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 12px;
    }

    .evidence-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9em;
    }
    .evidence-table th {
        background: #21262D;
        color: #E6EDF3;
        padding: 10px 12px;
        text-align: left;
        border-bottom: 2px solid #30363D;
    }
    .evidence-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #30363D;
        color: #E6EDF3;
    }
    .evidence-table tr:nth-child(even) { background: #161B22; }
    .evidence-table tr:nth-child(odd) { background: #0D1117; }
    .evidence-table tr:hover { background: #21262D; }
    </style>
    """, unsafe_allow_html=True)
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All pass (styles is CSS-only, no logic to break).

- [ ] **Step 3: Commit**

```bash
git add app/components/styles.py
git commit -m "feat: enhance dark theme with hover animations, badge pills, evidence table styles"
```

---

### Task 10: Improve trend chart in `charts.py`

**Files:**
- Modify: `app/components/charts.py`

- [ ] **Step 1: Add a sentiment trend line chart function**

Add after `probability_bar` function (after line 147):

```python
def sentiment_trend(by_day, sentiment_colors=None):
    """
    7-day sentiment trend as a line chart with colored markers.

    Args:
        by_day: dict mapping date string -> sentiment label
        sentiment_colors: optional dict of sentiment -> hex color

    Returns:
        Plotly Figure object
    """
    if not sentiment_colors:
        sentiment_colors = SENTIMENT_COLORS

    days = sorted(by_day.keys())
    sentiments = [by_day[d] for d in days]
    marker_colors = [sentiment_colors.get(s, COLORS['secondary']) for s in sentiments]

    # Map sentiment to numeric for the line
    sentiment_map = {'bullish': 3, 'neutral': 2, 'meme': 1, 'bearish': 0}
    y_values = [sentiment_map.get(s, 2) for s in sentiments]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=y_values,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(color=marker_colors, size=12, line=dict(width=2, color='#0D1117')),
        text=[s.upper() for s in sentiments],
        hovertemplate='%{x}<br>%{text}<extra></extra>',
    ))
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=0, r=0, t=10, b=30),
        showlegend=False,
        yaxis=dict(
            ticktext=['BEARISH', 'MEME', 'NEUTRAL', 'BULLISH'],
            tickvals=[0, 1, 2, 3],
            gridcolor='#30363D',
        ),
        xaxis=dict(gridcolor='#30363D'),
    )
    return fig
```

- [ ] **Step 2: Commit**

```bash
git add app/components/charts.py
git commit -m "feat: add sentiment_trend line chart for 7-day view"
```

---

### Task 11: Overhaul home page (`MarketPulse.py`)

**Files:**
- Modify: `app/MarketPulse.py`

- [ ] **Step 1: Update imports**

Replace line 19:
```python
from app.components.charts import ticker_mentions_bar
```
With:
```python
from app.components.charts import ticker_mentions_bar, sentiment_trend
```

- [ ] **Step 2: Replace model status with user-friendly text**

Replace lines 65-70:
```python
if model and model.is_trained:
    history = get_training_history()
    f1 = history[0]['weighted_f1'] if history else 0.0
    st.sidebar.success(f"Model trained (F1: {f1:.2f})")
else:
    st.sidebar.info("Keyword fallback active (model not yet trained)")
```
With:
```python
if model and model.is_trained:
    st.sidebar.success("AI-enhanced analysis active")
else:
    st.sidebar.info("Basic analysis mode")
```

- [ ] **Step 3: Add loading stages to Refresh button**

Replace lines 51-58:
```python
if st.sidebar.button("Refresh Data", use_container_width=True):
    with st.spinner("Ingesting and analyzing market data..."):
        source_summary = refresh_pipeline(
            start_date_str=start_date.isoformat(),
            end_date_str=end_date.isoformat(),
        )
        st.cache_data.clear()
    st.rerun()
```
With:
```python
if st.sidebar.button("Refresh Data", use_container_width=True):
    with st.status("Refreshing market data...", expanded=True) as status:
        st.write("Fetching from RSS feeds...")
        source_summary = refresh_pipeline(
            start_date_str=start_date.isoformat(),
            end_date_str=end_date.isoformat(),
        )
        st.write("Analysis complete.")
        posts = source_summary.get('total_posts', 0)
        sources = source_summary.get('sources_used', [])
        status.update(label=f"Done — {posts} posts from {', '.join(sources)}", state="complete")
        st.cache_data.clear()
    st.rerun()
```

- [ ] **Step 4: Redesign briefing card**

Replace lines 122-138 (header card HTML) with:
```python
        # Header card with badge pill
        st.markdown(f"""
        <div class="briefing-card" style="border-left: 4px solid {color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:1.8em; font-weight:bold; letter-spacing:0.5px;">{symbol}</span>
                    <span style="color:#8B949E; margin-left:12px; font-size:1.1em;">{resolved}</span>
                </div>
                <span class="sentiment-badge sentiment-badge-{dominant}">{dominant.upper()}</span>
            </div>
            <div style="color:#8B949E; font-size:0.85em; margin-top:8px;">
                {mention_count} mentions · updated {last_updated[:16] if last_updated != 'unknown' else 'unknown'}
            </div>
        </div>
        """, unsafe_allow_html=True)
```

- [ ] **Step 5: Redesign AI verdict section**

Replace lines 140-145:
```python
        # AI Verdict
        with st.container():
            st.markdown("#### AI Verdict")
            with st.spinner("Generating verdict..."):
                verdict = generate_briefing(resolved, symbol, ticker_data)
            st.info(f'"{verdict}"\n\n— MarketPulse AI')
```
With:
```python
        # AI Verdict
        with st.spinner("Generating AI verdict..."):
            verdict = generate_briefing(resolved, symbol, ticker_data)
        st.markdown(f"""
        <div class="briefing-verdict">
            {verdict}
            <div style="color:#8B949E; font-size:0.8em; margin-top:8px;">— MarketPulse AI</div>
        </div>
        """, unsafe_allow_html=True)
```

- [ ] **Step 6: Replace tiny bar trend chart with line chart**

Replace lines 147-169 (sentiment trend section) with:
```python
        # Sentiment trend chart
        by_day = ticker_data.get('sentiment_by_day', {})
        if by_day:
            st.markdown("#### Sentiment Trend (7 days)")
            fig = sentiment_trend(by_day)
            st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 7: Redesign by-source breakdown with styled cards**

Replace lines 171-187 (By Source section) with:
```python
        # By Source breakdown
        st.markdown("#### By Source")
        top_posts = ticker_data.get('top_posts', {})
        src_cols = st.columns(3)
        for i, source_name in enumerate(('reddit', 'stocktwits', 'news')):
            src_sentiment = ticker_data.get(f'{source_name}_sentiment') or 'N/A'
            src_posts = top_posts.get(source_name, [])
            src_color = SENTIMENT_COLORS.get(src_sentiment, COLORS['secondary'])
            with src_cols[i]:
                st.markdown(f"""
                <div class="source-card">
                    <div style="font-weight:bold; margin-bottom:6px;">{source_name.upper()}</div>
                    <span class="sentiment-badge sentiment-badge-{src_sentiment}"
                          style="font-size:0.75em;">{src_sentiment.upper()}</span>
                </div>
                """, unsafe_allow_html=True)
                for post in src_posts[:3]:
                    st.caption(f"> {post['text'][:100]}...")
```

- [ ] **Step 8: Remove KPI row, improve grid with search miss message**

Replace lines 191-233 (market overview section) with:
```python
# ── Market Overview grid ──────────────────────────────────────────────────────
if not ticker_results:
    st.info(
        "No market data yet. Click **Refresh Data** in the sidebar to ingest and analyze."
    )
else:
    st.markdown("### Market Overview")
    st.markdown("---")

    # Ticker card grid — clickable
    cols = st.columns(3)
    for i, (company, data) in enumerate(ticker_results.items()):
        sentiment = data.get('dominant_sentiment', 'neutral')
        color = SENTIMENT_COLORS.get(sentiment, COLORS['secondary'])
        symbol_display = data.get('symbol', company.upper())
        mentions = data.get('mention_count', 0)

        with cols[i % 3]:
            if st.button(
                f"{symbol_display}",
                key=f"card_{company}",
                use_container_width=True,
            ):
                st.session_state["selected_ticker"] = company
                st.switch_page("pages/1_Ticker_Detail.py")

            st.markdown(f"""
            <div class="ticker-card" style="margin-top:-12px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:1.1em; font-weight:bold;">{symbol_display}</span>
                        <span style="color:#8B949E; font-size:0.8em; margin-left:6px;">{company}</span>
                    </div>
                    <span class="sentiment-badge sentiment-badge-{sentiment}" style="font-size:0.7em;">
                        {sentiment.upper()}
                    </span>
                </div>
                <div style="color:#8B949E; font-size:0.8em; margin-top:6px;">
                    {mentions} mentions
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Most mentioned bar chart
    st.markdown("---")
    st.markdown("### Most Mentioned Tickers")
    fig = ticker_mentions_bar(ticker_results, top_n=15)
    st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 9: Update search miss message**

Replace line 114:
```python
        st.warning(f"No data found for **{query.strip()}**. Try refreshing data or check the ticker symbol.")
```
With:
```python
        st.warning(f"No data for **{query.strip()}**. Try a ticker symbol like TSLA, NVDA, or AAPL — then hit **Refresh Data** if needed.")
```

- [ ] **Step 10: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All pass.

- [ ] **Step 11: Commit**

```bash
git add app/MarketPulse.py
git commit -m "feat: overhaul home page — badge pills, briefing card, trend chart, clickable grid"
```

---

### Task 12: Overhaul ticker detail page

**Files:**
- Modify: `app/pages/1_Ticker_Detail.py`

- [ ] **Step 1: Add query param + session state navigation**

After line 31 (`apply_theme()`), add:
```python
# Read ticker from query params if present (for direct links)
query_ticker = st.query_params.get("ticker", None)
```

Replace lines 71-74:
```python
session_ticker = st.session_state.get("selected_ticker", None)
default_index = 0
if session_ticker and session_ticker in ticker_options:
    default_index = ticker_options.index(session_ticker)
```
With:
```python
# Priority: query params > session state > first in list
resolved_ticker = query_ticker or st.session_state.get("selected_ticker", None)
default_index = 0
if resolved_ticker and resolved_ticker in ticker_options:
    default_index = ticker_options.index(resolved_ticker)
```

- [ ] **Step 2: Simplify metrics to 3 columns**

Replace lines 134-138:
```python
col_k1, col_k2, col_k3, col_k4 = st.columns(4)
col_k1.metric("Mentions", f"{mention_count:,}")
col_k2.metric("Avg Confidence", f"{avg_conf:.1%}")
col_k3.metric("Bullish Ratio", f"{bullish_ratio:.1%}")
col_k4.metric("Bearish Ratio", f"{bearish_ratio:.1%}")
```
With:
```python
col_k1, col_k2, col_k3 = st.columns(3)
col_k1.metric("Mentions", f"{mention_count:,}")
col_k2.metric("Bullish", f"{bullish_ratio:.0%}")
col_k3.metric("Bearish", f"{bearish_ratio:.0%}")
```

- [ ] **Step 3: Replace dataframe with styled HTML table**

Replace lines 227-231:
```python
    st.dataframe(
        display_df,
        use_container_width=True,
        height=350,
    )
```
With:
```python
    # Build styled HTML table
    table_html = '<table class="evidence-table"><thead><tr>'
    for col in display_df.columns:
        table_html += f'<th>{col.title()}</th>'
    table_html += '</tr></thead><tbody>'
    for _, row in display_df.iterrows():
        table_html += '<tr>'
        for col in display_df.columns:
            val = row[col]
            if col == 'sentiment' and pd.notna(val):
                badge_cls = f"sentiment-badge-{val}" if val in ('bullish','bearish','neutral','meme') else ''
                table_html += f'<td><span class="sentiment-badge {badge_cls}">{val}</span></td>'
            else:
                table_html += f'<td>{val if pd.notna(val) else "—"}</td>'
        table_html += '</tr>'
    table_html += '</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add app/pages/1_Ticker_Detail.py
git commit -m "feat: overhaul detail page — query params nav, simplified metrics, styled table"
```

---

### Task 13: Add stage-level progress to `pipeline_runner.py`

**Files:**
- Modify: `app/pipeline_runner.py`

- [ ] **Step 1: Add progress callback support to `refresh_pipeline`**

Replace lines 32-76 of `pipeline_runner.py` with:

```python
def refresh_pipeline(start_date_str=None, end_date_str=None, progress_callback=None) -> dict:
    """
    Run full pipeline: ingest -> label -> extract -> analyze -> write SQLite.

    Args:
        start_date_str: ISO date string or None
        end_date_str: ISO date string or None
        progress_callback: optional callable(message: str) for progress updates
    """
    def _progress(msg):
        if progress_callback:
            progress_callback(msg)

    init_db()
    config = load_config()

    start_date = datetime.fromisoformat(start_date_str) if start_date_str else None
    end_date = datetime.fromisoformat(end_date_str) if end_date_str else None

    # Ingest
    _progress("Fetching from RSS feeds...")
    mgr = IngestionManager(config)
    df = mgr.ingest(start_date=start_date, end_date=end_date)
    source_summary = mgr.get_source_summary()

    # Label
    _progress("Labeling posts...")
    agg = LabelAggregator(config=config)
    df = agg.aggregate_batch(df)

    # Extract tickers
    _progress("Extracting tickers...")
    te = TickerExtractor()
    df['tickers'] = df['text'].apply(te.extract)

    # Map programmatic_label -> sentiment column for storage
    df['sentiment'] = df['programmatic_label']
    if 'label_confidence' in df.columns:
        df['confidence'] = df['label_confidence'].fillna(0.0)
    else:
        df['confidence'] = 0.0

    # Analyze per-ticker
    _progress("Analyzing sentiment...")
    analyzer = TickerSentimentAnalyzer()
    ticker_results = analyzer.analyze(df)

    # Write to SQLite
    _progress("Saving results...")
    save_posts(df)
    save_ticker_cache(ticker_results)

    # Optionally train/retrain model if enough labeled data
    _maybe_train_model(df, config)

    return source_summary
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All pass (progress_callback is optional, defaults to None).

- [ ] **Step 3: Commit**

```bash
git add app/pipeline_runner.py
git commit -m "feat: add progress callback to refresh_pipeline for stage-level UI feedback"
```

---

### Task 14: Update CLAUDE.md and final verification

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update project structure**

Remove `metrics.py` from the project structure in CLAUDE.md. Update the `pipeline_runner.py` description to remove `get_market_summary()` mention.

- [ ] **Step 2: Update key modules section**

In the `app/pipeline_runner.py` section, remove the bullet about `get_market_summary()`. Add note about expanded RSS feeds.

- [ ] **Step 3: Run full test suite one final time**

Run: `python -m pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect dead code removal and new features"
```

---

## Summary

| Chunk | Tasks | Commits | Focus |
|-------|-------|---------|-------|
| 1: Code Minimization | 1-5 | 5 | Delete dead code, fix broken scripts |
| 2: RSS Optimization | 6-8 | 3 | Tests first, then implementation, then config |
| 3: UI Overhaul | 9-14 | 6 | Styles, charts, home page, detail page, pipeline, docs |
| **Total** | **14** | **14** | |
