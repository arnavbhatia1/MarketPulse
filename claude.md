# CLAUDE.md — MarketPulse: Data-Centric Sentiment Intelligence for Financial Social Media

## PROJECT OVERVIEW

MarketPulse is a production-grade, data-centric ML pipeline: ingests real financial social media data, applies programmatic labeling via weak supervision, trains classical ML models, extracts ticker entities, and surfaces everything through an interactive Streamlit dashboard.

**Core Thesis:** *"A logistic regression trained on high-quality, programmatically labeled data is a production-ready system. The data is the model."*

### Pipeline Steps
1. Ingest from Reddit (WSB), Stocktwits, NewsAPI — fallback to synthetic if no API keys
2. Apply 15+ labeling functions encoding financial domain heuristics
3. Assess label quality (coverage, conflict, confidence, agreement)
4. Train TF-IDF + Logistic Regression on programmatic labels
5. Extract ticker entities with rule-based extraction + normalization
6. Evaluate with full metrics, error analysis, data quality diagnostics
7. Surface everything in an interactive Streamlit dashboard with live inference

---

## ARCHITECTURE

```
marketpulse/
├── CLAUDE.md / README.md / requirements.txt / setup.py / .env.example / Makefile
├── config/
│   ├── default.yaml
│   └── sources.yaml
├── data/
│   ├── raw/                        # gitignored
│   ├── labeled/
│   ├── gold/gold_standard.csv      # 100 hand-labeled posts
│   ├── synthetic/synthetic_posts.csv
│   └── models/
│       ├── tfidf_vectorizer.pkl
│       ├── sentiment_model.pkl
│       └── model_metadata.json
├── src/
│   ├── ingestion/   base.py, reddit.py, stocktwits.py, news.py, synthetic.py, manager.py
│   ├── labeling/    functions.py, aggregator.py, quality.py
│   ├── models/      pipeline.py, versioning.py
│   ├── extraction/  ticker_extractor.py, normalizer.py
│   ├── evaluation/  classification.py, extraction.py, label_quality.py
│   └── utils/       config.py, logger.py, cache.py
├── app/
│   ├── streamlit_app.py
│   ├── pages/  1_data_ingestion.py, 2_labeling_studio.py, 3_label_quality.py,
│   │           4_model_training.py, 5_entity_extraction.py, 6_evaluation.py, 7_live_inference.py
│   └── components/  charts.py, metrics.py, styles.py
├── scripts/  ingest.py, label.py, train.py, evaluate.py, run_pipeline.py
└── tests/    test_ingestion.py, test_labeling_functions.py, test_aggregator.py,
              test_pipeline.py, test_extraction.py, test_normalizer.py, test_evaluation.py
```

---

## CONFIGURATION

### `config/default.yaml`
```yaml
project:
  name: "MarketPulse"
  version: "1.0.0"

data:
  mode: "auto"  # "live" | "synthetic" | "auto"
  storage:
    raw_dir: "data/raw"
    labeled_dir: "data/labeled"
    gold_dir: "data/gold"
    model_dir: "data/models"
  synthetic:
    num_posts: 500
    seed: 42

ingestion:
  reddit:
    subreddits: ["wallstreetbets", "stocks", "investing"]
    post_limit_per_sub: 200
    include_comments: false
    min_score: 5
  stocktwits:
    symbols: ["AAPL", "TSLA", "NVDA", "GME", "AMC", "SPY", "MSFT", "AMZN"]
    limit_per_symbol: 50
  news:
    query_terms: ["stock market", "earnings", "IPO", "SEC", "Fed"]
    language: "en"
    page_size: 100
  date_range:
    start: null   # 7 days ago
    end: null     # today
    default_lookback_days: 7

labeling:
  aggregation_strategy: "confidence_weighted"  # "majority" | "weighted" | "confidence_weighted"
  confidence_threshold: 0.6
  min_votes: 2

model:
  max_features: 500
  ngram_range: [1, 2]
  min_df: 3
  C: 1.0
  class_weight: "balanced"
  test_size: 0.2
  random_state: 42

evaluation:
  gold_set_size: 100

dashboard:
  theme: "dark"
  port: 8501
```

### `.env.example`
```bash
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=MarketPulse/1.0
STOCKTWITS_ACCESS_TOKEN=your_token
NEWS_API_KEY=your_key
ALPHA_VANTAGE_KEY=your_key  # optional
```

---

## DATA INGESTION

### `src/ingestion/base.py` — `BaseIngester(ABC)`

**REQUIRED_COLUMNS schema** (all ingesters must return this):
```
post_id    # str: source_prefix + original_id
text       # str: raw post text
source     # str: 'reddit' | 'stocktwits' | 'news' | 'synthetic'
timestamp  # datetime
author     # str: username or 'unknown'
score      # int: upvotes/likes/engagement
url        # str: link or empty
metadata   # dict: source-specific fields
```
Abstract methods: `ingest(start_date, end_date) -> DataFrame`, `is_available() -> bool`
Concrete: `validate_output(df)` — drop null text, deduplicate by post_id, log issues.

### `src/ingestion/reddit.py` — `RedditIngester`
- PRAW with credentials from `.env`; fetch `subreddit.new()` + `subreddit.hot()` in date range
- Text = `title + selftext`; filter `score < min_score`; exponential backoff for rate limits
- `post_id` format: `"reddit_{subreddit}_{original_id}"`
- Metadata: `subreddit`, `num_comments`, `flair`, `is_self`, `link_flair_text`

### `src/ingestion/stocktwits.py` — `StocktwitsIngester`
- API: `https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json`
- Valuable: posts are ticker-associated; some have user-submitted bullish/bearish tags (weak supervision signal)
- `post_id` format: `"stocktwits_{message_id}"`
- Metadata: `symbols` (list), `user_sentiment` (str or null), `reshares`, `likes`

### `src/ingestion/news.py` — `NewsIngester`
- Primary: NewsAPI `https://newsapi.org/v2/everything`; fallback: Finnhub
- Why: news is almost always NEUTRAL → good training signal for neutral class
- Use `title + description` (not full body); deduplicate by headline similarity
- `post_id` format: `"news_{source}_{hash_of_url}"`
- Metadata: `news_source`, `article_url`, `image_url`, `published_at`

### `src/ingestion/synthetic.py` — `SyntheticIngester`
- Fallback only; `is_available()` always returns `True`
- Pre-generated at `data/synthetic/synthetic_posts.csv` (committed to git)
- `generate(num_posts=500, seed=42)` — called during setup or on-demand
- Distribution: 30% bullish, 20% bearish, 25% neutral, 25% meme; 50+ edge cases

### `src/ingestion/manager.py` — `IngestionManager`
Modes:
- `live`: all configured live sources; error if none available
- `synthetic`: only synthetic
- `auto` (default): all available live sources → fallback to synthetic if none

Handles: cross-source deduplication, schema validation, caching, unified DataFrame, source statistics.

`get_source_summary()` returns:
```python
{'total_posts': int, 'sources_used': [...], 'sources_unavailable': [...],
 'date_range': {'start': datetime, 'end': datetime},
 'posts_per_source': {'reddit': 150, ...}, 'mode': 'auto', 'used_fallback': False}
```

---

## SYNTHETIC DATA SPECIFICATION

### Sentiment Categories

**BULLISH (30%):** Conviction buys, TA breakouts, earnings optimism, dip buyers, options bulls, subtle bulls.
- Examples: `"Just loaded 500 shares of NVDA. This is going to $200."` / `"Loaded up on SPY 500c for December. Free money."`

**BEARISH (20%):** Short sellers, put buyers, warning posts, fundamental bears, macro bears, subtle bears.
- Examples: `"Shorting TSLA at these levels. P/E is insane."` / `"Fed isn't cutting rates. Markets will tank."`

**NEUTRAL (25%):** Questions, news sharing, analysis, educational, discussion starters, comparative.
- Examples: `"What's everyone thinking about AAPL earnings Thursday?"` / `"TSLA Q3 deliveries came in at 435K units."`

**MEME (25%):** Loss porn, self-deprecating, hype without substance, ironic, cultural, sarcastic.
- Examples: `"Down 90% on GME weeklies. See you behind Wendy's 🍔"` / `"APES TOGETHER STRONG 🦍💎🙌 GME TO THE MOON"`

### Required Edge Cases (50+)
1. **Sarcastic bullish → actually bearish/meme:** `"sure this time the short squeeze will definitely happen 🙄"`
2. **Meme with real conviction:** `"I know this sounds like a meme but I genuinely believe GME is transforming into a tech company."`
3. **Bearish words, bullish position:** `"Everyone says PLTR is garbage but I just doubled my position. Contrarian play."`
4. **Question with embedded sentiment:** `"Is anyone else worried NVDA is in a bubble?"` / `"Why would anyone sell AAPL before earnings?"`
5. **News mixed with opinion:** `"TSLA beat deliveries by 10%. This stock is going to $400 🚀"`
6. **Loss mention — meme or bearish?:** `"Down 80% on my calls. At least I still have my cardboard box 📦"` vs `"Down 80% on my calls. Getting out ASAP."`
7. **Contradictory:** `"I think TSLA is overvalued but I keep buying every dip lol"`
8. **Short and ambiguous:** `"TSLA 🚀"` / `"GUH"` / `"RIP"` / `"moon soon"`
9. **Neutral analysis sounding bearish:** `"NVDA P/E is 65x. Historical semi average is 20x. Make your own conclusions."`
10. **Multi-ticker mixed sentiment:** `"Selling AAPL to buy NVDA. One's dead money, other's the future."`

### Required Text Patterns (all categories)
- Tickers ($TSLA, $AAPL) — ≥70% of posts
- Emojis — ≥40%; ALL CAPS words — ≥30%; informal language — ≥25%
- Multiple tickers in one post — ≥15%; numbers/prices — ≥30%; options language — ≥20%
- WSB slang concentrated in meme; varied lengths (10–280 chars)
- Posts with ONLY emojis+tickers; hashtags (#YOLO, $NVDA); misspellings (stonks, gunna)

### Gold Standard: `data/gold/gold_standard.csv`
100 posts: 25 per category, ≥20 ambiguous edge cases.
```csv
post_id,text,sentiment_gold,tickers_gold,ambiguity_score,notes
1,"...",bullish,"['TSLA','AAPL']",1,"Clear bullish conviction buy"
42,"...",meme,"['GME']",4,"Uses bullish language but clearly ironic"
```
`ambiguity_score`: 1 (crystal clear) → 5 (genuinely ambiguous even for humans)

---

## PROGRAMMATIC LABELING

### `src/labeling/functions.py`

Labels: `BULLISH = "bullish"`, `BEARISH = "bearish"`, `NEUTRAL = "neutral"`, `MEME = "meme"`, `ABSTAIN = -1`

Convention: `lf_{signal_type}_{what_it_detects}`. Each function: accepts `text: str`, returns label or ABSTAIN, has docstring with heuristic + known limitations.

**Keyword functions:**
- `lf_keyword_bullish` — `['buy','buying','bought','long','calls','bullish','loading up','accumulating','undervalued','breakout','buy the dip','btd','price target','upside','going up','all in','free money','easy money','no brainer']` → BULLISH (~30% coverage, ~70% precision)
- `lf_keyword_bearish` — `['sell','selling','sold','short','puts','bearish','crash','dump','overvalued','bubble','top is in','rug pull','dead cat','exit','taking profits','get out','going down','bagholder']` → BEARISH
- `lf_keyword_neutral` — `['what do you think','thoughts on','anyone know','how does','when is','eli5','explains','thoughts?','opinions?','announces','reports','according to','earnings report','quarterly results']` → NEUTRAL
- `lf_keyword_meme` — `['apes','tendies',"wife's boyfriend",'diamond hands','paper hands','yolo','guh','stonks','smooth brain','degen','casino','gambling','loss porn','sir this is','to the moon','ape together strong','hodl']` → MEME

**Emoji functions:**
- `lf_emoji_bullish` — `['🚀','🌙','📈','🐂','💰','🤑','⬆️']` → BULLISH (⚠️ rockets also meme)
- `lf_emoji_bearish` — `['📉','🐻','💀','🔻','⬇️','😱','🩸']` → BEARISH
- `lf_emoji_meme` — `['💎','🙌','🦍','🤡','🎰','🍗','🫠']` → MEME

**Structural functions:**
- `lf_question_structure` — ends with `?` or `count('?') >= 2` → NEUTRAL
- `lf_short_post` — `len(text) < 15` → MEME
- `lf_all_caps_ratio` — `caps_ratio >= 0.4` + bear/bull keywords → BEARISH or BULLISH

**Financial pattern functions:**
- `lf_options_directional` — calls phrases → BULLISH; puts phrases → BEARISH (highest precision)
- `lf_price_target_mention` — regex `(PT|price target|see this at|heading to)\s*\$?\d+` → BULLISH
- `lf_loss_reporting` — regex `down \d+%` or `lost \$[\d,]+` or `bag holding` → MEME
- `lf_news_language` — ≥2 news pattern matches (`announces`,`SEC`,`IPO`,`acquisition`,`quarterly`,`upgrade`,`downgrade`,`breaking:`) → NEUTRAL

**Sarcasm/irony functions:**
- `lf_sarcasm_indicators` — sarcasm markers + bullish words, or 🤡/🙄 + bullish words → BEARISH
- `lf_self_deprecating` — `["wife's boyfriend",'smooth brain','behind wendy','cardboard box','financially ruined']` → MEME

**Source-aware functions (require metadata):**
- `lf_stocktwits_user_sentiment(text, metadata)` — `metadata['user_sentiment']` bullish/bearish → respective label
- `lf_reddit_flair(text, metadata)` — flair `yolo/loss/meme` → MEME; `discussion/news/dd` → NEUTRAL

```python
LABELING_FUNCTIONS = [lf_keyword_bullish, lf_keyword_bearish, lf_keyword_neutral,
    lf_keyword_meme, lf_emoji_bullish, lf_emoji_bearish, lf_emoji_meme,
    lf_question_structure, lf_short_post, lf_all_caps_ratio, lf_options_directional,
    lf_price_target_mention, lf_loss_reporting, lf_news_language,
    lf_sarcasm_indicators, lf_self_deprecating]
METADATA_FUNCTIONS = [lf_stocktwits_user_sentiment, lf_reddit_flair]
```

### `src/labeling/aggregator.py` — `LabelAggregator`

Strategies: `"majority"` | `"weighted"` | `"confidence_weighted"` (default)

`aggregate_single(text, metadata=None)` returns:
```python
{'final_label': str|None, 'confidence': float, 'votes': {'lf_name': label, ...},
 'num_votes': int, 'num_abstains': int, 'has_conflict': bool,
 'competing_labels': {'bullish': 3, 'meme': 2}}
```

`aggregate_batch(df)` adds columns: `programmatic_label`, `label_confidence`, `label_coverage`, `label_conflict`, `vote_breakdown`

**Weights for `_weighted_vote`:**
- `lf_options_directional`: 3.0 (very reliable)
- `lf_keyword_bullish/bearish`: 2.0 (moderately reliable)
- `lf_emoji_bullish`: 1.0 (noisy — rockets in memes)

`_confidence_weighted`: like weighted, but posts below threshold get `label=None` (uncertain → valuable for human review)

### `src/labeling/quality.py` — `LabelQualityAnalyzer`

`per_function_report(df)` → DataFrame (one row/function): coverage %, label distribution, conflict rate, accuracy on gold, overlap with other functions.

`aggregate_quality_report(df)` → dict: total_coverage, avg_votes_per_post, conflict_rate, confidence distribution, uncertain_count, label_distribution, expected_vs_actual.

`compare_to_gold(df, gold_df)` → per-class P/R/F1, confusion matrix, disagreement list (text + programmatic label + confidence + gold label + ambiguity score + vote breakdown + likely reason), agreement rate by ambiguity score.

`label_quality_experiment(df, gold_df, X_test, y_test_gold)` — **THE THESIS EXPERIMENT:**
Train identical TF-IDF+LogReg on:
1. Gold labels → F1 ~0.85+
2. Programmatic labels → F1 ~0.75-0.82
3. Noisy labels (30% noise injected) → F1 ~0.55-0.65
4. Random labels → F1 ~0.25

Returns comparison table + Plotly bar chart. Proves: same model + better labels = dramatically better performance.

---

## ML MODEL PIPELINE

### `src/models/pipeline.py` — `SentimentPipeline`

**`preprocess(text)`:** lowercase + whitespace normalize ONLY. Do NOT strip emojis, tickers, or punctuation — all carry signal.

**`train(texts, labels, validation_split=True)`:** preprocess → train/val split → fit TfidfVectorizer → fit LogisticRegression → evaluate → store metadata (date, size, metrics) → cross-validate.

**`predict(texts)`** returns:
```python
[{'text': str, 'label': str, 'confidence': float,
  'probabilities': {'bullish': 0.7, 'bearish': 0.1, ...}}]
```

**`get_feature_importance(top_n=15)`** returns:
```python
{'bullish': [('calls', 1.23), ...], 'bearish': [('puts', 1.45), ...],
 'neutral': [('thoughts', 0.9), ...], 'meme': [('apes', 1.5), ...]}
```

**`error_analysis(texts, true_labels, predicted_labels)`:** per-error breakdown (text, true vs predicted, confidence, driving features, category: `labeling_ambiguity|model_limitation|data_quality`) + aggregate (confused pairs, confidence on correct vs incorrect, errors by ambiguity score).

**`save(path)`:** `tfidf_vectorizer.pkl`, `sentiment_model.pkl`, `model_metadata.json`

### `src/models/versioning.py` — `ModelVersion`

`save_version(pipeline, label_source, metrics, notes)` → directory `data/models/v{N}_{label_source}_{timestamp}/`

Metadata: version, `label_source` (`gold|programmatic|noisy|random`), training_date, dataset_size, full classification report, config, notes.

Methods: `list_versions()`, `compare_versions(version_ids)`, `load_version(version_id)`

---

## ENTITY EXTRACTION

### `src/extraction/ticker_extractor.py` — `TickerExtractor`

**Extraction methods (applied in order):**
1. `_extract_cashtags` — `$TICKER` patterns; must distinguish `$5` (price) from `$AAPL`
2. `_extract_bare_tickers` — ALL-CAPS standalone words; skip `ambiguous_tickers = {'F','T','V','AI','ALL','IT','NOW'}` unless `$`-prefixed
3. `_extract_company_names` — via `company_aliases` dict
4. `_extract_informal` — via `informal_aliases`
5. `_extract_emoji` — via `emoji_map`

**Key data maps:**
```python
ticker_map = {
    'AAPL':'Apple','TSLA':'Tesla','MSFT':'Microsoft','GOOG':'Google','GOOGL':'Google',
    'AMZN':'Amazon','NVDA':'NVIDIA','META':'Meta','GME':'GameStop','AMC':'AMC',
    'SPY':'S&P 500 ETF','QQQ':'Nasdaq ETF','PLTR':'Palantir','NFLX':'Netflix',
    'DIS':'Disney','AMD':'AMD','INTC':'Intel','JPM':'JPMorgan','BAC':'Bank of America',
    'F':'Ford','T':'AT&T','COIN':'Coinbase','HOOD':'Robinhood','RIVN':'Rivian',
    'BABA':'Alibaba','TSM':'TSMC','ORCL':'Oracle','CRM':'Salesforce','PYPL':'PayPal',
    'V':'Visa','MA':'Mastercard', ...}

informal_aliases = {'papa musk':'Tesla','elon':'Tesla','zuck':'Meta',
    'zuckerberg':'Meta','tim cook':'Apple','bezos':'Amazon','jensen':'NVIDIA',
    'satya':'Microsoft','nadella':'Microsoft','lisa su':'AMD'}

emoji_map = {'🍎':'Apple'}
```

`extract_with_evidence(text)` returns:
```python
[{'canonical': 'Tesla', 'surface_form': '$TSLA', 'method': 'cashtag', 'position': (15,20)},
 {'canonical': 'Tesla', 'surface_form': 'Elon', 'method': 'informal_alias', 'position': (45,49)}]
```

### `src/extraction/normalizer.py` — `EntityNormalizer`

Maps all variations to single lowercase canonical. Key principle: **metrics without normalization are artificially low** — must show side-by-side comparison.

```python
mappings = {
    'apple':    ['apple','apple inc','aapl','$aapl','#aapl'],
    'tesla':    ['tesla','tesla motors','tsla','$tsla','#tsla'],
    'google':   ['google','alphabet','goog','googl','$goog','$googl'],
    'nvidia':   ['nvidia','nvda','$nvda','#nvda'],
    'meta':     ['meta','meta platforms','facebook','fb','$meta'],
    'gamestop': ['gamestop','gme','$gme','#gme','game stop'],
    's&p 500 etf': ['spy','$spy','s&p 500','s&p500','sp500'],
    ...}
```

Methods: `normalize(entity)`, `normalize_set(entities)`, `entities_match(a, b)`

---

## EVALUATION FRAMEWORK

### `src/evaluation/classification.py`

`evaluate_classification(y_true, y_pred, texts, confidence_scores, gold_metadata)` returns:
```python
{'report': dict,  # sklearn classification_report
 'confusion_matrix': DataFrame,
 'accuracy': float, 'weighted_f1': float,
 'per_class': {'bullish': {'precision':x,'recall':x,'f1':x,'support':n}, ...},
 'errors': [{'text','true_label','predicted_label','confidence',
              'error_category','ambiguity_score'}],
 'summary': {'most_confused_pair': ('meme','bullish'),
             'avg_confidence_correct': float, 'avg_confidence_incorrect': float,
             'error_rate_by_ambiguity': {1:0.05, 2:0.12, 3:0.25, 4:0.45, 5:0.60}}}
```

### `src/evaluation/extraction.py`

`evaluate_extraction(predictions, ground_truths, normalizer)` returns:
```python
{'metrics': {'precision','recall','f1'},
 'metrics_without_normalization': {...},   # SHOW THIS to prove normalization matters
 'normalization_lift': {...},
 'per_entity_performance': {'apple': {'tp','fp','fn','f1'}, ...},
 'errors': [{'post_idx','text','predicted','ground_truth',
             'false_positives','false_negatives',
             'error_type'}]}  # 'over_extraction'|'missed'|'normalization_gap'
```

### `src/evaluation/label_quality.py`

`run_thesis_experiment(df_programmatic, df_gold, config)` — see thesis experiment above.

Returns:
```python
{'results_table': DataFrame,  # columns: label_source, f1, precision, recall
 'per_class_comparison': nested dict,
 'thesis_validated': bool,  # is programmatic > noisy?
 'programmatic_vs_gold_gap': float,
 'visualization_data': dict}  # ready for Plotly bar chart
```

---

## STREAMLIT DASHBOARD

Run: `streamlit run app/streamlit_app.py`

**Home page:** 4 metric cards (total posts, labeling coverage %, model F1, thesis result), label distribution pie, data source composition, thesis bar chart.

### Page 1: Data Ingestion
- Source toggle (Live/Synthetic/Auto), date range picker, "Ingest Data" button, source status indicators
- Results: post count per source, filterable sample table, text length distribution, posts-over-time line chart, source pie, data quality audit (nulls, duplicates, avg length)

### Page 2: Labeling Studio
- Function explorer: all 18 functions with coverage %, label distribution, gold precision, expandable examples
- Interactive labeler: text input → shows which functions fired, votes, aggregated result, vote breakdown bar
- Conflict explorer: disagreement table filterable by conflict type
- "Run Labeling" button with progress bar + summary stats

### Page 3: Label Quality
- **HERO VIZ:** Thesis bar chart (gold vs programmatic vs noisy vs random F1), annotated "Same model. Different data quality."
- Label quality metrics: coverage, conflict rate, confidence histogram, per-function accuracy heatmap
- Programmatic vs Gold: confusion matrix, per-class agreement, clickable disagreement explorer
- Uncertain posts table (sorted by confidence), coverage heatmap (posts × functions)

### Page 4: Model Training
- Config: label source selector, hyperparameter controls (max_features, ngram_range, C), "Train Model" button
- Results: classification report, confusion matrix heatmap, cross-val scores
- Feature importance: top 15 per class (bar chart), suspicious feature flagging
- Version list with compare button + best model indicator

### Page 5: Entity Extraction
- Entity P/R/F1, most-mentioned companies bar chart
- **Side-by-side normalization impact** (with vs without — proves why it matters)
- Extraction evidence viewer: text | entities | method | confidence
- Error analysis (FP/FN categorized), interactive text input extractor

### Page 6: Evaluation Deep Dive
- All metrics (classification + extraction + label quality) in one dashboard
- Filterable error explorer (by class, confidence range); errors tagged by category
- F1 vs ambiguity score chart (expected: good on easy posts, struggles on ambiguous)
- Auto-generated data improvement recommendations from error patterns

### Page 7: Live Inference
- Text input → predicted label + confidence bar, probability distribution chart, extracted tickers + evidence, labeling functions that would fire, top driving features
- Batch: upload CSV → download with predictions + confidence + entities
- Session prediction log

---

## VISUAL DESIGN

```python
COLORS = {
    'bullish': '#00C853', 'bearish': '#FF1744', 'neutral': '#78909C', 'meme': '#FFD600',
    'primary': '#58A6FF', 'secondary': '#8B949E', 'success': '#3FB950',
    'warning': '#D29922', 'danger': '#F85149',
    'bg_primary': '#0D1117', 'bg_secondary': '#161B22', 'bg_tertiary': '#21262D',
    'text_primary': '#E6EDF3', 'text_secondary': '#8B949E', 'border': '#30363D',
}
```
All Plotly charts: `template='plotly_dark'`. Streamlit: custom CSS for card-based metric displays.

---

## CLI AND MAKEFILE

```makefile
setup:    pip install -r requirements.txt && python scripts/setup.py
ingest:   python scripts/ingest.py --days 7
label:    python scripts/label.py
train:    python scripts/train.py --source programmatic
evaluate: python scripts/evaluate.py
run:      streamlit run app/streamlit_app.py
pipeline: python scripts/run_pipeline.py  # ingest → label → train → evaluate
all:      ingest label train evaluate run
clean:    rm -rf data/raw/* data/labeled/* data/models/*
test:     pytest tests/ -v
```

`scripts/run_pipeline.py` steps: ingest → label → aggregate → assess quality → train → thesis experiment → extract entities → evaluate → print summary → save artifacts.

---

## REQUIREMENTS

```
pandas>=2.0.0 / numpy>=1.24.0 / scikit-learn>=1.3.0 / plotly>=5.18.0
streamlit>=1.29.0 / praw>=7.7.0 / requests>=2.31.0 / python-dotenv>=1.0.0
pyyaml>=6.0 / joblib>=1.3.0 / wordcloud>=1.9.0
```
