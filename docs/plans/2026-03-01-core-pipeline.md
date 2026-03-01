# MarketPulse Core Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the end-to-end ML pipeline — synthetic data, programmatic labeling, model training, entity extraction, evaluation with thesis experiment — runnable via `python scripts/run_pipeline.py`.

**Architecture:** Bottom-up build: scaffolding → config/utils → ingestion (synthetic-only with stubs) → labeling functions + aggregator → ML pipeline → entity extraction → evaluation → CLI scripts. Each layer depends only on completed layers below it.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, PyYAML, python-dotenv, joblib

---

### Task 1: Project Scaffolding

**Files:**
- Create: all directories, `__init__.py` files, `requirements.txt`, `setup.py`, `.env.example`, `.gitignore`, `Makefile`

**Step 1: Create directory structure**

```bash
mkdir -p config data/{raw,labeled,gold,synthetic,models} src/{ingestion,labeling,models,extraction,evaluation,utils} scripts tests
```

**Step 2: Create all `__init__.py` files**

```bash
touch src/__init__.py src/ingestion/__init__.py src/labeling/__init__.py src/models/__init__.py src/extraction/__init__.py src/evaluation/__init__.py src/utils/__init__.py
```

**Step 3: Create `requirements.txt`**

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
streamlit>=1.29.0
praw>=7.7.0
requests>=2.31.0
python-dotenv>=1.0.0
pyyaml>=6.0
joblib>=1.3.0
```

**Step 4: Create `.env.example`**

```
# Reddit API (https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=MarketPulse/1.0

# Stocktwits (https://api.stocktwits.com/developers)
STOCKTWITS_ACCESS_TOKEN=your_token

# News API (https://newsapi.org/)
NEWS_API_KEY=your_key

# Optional: Alpha Vantage for price data context
ALPHA_VANTAGE_KEY=your_key
```

**Step 5: Create `.gitignore`**

```
data/raw/*
data/models/*.pkl
data/models/*.json
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
```

Keep `data/synthetic/` and `data/gold/` tracked.

**Step 6: Create `Makefile`**

As specified in CLAUDE.md.

**Step 7: Create `setup.py`**

Minimal setup.py with `find_packages()`.

**Step 8: Commit**

```bash
git add -A && git commit -m "feat: project scaffolding — directories, config, requirements"
```

---

### Task 2: Configuration and Utilities

**Files:**
- Create: `config/default.yaml`, `src/utils/config.py`, `src/utils/logger.py`

**Step 1: Create `config/default.yaml`**

Copy the full YAML config from CLAUDE.md spec exactly.

**Step 2: Create `src/utils/config.py`**

```python
import yaml
import os
from dotenv import load_dotenv

def load_config(path="config/default.yaml"):
    """Load YAML config and merge with environment variables."""
    load_dotenv()
    with open(path) as f:
        config = yaml.safe_load(f)
    return config
```

**Step 3: Create `src/utils/logger.py`**

```python
import logging

def get_logger(name):
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

**Step 4: Verify imports work**

```bash
python -c "from src.utils.config import load_config; print(load_config()['project']['name'])"
```

Expected: `MarketPulse`

**Step 5: Commit**

```bash
git add config/ src/utils/ && git commit -m "feat: config loader and logging utilities"
```

---

### Task 3: Ingestion Framework — Base Class and Stubs

**Files:**
- Create: `src/ingestion/base.py`, `src/ingestion/reddit.py`, `src/ingestion/stocktwits.py`, `src/ingestion/news.py`

**Step 1: Create `src/ingestion/base.py`**

Implement `BaseIngester` ABC with `REQUIRED_COLUMNS`, abstract `ingest()` and `is_available()`, and working `validate_output()` that drops null texts and deduplicates by post_id.

**Step 2: Create stub ingesters**

Each of `reddit.py`, `stocktwits.py`, `news.py` should:
- Inherit `BaseIngester`
- `__init__` stores config
- `is_available()` returns `False` (no API keys)
- `ingest()` returns empty DataFrame with correct columns

**Step 3: Verify stubs**

```bash
python -c "
from src.utils.config import load_config
from src.ingestion.reddit import RedditIngester
cfg = load_config()
r = RedditIngester(cfg)
print(f'Reddit available: {r.is_available()}')
print(f'Columns: {list(r.ingest(None, None).columns)}')
"
```

Expected: `Reddit available: False` and correct column list.

**Step 4: Commit**

```bash
git add src/ingestion/ && git commit -m "feat: ingestion base class and API source stubs"
```

---

### Task 4: Synthetic Data Generator

**Files:**
- Create: `src/ingestion/synthetic.py`
- Create: `data/synthetic/synthetic_posts.csv`
- Create: `data/gold/gold_standard.csv`

This is the largest single task. The generator must produce 500+ realistic financial social media posts following the detailed spec in CLAUDE.md.

**Step 1: Implement `SyntheticIngester`**

- `__init__`: store config, set paths to `data/synthetic/synthetic_posts.csv`
- `is_available()`: return `True`
- `ingest()`: load CSV from disk, filter by date range if provided
- `generate()`: create 500+ posts and 100-post gold standard

**Step 2: Implement `generate()` method**

Generate posts with these distributions:
- 150 bullish (30%): conviction buys, technical analysis, earnings optimism, dip buyers, options bulls, subtle bulls
- 100 bearish (20%): short sellers, put buyers, warnings, fundamental bears, macro bears, subtle bears
- 125 neutral (25%): questions, news sharing, analysis, educational, discussion starters, comparative
- 125 meme (25%): loss porn, self-deprecating, hype, ironic, cultural, sarcastic

Plus 50+ edge cases distributed across categories per the spec.

Text pattern requirements:
- 70%+ include ticker symbols ($TSLA, AAPL, etc.)
- 40%+ include emojis
- 30%+ include ALL CAPS words
- 25%+ include informal language (ngl, imo, tbh, lol)
- 15%+ include multiple tickers
- 30%+ include numbers/prices
- 20%+ include options language
- Varied post lengths (10-280 chars)

Each post has: post_id, text, source='synthetic', timestamp, author, score, url='', metadata (dict with `true_sentiment` key).

Score distribution: power law (most posts 1-20, few viral posts 500+).
Timestamps: spread across 7-day window.

**Step 3: Generate gold standard**

Select 100 posts: 25 per category, at least 20 edge cases.
CSV columns: `post_id,text,sentiment_gold,tickers_gold,ambiguity_score,notes`
- ambiguity_score: 1-5 (1=crystal clear, 5=genuinely ambiguous)

**Step 4: Run generator and save files**

```bash
python -c "
from src.utils.config import load_config
from src.ingestion.synthetic import SyntheticIngester
cfg = load_config()
s = SyntheticIngester(cfg)
s.generate()
df = s.ingest(None, None)
print(f'Total posts: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Sources: {df.source.unique()}')
"
```

Expected: 500+ posts, correct columns, source='synthetic'.

**Step 5: Verify gold standard**

```bash
python -c "
import pandas as pd
gold = pd.read_csv('data/gold/gold_standard.csv')
print(f'Gold posts: {len(gold)}')
print(f'Distribution: {gold.sentiment_gold.value_counts().to_dict()}')
print(f'Ambiguity range: {gold.ambiguity_score.min()}-{gold.ambiguity_score.max()}')
"
```

Expected: 100 posts, 25 per class, ambiguity 1-5.

**Step 6: Commit**

```bash
git add src/ingestion/synthetic.py data/synthetic/ data/gold/ && git commit -m "feat: synthetic data generator with 500+ posts and gold standard"
```

---

### Task 5: Ingestion Manager

**Files:**
- Create: `src/ingestion/manager.py`

**Step 1: Implement `IngestionManager`**

- `__init__`: takes config, creates all source instances
- `ingest()`: checks mode (auto/live/synthetic), tries live sources, falls back to synthetic
- `get_source_summary()`: returns stats dict

In "auto" mode with no API keys, it should try live sources, find none available, and fall back to synthetic.

**Step 2: Verify end-to-end ingestion**

```bash
python -c "
from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
cfg = load_config()
mgr = IngestionManager(cfg)
df = mgr.ingest()
print(f'Posts: {len(df)}')
summary = mgr.get_source_summary()
print(f'Summary: {summary}')
"
```

Expected: 500+ posts from synthetic, summary shows fallback used.

**Step 3: Commit**

```bash
git add src/ingestion/manager.py && git commit -m "feat: ingestion manager with auto-fallback to synthetic"
```

---

### Task 6: Labeling Functions

**Files:**
- Create: `src/labeling/functions.py`

**Step 1: Implement all labeling functions**

Copy all 18 functions from the CLAUDE.md spec. They are fully implemented there:
- 4 keyword functions (bullish, bearish, neutral, meme)
- 3 emoji functions (bullish, bearish, meme)
- 3 structural functions (question, short_post, all_caps_ratio)
- 4 financial pattern functions (options_directional, price_target, loss_reporting, news_language)
- 2 sarcasm functions (sarcasm_indicators, self_deprecating)
- 2 metadata functions (stocktwits_sentiment, reddit_flair)

Include constants: BULLISH, BEARISH, NEUTRAL, MEME, ABSTAIN
Include registries: LABELING_FUNCTIONS, METADATA_FUNCTIONS

**Step 2: Quick smoke test**

```bash
python -c "
from src.labeling.functions import *
print(lf_keyword_bullish('buying TSLA calls'))  # bullish
print(lf_keyword_bearish('crash incoming'))      # bearish
print(lf_emoji_meme('GME 💎🙌'))                # meme
print(lf_question_structure('thoughts?'))        # neutral
print(lf_keyword_bullish('nice weather'))        # -1 (ABSTAIN)
print(f'Total functions: {len(LABELING_FUNCTIONS)}')
"
```

**Step 3: Commit**

```bash
git add src/labeling/functions.py && git commit -m "feat: 18 labeling functions for financial sentiment heuristics"
```

---

### Task 7: Label Aggregator

**Files:**
- Create: `src/labeling/aggregator.py`

**Step 1: Implement `LabelAggregator`**

Three strategies:
- `_majority_vote`: count votes, most common wins, ties broken by priority [neutral, bullish, bearish, meme]
- `_weighted_vote`: per-function weights (options_directional=3.0, keyword_*=2.0, emoji_*=1.0, structural=1.5, etc.)
- `_confidence_weighted`: weighted vote + threshold check. Confidence = max_weight_sum / total_weight_sum. If below `confidence_threshold` from config (0.6), set label=None.

`aggregate_single(text, metadata=None)` returns dict with: final_label, confidence, votes, num_votes, num_abstains, has_conflict, competing_labels.

`aggregate_batch(df)` processes entire DataFrame, adding columns: programmatic_label, label_confidence, label_coverage, label_conflict, vote_breakdown.

**Step 2: Verify aggregation**

```bash
python -c "
from src.labeling.aggregator import LabelAggregator
agg = LabelAggregator(strategy='confidence_weighted')
result = agg.aggregate_single('buying TSLA calls 🚀')
print(f'Label: {result[\"final_label\"]}')
print(f'Confidence: {result[\"confidence\"]:.2f}')
print(f'Votes: {result[\"votes\"]}')
"
```

Expected: bullish label with reasonable confidence.

**Step 3: Verify batch**

```bash
python -c "
from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
from src.labeling.aggregator import LabelAggregator
cfg = load_config()
mgr = IngestionManager(cfg)
df = mgr.ingest()
agg = LabelAggregator(strategy='confidence_weighted', config=cfg)
labeled = agg.aggregate_batch(df)
print(f'Labeled posts: {labeled.programmatic_label.notna().sum()}/{len(labeled)}')
print(f'Label distribution:\\n{labeled.programmatic_label.value_counts()}')
"
```

**Step 4: Commit**

```bash
git add src/labeling/aggregator.py && git commit -m "feat: label aggregator with majority, weighted, and confidence-weighted strategies"
```

---

### Task 8: Label Quality Analyzer

**Files:**
- Create: `src/labeling/quality.py`

**Step 1: Implement `LabelQualityAnalyzer`**

Methods:
- `per_function_report(df)`: for each LF, compute coverage %, label distribution, conflict rate with final label. Returns DataFrame.
- `aggregate_quality_report(df)`: total coverage, avg votes/post, conflict rate, confidence distribution, label distribution.
- `compare_to_gold(df, gold_df)`: merge on post_id, compute per-class P/R/F1, confusion matrix, list disagreements.

**Step 2: Verify**

```bash
python -c "
import pandas as pd
from src.labeling.quality import LabelQualityAnalyzer
from src.labeling.functions import LABELING_FUNCTIONS
from src.labeling.aggregator import LabelAggregator
from src.ingestion.manager import IngestionManager
from src.utils.config import load_config
cfg = load_config()
mgr = IngestionManager(cfg)
df = mgr.ingest()
agg = LabelAggregator(config=cfg)
df = agg.aggregate_batch(df)
analyzer = LabelQualityAnalyzer(LABELING_FUNCTIONS, agg)
report = analyzer.aggregate_quality_report(df)
print(f'Coverage: {report[\"total_coverage\"]:.1%}')
print(f'Avg votes: {report[\"avg_votes_per_post\"]:.1f}')
"
```

**Step 3: Commit**

```bash
git add src/labeling/quality.py && git commit -m "feat: label quality analyzer with per-function and aggregate reports"
```

---

### Task 9: ML Training Pipeline

**Files:**
- Create: `src/models/pipeline.py`

**Step 1: Implement `SentimentPipeline`**

- `preprocess(text)`: lowercase + whitespace normalization only
- `train(texts, labels, validation_split=True)`: fit TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=3) + LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000). Split 80/20 if validation_split. Run 5-fold cross-val. Store metadata.
- `predict(texts)`: return list of dicts with label, confidence, probabilities
- `predict_single(text)`: convenience wrapper
- `get_feature_importance(top_n=15)`: extract top features per class from model coefficients
- `error_analysis(texts, true_labels, predicted_labels)`: categorize every misclassification
- `save(path)` / `load(path)`: joblib for vectorizer+model, JSON for metadata

**Step 2: Verify training**

```bash
python -c "
from src.utils.config import load_config
from src.ingestion.manager import IngestionManager
from src.labeling.aggregator import LabelAggregator
from src.models.pipeline import SentimentPipeline
cfg = load_config()
df = IngestionManager(cfg).ingest()
df = LabelAggregator(config=cfg).aggregate_batch(df)
labeled = df[df.programmatic_label.notna()]
pipeline = SentimentPipeline(cfg.get('model', {}))
report = pipeline.train(labeled.text.tolist(), labeled.programmatic_label.tolist())
print(f'Validation F1: {report[\"validation_metrics\"][\"weighted_f1\"]:.3f}')
pred = pipeline.predict_single('buying TSLA calls tomorrow')
print(f'Prediction: {pred[\"label\"]} ({pred[\"confidence\"]:.2f})')
"
```

**Step 3: Verify save/load**

```bash
python -c "
from src.models.pipeline import SentimentPipeline
p = SentimentPipeline()
p.load('data/models')
pred = p.predict_single('crash is coming sell everything')
print(f'{pred[\"label\"]} ({pred[\"confidence\"]:.2f})')
"
```

**Step 4: Commit**

```bash
git add src/models/pipeline.py && git commit -m "feat: TF-IDF + LogReg sentiment pipeline with train, predict, save/load"
```

---

### Task 10: Model Versioning

**Files:**
- Create: `src/models/versioning.py`

**Step 1: Implement `ModelVersion`**

- `save_version(pipeline, label_source, metrics, notes)`: creates `data/models/v{N}_{source}_{timestamp}/` with vectorizer, model, metadata JSON
- `list_versions()`: scan model_dir for version folders, return summary list
- `compare_versions(version_ids)`: load metadata for each, return comparison DataFrame
- `load_version(version_id)`: load specific version into SentimentPipeline

**Step 2: Commit**

```bash
git add src/models/versioning.py && git commit -m "feat: model versioning with save, list, compare"
```

---

### Task 11: Ticker Extractor

**Files:**
- Create: `src/extraction/ticker_extractor.py`

**Step 1: Implement `TickerExtractor`**

Include all maps from spec: ticker_map, ambiguous_tickers, company_aliases, informal_aliases, emoji_map.

Methods:
- `_extract_cashtags(text)`: regex for `$[A-Z]{1,5}`, filter out `$\d+` (prices), return list of (ticker, span)
- `_extract_bare_tickers(text)`: regex for standalone ALL-CAPS 2-5 letter words matching known tickers not in ambiguous set
- `_extract_company_names(text)`: case-insensitive search for company_aliases keys
- `_extract_informal(text)`: case-insensitive search for informal_aliases keys
- `_extract_emoji(text)`: scan for emoji_map keys
- `extract(text)`: combine all methods, return deduplicated list of canonical names
- `extract_with_evidence(text)`: combine all methods, return list of dicts with canonical, surface_form, method, position

**Step 2: Verify**

```bash
python -c "
from src.extraction.ticker_extractor import TickerExtractor
te = TickerExtractor()
print(te.extract('buying \$TSLA and \$AAPL, Elon is a genius'))
print(te.extract_with_evidence('NVDA calls, Jensen is killing it 🍎'))
"
```

Expected: ['Tesla', 'Apple'] for first; detailed evidence for second.

**Step 3: Commit**

```bash
git add src/extraction/ticker_extractor.py && git commit -m "feat: ticker extractor with cashtag, bare ticker, alias, and emoji methods"
```

---

### Task 12: Entity Normalizer

**Files:**
- Create: `src/extraction/normalizer.py`

**Step 1: Implement `EntityNormalizer`**

Build canonical_map from all known variations. Methods:
- `normalize(entity)`: strip $/#, lowercase, lookup in map
- `normalize_set(entities)`: normalize + deduplicate
- `entities_match(a, b)`: compare normalized forms

**Step 2: Verify**

```bash
python -c "
from src.extraction.normalizer import EntityNormalizer
n = EntityNormalizer()
print(n.normalize('\$TSLA'))   # tesla
print(n.normalize('Tesla'))    # tesla
print(n.entities_match('\$TSLA', 'Tesla Inc'))  # True
print(n.normalize_set(['AAPL', '\$AAPL', 'Apple', 'apple inc']))  # ['apple']
"
```

**Step 3: Commit**

```bash
git add src/extraction/normalizer.py && git commit -m "feat: entity normalizer mapping all ticker/company variations to canonical forms"
```

---

### Task 13: Classification Evaluation

**Files:**
- Create: `src/evaluation/classification.py`

**Step 1: Implement `evaluate_classification()`**

Takes y_true, y_pred, optional texts/confidence_scores/gold_metadata.
Returns dict with:
- `report`: sklearn classification_report as dict
- `confusion_matrix`: as labeled DataFrame
- `accuracy`, `weighted_f1`
- `per_class`: per-class P/R/F1/support
- `errors`: list of misclassifications with text, labels, confidence, error category
- `summary`: most confused pair, avg confidence correct vs incorrect

**Step 2: Commit**

```bash
git add src/evaluation/classification.py && git commit -m "feat: classification evaluation with error analysis"
```

---

### Task 14: Extraction Evaluation

**Files:**
- Create: `src/evaluation/extraction.py`

**Step 1: Implement `evaluate_extraction()`**

Takes predictions (list of sets), ground_truths (list of sets), normalizer.
Returns dict with:
- `metrics`: P/R/F1 with normalization
- `metrics_without_normalization`: strict string matching P/R/F1
- `normalization_lift`: difference showing normalization impact
- `per_entity_performance`: per-company TP/FP/FN/F1
- `errors`: list of FPs, FNs, error types

**Step 2: Commit**

```bash
git add src/evaluation/extraction.py && git commit -m "feat: entity extraction evaluation with normalization impact comparison"
```

---

### Task 15: Thesis Experiment (Label Quality Evaluation)

**Files:**
- Create: `src/evaluation/label_quality.py`

**Step 1: Implement `run_thesis_experiment()`**

This is the central experiment. Steps:
1. Load gold standard, split 70/30 train/test
2. Get programmatic labels for gold train posts (from labeled DataFrame)
3. Create noisy labels: take gold train labels, randomly flip 30%
4. Create random labels: completely random for gold train posts
5. Train 4 identical SentimentPipelines on: gold_train, programmatic_train, noisy_train, random_train
6. Evaluate all 4 on gold_test
7. Return results_table DataFrame, per_class_comparison, thesis_validated bool, gap metric

**Step 2: Verify**

```bash
python -c "
from src.evaluation.label_quality import run_thesis_experiment
# (will be called from run_pipeline.py)
print('Thesis experiment module loads successfully')
"
```

**Step 3: Commit**

```bash
git add src/evaluation/label_quality.py && git commit -m "feat: thesis experiment — gold vs programmatic vs noisy vs random labels"
```

---

### Task 16: CLI Scripts

**Files:**
- Create: `scripts/ingest.py`, `scripts/label.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/run_pipeline.py`

**Step 1: Create `scripts/run_pipeline.py`**

The main entry point that chains everything:
1. Load config
2. Ingest data (auto mode → synthetic fallback)
3. Run labeling + aggregation
4. Print label quality report
5. Train model on programmatic labels
6. Save model
7. Run thesis experiment
8. Extract entities from labeled data
9. Print summary

**Step 2: Create individual scripts**

- `ingest.py`: argparse with `--days`, `--source`, calls IngestionManager
- `label.py`: loads ingested data, runs aggregator, saves labeled CSV
- `train.py`: argparse with `--source` (programmatic/gold), trains and saves model
- `evaluate.py`: loads model + gold set, runs full evaluation

**Step 3: Run full pipeline**

```bash
python scripts/run_pipeline.py
```

Expected: Full pipeline runs, prints thesis experiment results showing programmatic labels outperform noisy labels.

**Step 4: Commit**

```bash
git add scripts/ && git commit -m "feat: CLI scripts and full pipeline runner"
```

---

### Task 17: Final Integration Verification

**Step 1: Clean run from scratch**

```bash
rm -rf data/raw/* data/labeled/* data/models/*.pkl data/models/*.json
python scripts/run_pipeline.py
```

Verify:
- Synthetic data generated/loaded
- Labeling completes with coverage stats
- Model trains with reasonable F1
- Thesis experiment shows programmatic > noisy > random
- Entities extracted
- All artifacts saved

**Step 2: Verify Makefile commands**

```bash
make pipeline
```

**Step 3: Final commit**

```bash
git add -A && git commit -m "feat: MarketPulse core pipeline — complete and verified"
```
