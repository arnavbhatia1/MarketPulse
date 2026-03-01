# MarketPulse Core Pipeline — Phase 1 Design

## Scope

Build the end-to-end ML pipeline: synthetic data generation, programmatic labeling, model training, entity extraction, evaluation, and CLI scripts. No dashboard. No live API sources. Minimal tests.

## Modules

1. **Config/Utils** — `config/default.yaml`, `src/utils/{config,logger}.py`
2. **Ingestion** — `src/ingestion/{base,synthetic,reddit,stocktwits,news,manager}.py`
   - Synthetic generator produces 500+ posts + gold standard (100 posts)
   - Live sources stubbed with `is_available() -> False`
3. **Labeling** — `src/labeling/{functions,aggregator,quality}.py`
   - 18 labeling functions from spec (16 text-only + 2 metadata-aware)
   - Confidence-weighted aggregation
4. **Models** — `src/models/{pipeline,versioning}.py`
   - TF-IDF + Logistic Regression
   - Save/load with joblib
5. **Extraction** — `src/extraction/{ticker_extractor,normalizer}.py`
   - Rule-based ticker extraction with evidence tracking
6. **Evaluation** — `src/evaluation/{classification,extraction,label_quality}.py`
   - Thesis experiment: gold vs programmatic vs noisy vs random
7. **Scripts** — `scripts/{ingest,label,train,evaluate,run_pipeline}.py`

## Data Flow

```
synthetic_posts.csv -> IngestionManager -> raw DataFrame
-> LabelAggregator -> labeled DataFrame
-> SentimentPipeline.train() -> model artifacts
-> evaluate + thesis experiment -> results printed to console
```

## Key Decisions

- Labeling functions copied from CLAUDE.md spec (already written)
- Gold standard generated alongside synthetic data with ambiguity scores
- Live API ingesters are stubs (`is_available()` returns False)
- No Streamlit dashboard in Phase 1
- Pipeline runnable via `python scripts/run_pipeline.py`
