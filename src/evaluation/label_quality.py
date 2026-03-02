"""
Label Quality Evaluation -- The Thesis Experiment

Proves: data quality > model complexity by training identical models
on gold, programmatic, noisy, and random labels.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from src.models.pipeline import SentimentPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

LABELS = ['bullish', 'bearish', 'neutral', 'meme']


def run_thesis_experiment(df_programmatic, df_gold, config=None):
    """
    THE CENTRAL EXPERIMENT.

    Train identical models on:
    1. Gold labels (best possible)
    2. Programmatic labels (our pipeline)
    3. Noisy labels (30% random flip of gold)
    4. Random labels (baseline)

    Evaluate all on gold test set.

    Args:
        df_programmatic: DataFrame with 'text' and 'programmatic_label' columns
        df_gold: DataFrame with 'post_id', 'text', 'sentiment_gold' columns
        config: model config dict

    Returns dict with results_table, per_class, thesis_validated, gap.
    """
    config = config or {}
    model_config = config.get('model', config)
    random_state = model_config.get('random_state', 42)
    rng = np.random.RandomState(random_state)

    logger.info("=" * 60)
    logger.info("THESIS EXPERIMENT: Data Quality > Model Complexity")
    logger.info("=" * 60)

    # Split gold into train/test (70/30)
    gold_train, gold_test = train_test_split(
        df_gold, test_size=0.3, random_state=random_state,
        stratify=df_gold['sentiment_gold']
    )

    test_texts = gold_test['text'].tolist()
    test_labels = gold_test['sentiment_gold'].tolist()

    logger.info(f"Gold train: {len(gold_train)}, Gold test: {len(gold_test)}")

    # Prepare 4 training sets
    train_sets = {}

    # 1. Gold labels
    train_sets['gold'] = {
        'texts': gold_train['text'].tolist(),
        'labels': gold_train['sentiment_gold'].tolist(),
    }

    # 2. Programmatic labels -- match gold_train post_ids
    prog_merged = gold_train.merge(
        df_programmatic[['post_id', 'programmatic_label']].dropna(subset=['programmatic_label']),
        on='post_id', how='inner'
    )
    if len(prog_merged) >= 10:
        train_sets['programmatic'] = {
            'texts': prog_merged['text'].tolist(),
            'labels': prog_merged['programmatic_label'].tolist(),
        }
    else:
        # Fallback: use all programmatic labeled data
        prog_labeled = df_programmatic[df_programmatic['programmatic_label'].notna()]
        train_sets['programmatic'] = {
            'texts': prog_labeled['text'].tolist(),
            'labels': prog_labeled['programmatic_label'].tolist(),
        }
    logger.info(f"Programmatic training set: {len(train_sets['programmatic']['texts'])} samples")

    # 3. Noisy labels -- flip 30% of gold train labels randomly
    noisy_labels = gold_train['sentiment_gold'].tolist()
    n_flip = int(len(noisy_labels) * 0.3)
    flip_indices = rng.choice(len(noisy_labels), size=n_flip, replace=False)
    for idx in flip_indices:
        others = [l for l in LABELS if l != noisy_labels[idx]]
        noisy_labels[idx] = rng.choice(others)
    train_sets['noisy'] = {
        'texts': gold_train['text'].tolist(),
        'labels': noisy_labels,
    }

    # 4. Random labels
    random_labels = rng.choice(LABELS, size=len(gold_train)).tolist()
    train_sets['random'] = {
        'texts': gold_train['text'].tolist(),
        'labels': random_labels,
    }

    # Train and evaluate each
    results = []
    per_class_all = {}

    for source_name, data in train_sets.items():
        logger.info(f"\nTraining on {source_name} labels ({len(data['texts'])} samples)...")

        pipeline = SentimentPipeline(model_config)

        try:
            pipeline.train(data['texts'], data['labels'], validation_split=False)

            predictions = pipeline.predict(test_texts)
            pred_labels = [p['label'] for p in predictions]

            weighted_f1 = float(f1_score(test_labels, pred_labels,
                                         average='weighted', zero_division=0))
            report = classification_report(test_labels, pred_labels,
                                          output_dict=True, zero_division=0)

            results.append({
                'label_source': source_name,
                'weighted_f1': weighted_f1,
                'accuracy': report.get('accuracy', 0),
                'train_size': len(data['texts']),
            })

            per_class_all[source_name] = {
                label: {
                    'precision': report.get(label, {}).get('precision', 0),
                    'recall': report.get(label, {}).get('recall', 0),
                    'f1': report.get(label, {}).get('f1-score', 0),
                }
                for label in LABELS if label in report
            }

            logger.info(f"  {source_name}: F1={weighted_f1:.3f}")

        except Exception as e:
            logger.warning(f"  {source_name} training failed: {e}")
            results.append({
                'label_source': source_name,
                'weighted_f1': 0.0,
                'accuracy': 0.0,
                'train_size': len(data['texts']),
            })

    # Build results table
    results_table = pd.DataFrame(results)
    results_table = results_table.sort_values('weighted_f1', ascending=False)

    # Determine if thesis validated
    f1_scores = {r['label_source']: r['weighted_f1'] for r in results}
    thesis_validated = f1_scores.get('programmatic', 0) > f1_scores.get('noisy', 0)
    prog_vs_gold_gap = f1_scores.get('gold', 0) - f1_scores.get('programmatic', 0)

    logger.info("\n" + "=" * 60)
    logger.info("THESIS EXPERIMENT RESULTS")
    logger.info("=" * 60)
    for _, row in results_table.iterrows():
        marker = " <-- OUR PIPELINE" if row['label_source'] == 'programmatic' else ""
        logger.info(f"  {row['label_source']:>15}: F1 = {row['weighted_f1']:.3f}{marker}")
    logger.info(f"\n  Thesis validated: {thesis_validated}")
    logger.info(f"  Programmatic vs Gold gap: {prog_vs_gold_gap:.3f}")
    logger.info("  Same model. Different data quality.")
    logger.info("=" * 60)

    return {
        'results_table': results_table,
        'per_class_comparison': per_class_all,
        'thesis_validated': thesis_validated,
        'programmatic_vs_gold_gap': prog_vs_gold_gap,
        'f1_scores': f1_scores,
    }
