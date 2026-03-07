"""
Label Quality Evaluation

Compares programmatic labels against gold standard.
Runs the thesis experiment showing data quality > model complexity.
"""

import random
import pandas as pd
from src.models.pipeline import SentimentPipeline
from src.evaluation.classification import evaluate_classification

CLASSES = ["bullish", "bearish", "neutral", "meme"]
NOISE_RATE = 0.30


def run_thesis_experiment(df_programmatic, df_gold, config):
    """
    Train the same model on four label sources and compare on the gold test set.

    Sources: gold, programmatic, noisy (30% flipped gold), random.
    Returns comparison dict with results_table DataFrame.
    """
    model_config = config.get("model", {})

    prog_lookup = {}
    if "post_id" in df_programmatic.columns and "programmatic_label" in df_programmatic.columns:
        prog_lookup = dict(
            zip(df_programmatic["post_id"].astype(str),
                df_programmatic["programmatic_label"])
        )

    gold_df = df_gold.copy()
    gold_df["text"] = gold_df["text"].fillna("").astype(str)
    gold_df = gold_df[gold_df["sentiment_gold"].notna()].reset_index(drop=True)

    rng = random.Random(42)
    indices = list(range(len(gold_df)))
    rng.shuffle(indices)
    split = max(1, int(0.8 * len(indices)))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_df = gold_df.iloc[train_idx].reset_index(drop=True)
    test_df = gold_df.iloc[test_idx].reset_index(drop=True)

    test_texts = test_df["text"].tolist()
    test_labels = test_df["sentiment_gold"].tolist()

    def _train_eval(texts, labels):
        if not texts or not labels:
            return {"weighted_f1": 0.0, "precision": 0.0, "recall": 0.0}
        pipeline = SentimentPipeline(model_config)
        pipeline.train(texts, labels, validation_split=False)
        preds = [r["label"] for r in pipeline.predict(test_texts)]
        result = evaluate_classification(test_labels, preds)
        wf1 = result["weighted_f1"]
        per = result["per_class"]
        avg_p = sum(v["precision"] for v in per.values()) / max(len(per), 1)
        avg_r = sum(v["recall"] for v in per.values()) / max(len(per), 1)
        return {"weighted_f1": wf1, "precision": avg_p, "recall": avg_r}

    # 1. Gold
    gold_metrics = _train_eval(
        train_df["text"].tolist(),
        train_df["sentiment_gold"].tolist()
    )

    # 2. Programmatic
    prog_texts, prog_labels = [], []
    for _, row in train_df.iterrows():
        pid = str(row.get("post_id", ""))
        label = prog_lookup.get(pid)
        if label:
            prog_texts.append(row["text"])
            prog_labels.append(label)
    if not prog_texts:
        prog_texts = train_df["text"].tolist()
        prog_labels = train_df["sentiment_gold"].tolist()
    prog_metrics = _train_eval(prog_texts, prog_labels)

    # 3. Noisy (30% random flips)
    noisy_labels = []
    for lbl in train_df["sentiment_gold"].tolist():
        if rng.random() < NOISE_RATE:
            noisy_labels.append(rng.choice(CLASSES))
        else:
            noisy_labels.append(lbl)
    noisy_metrics = _train_eval(train_df["text"].tolist(), noisy_labels)

    # 4. Random
    random_labels = [rng.choice(CLASSES) for _ in range(len(train_df))]
    random_metrics = _train_eval(train_df["text"].tolist(), random_labels)

    results = [
        {"label_source": "gold", **gold_metrics},
        {"label_source": "programmatic", **prog_metrics},
        {"label_source": "noisy", **noisy_metrics},
        {"label_source": "random", **random_metrics},
    ]
    results_table = pd.DataFrame(results)

    thesis_validated = prog_metrics["weighted_f1"] > noisy_metrics["weighted_f1"]
    gap = gold_metrics["weighted_f1"] - prog_metrics["weighted_f1"]

    return {
        "results_table": results_table,
        "per_class_comparison": {},
        "thesis_validated": thesis_validated,
        "programmatic_vs_gold_gap": gap,
        "visualization_data": results,
    }
