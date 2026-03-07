"""
Sentiment Classification Evaluation

Goes beyond sklearn's classification_report to provide
production-relevant analysis.
"""

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def evaluate_classification(y_true, y_pred, texts=None,
                            confidence_scores=None, gold_metadata=None):
    """
    Comprehensive classification evaluation.

    Returns dict with standard metrics plus optional error analysis.
    """
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    weighted_f1 = report.get("weighted avg", {}).get("f1-score", 0.0)
    accuracy = report.get("accuracy", sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1))

    per_class = {}
    for label in labels:
        stats = report.get(label, {})
        per_class[label] = {
            "precision": stats.get("precision", 0.0),
            "recall": stats.get("recall", 0.0),
            "f1": stats.get("f1-score", 0.0),
            "support": stats.get("support", 0),
        }

    errors = []
    if texts is not None:
        for i, (text, true, pred) in enumerate(zip(texts, y_true, y_pred)):
            if true != pred:
                conf = confidence_scores[i] if confidence_scores is not None else None
                ambiguity = None
                if gold_metadata is not None and i < len(gold_metadata):
                    ambiguity = gold_metadata[i].get("ambiguity_score")
                errors.append({
                    "text": text,
                    "true_label": true,
                    "predicted_label": pred,
                    "confidence": conf,
                    "error_category": "labeling_ambiguity" if ambiguity and ambiguity >= 4 else "model_limitation",
                    "ambiguity_score": ambiguity,
                })

    most_confused_pair = None
    if cm.size > 0:
        cm_copy = cm.copy()
        for i in range(len(labels)):
            cm_copy[i][i] = 0
        idx = cm_copy.argmax()
        row, col = divmod(idx, len(labels))
        if cm_copy[row][col] > 0:
            most_confused_pair = (labels[row], labels[col])

    return {
        "report": report,
        "confusion_matrix": cm_df,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "errors": errors,
        "summary": {
            "most_confused_pair": most_confused_pair,
            "avg_confidence_correct": None,
            "avg_confidence_incorrect": None,
            "error_rate_by_ambiguity": {},
        },
    }
