"""
Sentiment Classification Evaluation
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_classification(y_true, y_pred, texts=None, confidence_scores=None, gold_metadata=None):
    """
    Comprehensive classification evaluation.

    Returns dict with report, confusion_matrix, accuracy, weighted_f1,
    per_class metrics, errors list, and summary.
    """
    labels = sorted(set(list(y_true) + list(y_pred)))

    # Classification report
    report = classification_report(y_true, y_pred, labels=labels,
                                   output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # Per-class metrics
    per_class = {}
    for label in labels:
        if label in report:
            per_class[label] = {
                'precision': report[label]['precision'],
                'recall': report[label]['recall'],
                'f1': report[label]['f1-score'],
                'support': report[label]['support'],
            }

    # Error analysis
    errors = []
    correct_confs = []
    incorrect_confs = []
    confusion_pairs = Counter()

    for i in range(len(y_true)):
        conf = confidence_scores[i] if confidence_scores and i < len(confidence_scores) else None

        if y_true[i] == y_pred[i]:
            if conf is not None:
                correct_confs.append(conf)
            continue

        if conf is not None:
            incorrect_confs.append(conf)

        confusion_pairs[(y_true[i], y_pred[i])] += 1

        # Categorize error
        ambiguity = None
        if gold_metadata and i < len(gold_metadata):
            ambiguity = gold_metadata[i].get('ambiguity_score')

        if ambiguity and ambiguity >= 4:
            category = 'labeling_ambiguity'
        elif conf and conf < 0.4:
            category = 'model_uncertainty'
        else:
            category = 'model_limitation'

        error = {
            'true_label': y_true[i],
            'predicted_label': y_pred[i],
            'confidence': conf,
            'error_category': category,
            'ambiguity_score': ambiguity,
        }
        if texts and i < len(texts):
            error['text'] = texts[i][:200]
        errors.append(error)

    # Summary
    most_confused = confusion_pairs.most_common(1)[0] if confusion_pairs else None

    # Error rate by ambiguity
    error_by_ambiguity = {}
    if gold_metadata:
        for score in range(1, 6):
            indices = [i for i, m in enumerate(gold_metadata)
                      if m.get('ambiguity_score') == score]
            if indices:
                errs = sum(1 for i in indices if y_true[i] != y_pred[i])
                error_by_ambiguity[score] = errs / len(indices)

    result = {
        'report': report,
        'confusion_matrix': cm_df,
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'per_class': per_class,
        'errors': errors,
        'summary': {
            'most_confused_pair': most_confused,
            'avg_confidence_correct': float(np.mean(correct_confs)) if correct_confs else None,
            'avg_confidence_incorrect': float(np.mean(incorrect_confs)) if incorrect_confs else None,
            'error_rate_by_ambiguity': error_by_ambiguity,
        },
    }

    logger.info(f"Classification eval: accuracy={accuracy:.3f}, weighted_f1={weighted_f1:.3f}, "
                f"errors={len(errors)}/{len(y_true)}")
    return result
