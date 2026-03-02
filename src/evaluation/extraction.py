"""
Entity Extraction Evaluation

Entity-level precision, recall, F1 with and without normalization.
"""

from collections import Counter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_extraction(predictions, ground_truths, normalizer=None):
    """
    Entity-level evaluation.

    Args:
        predictions: list of sets/lists of extracted entities per post
        ground_truths: list of sets/lists of gold entities per post
        normalizer: EntityNormalizer instance (optional)

    Returns dict with metrics, metrics_without_normalization,
    normalization_lift, per_entity_performance, errors.
    """
    # With normalization
    if normalizer:
        norm_preds = [set(normalizer.normalize_set(p)) for p in predictions]
        norm_truths = [set(normalizer.normalize_set(g)) for g in ground_truths]
    else:
        norm_preds = [set(str(x).lower() for x in p) for p in predictions]
        norm_truths = [set(str(x).lower() for x in g) for g in ground_truths]

    metrics_norm = _compute_entity_metrics(norm_preds, norm_truths)

    # Without normalization (strict string matching)
    raw_preds = [set(str(x) for x in p) for p in predictions]
    raw_truths = [set(str(x) for x in g) for g in ground_truths]
    metrics_raw = _compute_entity_metrics(raw_preds, raw_truths)

    # Normalization lift
    lift = {
        'precision_lift': metrics_norm['precision'] - metrics_raw['precision'],
        'recall_lift': metrics_norm['recall'] - metrics_raw['recall'],
        'f1_lift': metrics_norm['f1'] - metrics_raw['f1'],
    }

    # Per-entity performance
    entity_tp = Counter()
    entity_fp = Counter()
    entity_fn = Counter()

    for pred_set, truth_set in zip(norm_preds, norm_truths):
        for e in pred_set & truth_set:
            entity_tp[e] += 1
        for e in pred_set - truth_set:
            entity_fp[e] += 1
        for e in truth_set - pred_set:
            entity_fn[e] += 1

    all_entities = set(entity_tp.keys()) | set(entity_fp.keys()) | set(entity_fn.keys())
    per_entity = {}
    for e in all_entities:
        tp = entity_tp[e]
        fp = entity_fp[e]
        fn = entity_fn[e]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_entity[e] = {'tp': tp, 'fp': fp, 'fn': fn, 'f1': round(f1, 3)}

    # Error analysis
    errors = []
    for i, (pred_set, truth_set) in enumerate(zip(norm_preds, norm_truths)):
        fps = pred_set - truth_set
        fns = truth_set - pred_set
        if fps or fns:
            error_type = 'over_extraction' if fps and not fns else \
                         'missed' if fns and not fps else 'mixed'
            errors.append({
                'post_idx': i,
                'predicted': sorted(pred_set),
                'ground_truth': sorted(truth_set),
                'false_positives': sorted(fps),
                'false_negatives': sorted(fns),
                'error_type': error_type,
            })

    result = {
        'metrics': metrics_norm,
        'metrics_without_normalization': metrics_raw,
        'normalization_lift': lift,
        'per_entity_performance': per_entity,
        'errors': errors,
    }

    logger.info(f"Extraction eval: P={metrics_norm['precision']:.3f}, "
                f"R={metrics_norm['recall']:.3f}, F1={metrics_norm['f1']:.3f} "
                f"(norm lift: +{lift['f1_lift']:.3f})")
    return result


def _compute_entity_metrics(predictions, ground_truths):
    """Compute micro-averaged entity-level P/R/F1."""
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred_set, truth_set in zip(predictions, ground_truths):
        total_tp += len(pred_set & truth_set)
        total_fp += len(pred_set - truth_set)
        total_fn += len(truth_set - pred_set)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1': round(float(f1), 4),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
    }
