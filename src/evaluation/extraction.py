"""
Entity Extraction Evaluation

Entity-level precision, recall, F1 with normalization.
Demonstrates that normalization is critical for fair evaluation.
"""


def evaluate_extraction(predictions, ground_truths, normalizer):
    """
    Entity-level evaluation with normalization.

    predictions: list of sets of extracted entity strings
    ground_truths: list of sets of gold entity strings
    normalizer: EntityNormalizer instance
    """

    def _compute_prf(preds_list, golds_list, norm=None):
        tp = fp = fn = 0
        per_entity = {}
        for preds, golds in zip(preds_list, golds_list):
            if norm:
                preds = set(norm.normalize(e) for e in preds)
                golds = set(norm.normalize(e) for e in golds)
            for e in preds & golds:
                tp += 1
                per_entity.setdefault(e, {"tp": 0, "fp": 0, "fn": 0})
                per_entity[e]["tp"] += 1
            for e in preds - golds:
                fp += 1
                per_entity.setdefault(e, {"tp": 0, "fp": 0, "fn": 0})
                per_entity[e]["fp"] += 1
            for e in golds - preds:
                fn += 1
                per_entity.setdefault(e, {"tp": 0, "fp": 0, "fn": 0})
                per_entity[e]["fn"] += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        for e in per_entity:
            e_tp = per_entity[e]["tp"]
            e_fp = per_entity[e]["fp"]
            e_fn = per_entity[e]["fn"]
            p = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
            r = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0.0
            per_entity[e]["f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}, per_entity

    norm_metrics, per_entity = _compute_prf(predictions, ground_truths, norm=normalizer)
    unnorm_metrics, _ = _compute_prf(predictions, ground_truths, norm=None)

    f1_lift = norm_metrics["f1"] - unnorm_metrics["f1"]
    p_lift = norm_metrics["precision"] - unnorm_metrics["precision"]
    r_lift = norm_metrics["recall"] - unnorm_metrics["recall"]

    errors = []
    for i, (preds, golds) in enumerate(zip(predictions, ground_truths)):
        norm_preds = set(normalizer.normalize(e) for e in preds)
        norm_golds = set(normalizer.normalize(e) for e in golds)
        fps = norm_preds - norm_golds
        fns = norm_golds - norm_preds
        if fps or fns:
            error_type = ("over_extraction" if fps and not fns
                          else "missed" if fns and not fps
                          else "normalization_gap")
            errors.append({
                "post_idx": i,
                "predicted": norm_preds,
                "ground_truth": norm_golds,
                "false_positives": fps,
                "false_negatives": fns,
                "error_type": error_type,
            })

    return {
        "metrics": norm_metrics,
        "metrics_without_normalization": unnorm_metrics,
        "normalization_lift": {
            "f1_lift": f1_lift,
            "precision_lift": p_lift,
            "recall_lift": r_lift,
        },
        "per_entity_performance": per_entity,
        "errors": errors,
    }
