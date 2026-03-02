"""
Tests for classification and extraction assessment functions.
"""

import pytest
from src.evaluation.classification import evaluate_classification
from src.evaluation.extraction import evaluate_extraction
from src.extraction.normalizer import EntityNormalizer


class TestClassificationMetrics:
    @pytest.fixture
    def basic_result(self):
        y_true = ['bullish', 'bearish', 'neutral', 'meme', 'bullish', 'bearish']
        y_pred = ['bullish', 'bearish', 'neutral', 'meme', 'bearish', 'bearish']
        return evaluate_classification(y_true, y_pred)

    def test_returns_expected_keys(self, basic_result):
        expected = {'report', 'confusion_matrix', 'accuracy', 'weighted_f1',
                    'per_class', 'errors', 'summary'}
        assert set(basic_result.keys()) == expected

    def test_accuracy_range(self, basic_result):
        assert 0 <= basic_result['accuracy'] <= 1

    def test_weighted_f1_range(self, basic_result):
        assert 0 <= basic_result['weighted_f1'] <= 1

    def test_confusion_matrix_shape(self, basic_result):
        cm = basic_result['confusion_matrix']
        # Should be square
        assert cm.shape[0] == cm.shape[1]

    def test_per_class_has_metrics(self, basic_result):
        for cls, metrics in basic_result['per_class'].items():
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert 'support' in metrics

    def test_errors_detected(self, basic_result):
        # One error: bullish predicted as bearish
        assert len(basic_result['errors']) == 1
        assert basic_result['errors'][0]['true_label'] == 'bullish'
        assert basic_result['errors'][0]['predicted_label'] == 'bearish'

    def test_perfect_classification(self):
        y = ['bullish', 'bearish', 'neutral', 'meme']
        result = evaluate_classification(y, y)
        assert result['accuracy'] == 1.0
        assert result['weighted_f1'] == 1.0
        assert len(result['errors']) == 0

    def test_with_texts(self):
        y_true = ['bullish', 'bearish']
        y_pred = ['bearish', 'bearish']
        texts = ['text1', 'text2']
        result = evaluate_classification(y_true, y_pred, texts=texts)
        assert result['errors'][0]['text'] == 'text1'

    def test_with_confidence_scores(self):
        y_true = ['bullish', 'bearish']
        y_pred = ['bearish', 'bearish']
        confs = [0.3, 0.9]
        result = evaluate_classification(y_true, y_pred, confidence_scores=confs)
        assert result['summary']['avg_confidence_correct'] is not None


class TestExtractionMetrics:
    @pytest.fixture
    def basic_extraction_result(self):
        predictions = [{'Apple', 'Tesla'}, {'NVIDIA'}, {'Google'}]
        ground_truths = [{'Apple', 'Tesla'}, {'NVIDIA', 'AMD'}, {'Google'}]
        norm = EntityNormalizer()
        return evaluate_extraction(predictions, ground_truths, norm)

    def test_returns_expected_keys(self, basic_extraction_result):
        expected = {'metrics', 'metrics_without_normalization',
                    'normalization_lift', 'per_entity_performance', 'errors'}
        assert set(basic_extraction_result.keys()) == expected

    def test_metrics_structure(self, basic_extraction_result):
        m = basic_extraction_result['metrics']
        assert 'precision' in m
        assert 'recall' in m
        assert 'f1' in m

    def test_precision_recall_range(self, basic_extraction_result):
        m = basic_extraction_result['metrics']
        assert 0 <= m['precision'] <= 1
        assert 0 <= m['recall'] <= 1
        assert 0 <= m['f1'] <= 1

    def test_normalization_lift_structure(self, basic_extraction_result):
        lift = basic_extraction_result['normalization_lift']
        assert 'precision_lift' in lift
        assert 'recall_lift' in lift
        assert 'f1_lift' in lift

    def test_perfect_extraction(self):
        preds = [{'Apple'}, {'Tesla'}]
        truths = [{'Apple'}, {'Tesla'}]
        norm = EntityNormalizer()
        result = evaluate_extraction(preds, truths, norm)
        assert result['metrics']['precision'] == 1.0
        assert result['metrics']['recall'] == 1.0

    def test_normalization_lift_positive(self):
        # Use different surface forms that normalize to same entity
        preds = [{'Apple'}]
        truths = [{'AAPL'}]
        norm = EntityNormalizer()
        result = evaluate_extraction(preds, truths, norm)
        # With normalization: Apple == AAPL (both normalize to 'apple')
        assert result['metrics']['f1'] > result['metrics_without_normalization']['f1']
        assert result['normalization_lift']['f1_lift'] > 0

    def test_errors_list(self, basic_extraction_result):
        errors = basic_extraction_result['errors']
        # One error: missed AMD in post index 1
        assert len(errors) >= 1
        for err in errors:
            assert 'post_idx' in err
            assert 'error_type' in err
