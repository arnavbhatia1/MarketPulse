"""
Tests for SentimentPipeline.
"""

import os
import pytest
import tempfile
from src.models.pipeline import SentimentPipeline


class TestPreprocess:
    def test_lowercase(self):
        p = SentimentPipeline()
        assert p.preprocess("BUYING AAPL") == "buying aapl"

    def test_whitespace_normalized(self):
        p = SentimentPipeline()
        assert p.preprocess("too   many    spaces") == "too many spaces"

    def test_emojis_preserved(self):
        p = SentimentPipeline()
        result = p.preprocess("TSLA 🚀🚀🚀")
        assert "🚀" in result

    def test_tickers_preserved(self):
        p = SentimentPipeline()
        result = p.preprocess("$AAPL going up")
        assert "$aapl" in result

    def test_strips_leading_trailing(self):
        p = SentimentPipeline()
        assert p.preprocess("  hello world  ") == "hello world"


class TestTrainAndPredict:
    def test_train_returns_report(self, trained_pipeline):
        assert trained_pipeline.is_trained is True
        assert trained_pipeline.vectorizer is not None
        assert trained_pipeline.model is not None

    def test_predict_single(self, trained_pipeline):
        result = trained_pipeline.predict_single("Buying AAPL calls, very bullish")
        assert 'label' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['label'] in ['bullish', 'bearish', 'neutral', 'meme']
        assert 0 <= result['confidence'] <= 1

    def test_predict_batch(self, trained_pipeline):
        texts = ["Buying calls", "Selling puts", "What do you think?"]
        results = trained_pipeline.predict(texts)
        assert len(results) == 3
        for r in results:
            assert 'text' in r
            assert 'label' in r
            assert 'confidence' in r
            assert 'probabilities' in r

    def test_probabilities_sum_to_one(self, trained_pipeline):
        result = trained_pipeline.predict_single("Some test text here")
        prob_sum = sum(result['probabilities'].values())
        assert abs(prob_sum - 1.0) < 0.01

    def test_predict_string_input(self, trained_pipeline):
        # predict() should accept a single string
        results = trained_pipeline.predict("Buying AAPL")
        assert len(results) == 1

    def test_predict_untrained_raises(self):
        p = SentimentPipeline()
        with pytest.raises(RuntimeError):
            p.predict("test")


class TestFeatureImportance:
    def test_returns_dict(self, trained_pipeline):
        fi = trained_pipeline.get_feature_importance(top_n=5)
        assert isinstance(fi, dict)
        # Should have one key per class
        for cls in trained_pipeline.model.classes_:
            assert cls in fi

    def test_each_class_has_features(self, trained_pipeline):
        fi = trained_pipeline.get_feature_importance(top_n=5)
        for cls, features in fi.items():
            assert len(features) > 0
            # Each feature is a (name, weight) tuple
            name, weight = features[0]
            assert isinstance(name, str)
            assert isinstance(weight, float)

    def test_untrained_raises(self):
        p = SentimentPipeline()
        with pytest.raises(RuntimeError):
            p.get_feature_importance()


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, trained_pipeline):
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_pipeline.save(tmpdir)

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, 'tfidf_vectorizer.pkl'))
            assert os.path.exists(os.path.join(tmpdir, 'sentiment_model.pkl'))
            assert os.path.exists(os.path.join(tmpdir, 'model_metadata.json'))

            # Load into a new pipeline
            new_pipeline = SentimentPipeline()
            new_pipeline.load(tmpdir)
            assert new_pipeline.is_trained is True

            # Predictions should be identical
            text = "Buying AAPL calls, bullish"
            orig = trained_pipeline.predict_single(text)
            loaded = new_pipeline.predict_single(text)
            assert orig['label'] == loaded['label']
            assert abs(orig['confidence'] - loaded['confidence']) < 0.001

    def test_save_untrained_raises(self):
        p = SentimentPipeline()
        with pytest.raises(RuntimeError):
            p.save("/tmp/test_model")
