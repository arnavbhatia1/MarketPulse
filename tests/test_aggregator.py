"""
Tests for the LabelAggregator.
"""

import pytest
import pandas as pd
from src.labeling.aggregator import LabelAggregator
from src.labeling.functions import BULLISH, BEARISH, NEUTRAL, MEME, ABSTAIN


@pytest.fixture
def majority_agg(config):
    return LabelAggregator(strategy="majority", config=config)


@pytest.fixture
def weighted_agg(config):
    return LabelAggregator(strategy="weighted", config=config)


@pytest.fixture
def confidence_agg(config):
    return LabelAggregator(strategy="confidence_weighted", config=config)


class TestAggregateSingle:
    def test_clear_bullish(self, majority_agg):
        result = majority_agg.aggregate_single("Buying AAPL calls, very bullish, loading up 🚀")
        assert result['final_label'] == BULLISH
        assert result['confidence'] > 0
        assert result['num_votes'] > 0

    def test_clear_bearish(self, majority_agg):
        result = majority_agg.aggregate_single("Selling everything, puts loaded, crash incoming 📉")
        assert result['final_label'] == BEARISH

    def test_clear_meme(self, majority_agg):
        result = majority_agg.aggregate_single("Apes together strong, diamond hands, tendies 💎🙌🦍")
        assert result['final_label'] == MEME

    def test_clear_neutral(self, majority_agg):
        result = majority_agg.aggregate_single("What do you think about AAPL earnings report? Thoughts?")
        assert result['final_label'] == NEUTRAL

    def test_all_abstain(self, majority_agg):
        result = majority_agg.aggregate_single("The weather is beautiful today.")
        assert result['num_votes'] == 0
        assert result['final_label'] is None
        assert result['confidence'] == 0.0

    def test_result_structure(self, majority_agg):
        result = majority_agg.aggregate_single("Buying AAPL")
        expected_keys = {
            'final_label', 'confidence', 'votes', 'num_votes',
            'num_abstains', 'has_conflict', 'competing_labels'
        }
        assert set(result.keys()) == expected_keys

    def test_conflict_detection(self, majority_agg):
        # This post has both bullish keywords and meme emojis
        result = majority_agg.aggregate_single("Buying GME to the moon 🦍💎")
        assert result['has_conflict'] is True
        assert len(result['competing_labels']) > 1


class TestMinVotes:
    def test_below_min_votes_returns_none(self):
        agg = LabelAggregator(strategy="majority", config={'labeling': {'min_votes': 5}})
        result = agg.aggregate_single("Buying AAPL")
        # Only a couple of functions fire on this, well below 5
        if result['num_votes'] < 5:
            assert result['final_label'] is None


class TestStrategies:
    def test_majority_vote_works(self, majority_agg):
        result = majority_agg.aggregate_single("Buying calls on AAPL, very bullish 🚀")
        assert result['final_label'] is not None

    def test_weighted_vote_works(self, weighted_agg):
        result = weighted_agg.aggregate_single("Bought calls on AAPL")
        # Options directional has high weight, should be bullish
        assert result['final_label'] == BULLISH

    def test_confidence_weighted_works(self, confidence_agg):
        result = confidence_agg.aggregate_single("Bought calls on AAPL, very bullish, loading up")
        # With enough signal, should get a label
        assert result['confidence'] >= 0


class TestAggregateBatch:
    def test_adds_columns(self, majority_agg, sample_df):
        result_df = majority_agg.aggregate_batch(sample_df)
        expected_cols = [
            'programmatic_label', 'label_confidence',
            'label_coverage', 'label_conflict', 'vote_breakdown'
        ]
        for col in expected_cols:
            assert col in result_df.columns

    def test_preserves_original_columns(self, majority_agg, sample_df):
        result_df = majority_agg.aggregate_batch(sample_df)
        for col in sample_df.columns:
            assert col in result_df.columns

    def test_same_row_count(self, majority_agg, sample_df):
        result_df = majority_agg.aggregate_batch(sample_df)
        assert len(result_df) == len(sample_df)

    def test_coverage_is_boolean(self, majority_agg, sample_df):
        result_df = majority_agg.aggregate_batch(sample_df)
        assert result_df['label_coverage'].dtype == bool

    def test_confidence_is_numeric(self, majority_agg, sample_df):
        result_df = majority_agg.aggregate_batch(sample_df)
        assert result_df['label_confidence'].dtype == float
