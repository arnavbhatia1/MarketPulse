"""
Tests for EntityNormalizer.
"""

import pytest
from src.extraction.normalizer import EntityNormalizer


class TestNormalize:
    def test_ticker_symbol(self, normalizer):
        assert normalizer.normalize('AAPL') == 'apple'

    def test_cashtag(self, normalizer):
        assert normalizer.normalize('$AAPL') == 'apple'

    def test_hashtag(self, normalizer):
        assert normalizer.normalize('#AAPL') == 'apple'

    def test_company_name(self, normalizer):
        assert normalizer.normalize('Apple') == 'apple'

    def test_company_name_with_inc(self, normalizer):
        assert normalizer.normalize('Apple Inc') == 'apple'

    def test_case_insensitive(self, normalizer):
        assert normalizer.normalize('APPLE') == 'apple'

    def test_all_variations_match(self, normalizer):
        variations = ['$AAPL', 'AAPL', 'Apple', 'apple', 'apple inc', 'Apple Inc.', '#AAPL']
        canonicals = {normalizer.normalize(v) for v in variations}
        assert len(canonicals) == 1
        assert 'apple' in canonicals

    def test_tesla_variations(self, normalizer):
        variations = ['TSLA', '$TSLA', 'Tesla', 'tesla motors', 'Tesla Inc.']
        canonicals = {normalizer.normalize(v) for v in variations}
        assert len(canonicals) == 1
        assert 'tesla' in canonicals

    def test_unknown_entity_returned_lowercased(self, normalizer):
        assert normalizer.normalize('SomeRandomThing') == 'somerandomthing'

    def test_empty_string(self, normalizer):
        assert normalizer.normalize('') == ''

    def test_whitespace_handling(self, normalizer):
        assert normalizer.normalize('  Apple  ') == 'apple'


class TestNormalizeSet:
    def test_deduplicates(self, normalizer):
        entities = ['$AAPL', 'Apple', 'AAPL', 'apple inc']
        result = normalizer.normalize_set(entities)
        assert result == ['apple']

    def test_multiple_entities(self, normalizer):
        entities = ['$AAPL', '$TSLA', 'NVDA']
        result = normalizer.normalize_set(entities)
        assert 'apple' in result
        assert 'tesla' in result
        assert 'nvidia' in result
        assert len(result) == 3

    def test_empty_list(self, normalizer):
        assert normalizer.normalize_set([]) == []

    def test_sorted_output(self, normalizer):
        entities = ['TSLA', 'AAPL', 'NVDA']
        result = normalizer.normalize_set(entities)
        assert result == sorted(result)


class TestEntitiesMatch:
    def test_same_entity_different_forms(self, normalizer):
        assert normalizer.entities_match('$AAPL', 'Apple') is True

    def test_ticker_and_name(self, normalizer):
        assert normalizer.entities_match('TSLA', 'Tesla') is True

    def test_different_entities(self, normalizer):
        assert normalizer.entities_match('AAPL', 'TSLA') is False

    def test_cashtag_and_bare(self, normalizer):
        assert normalizer.entities_match('$NVDA', 'NVDA') is True

    def test_google_variations(self, normalizer):
        assert normalizer.entities_match('GOOG', 'Alphabet') is True
        assert normalizer.entities_match('$GOOGL', 'Google') is True
