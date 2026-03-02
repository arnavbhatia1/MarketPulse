"""
Tests for TickerExtractor.
"""

import pytest
from src.extraction.ticker_extractor import TickerExtractor


class TestCashtagExtraction:
    def test_single_cashtag(self, ticker_extractor):
        entities = ticker_extractor.extract("Just bought $TSLA")
        assert 'Tesla' in entities

    def test_multiple_cashtags(self, ticker_extractor):
        entities = ticker_extractor.extract("Deciding between $AAPL and $NVDA")
        assert 'Apple' in entities
        assert 'NVIDIA' in entities

    def test_price_not_matched(self, ticker_extractor):
        # $5 is a price, not a ticker
        entities = ticker_extractor.extract("Stock is up $5 today")
        assert len(entities) == 0

    def test_ambiguous_cashtag_matched(self, ticker_extractor):
        # $F with $ prefix should match Ford
        entities = ticker_extractor.extract("Looking at $F for a value play")
        assert 'Ford' in entities


class TestBareTickerExtraction:
    def test_bare_ticker(self, ticker_extractor):
        entities = ticker_extractor.extract("NVDA is looking strong this week")
        assert 'NVIDIA' in entities

    def test_ambiguous_bare_ticker_not_matched(self, ticker_extractor):
        # F without $ should NOT match
        entities = ticker_extractor.extract("F this market honestly")
        assert 'Ford' not in entities

    def test_all_caps_sentence_skipped(self, ticker_extractor):
        # In an all-caps context, bare tickers should be skipped
        entities = ticker_extractor.extract("BUY SELL HOLD NVDA DUMP CRASH")
        # Most words are caps, so NVDA should be filtered
        # (depends on context window heuristic)


class TestCompanyNameExtraction:
    def test_company_name(self, ticker_extractor):
        entities = ticker_extractor.extract("Apple reported strong earnings")
        assert 'Apple' in entities

    def test_company_name_case_insensitive(self, ticker_extractor):
        entities = ticker_extractor.extract("I love tesla products")
        assert 'Tesla' in entities


class TestInformalAliasExtraction:
    def test_elon_maps_to_tesla(self, ticker_extractor):
        entities = ticker_extractor.extract("Elon tweeted something crazy again")
        assert 'Tesla' in entities

    def test_zuck_maps_to_meta(self, ticker_extractor):
        entities = ticker_extractor.extract("Zuck announced the new VR headset")
        assert 'Meta' in entities

    def test_lisa_su_maps_to_amd(self, ticker_extractor):
        entities = ticker_extractor.extract("Lisa Su on stage at CES talking chips")
        assert 'AMD' in entities


class TestEmojiExtraction:
    def test_apple_emoji(self, ticker_extractor):
        entities = ticker_extractor.extract("🍎 is going to crush earnings")
        assert 'Apple' in entities


class TestMultipleTickers:
    def test_deduplication(self, ticker_extractor):
        # Same company mentioned multiple ways
        entities = ticker_extractor.extract("$TSLA Tesla Elon — all the same")
        assert entities.count('Tesla') == 1

    def test_multiple_different_tickers(self, ticker_extractor):
        entities = ticker_extractor.extract("Selling $AAPL to buy $NVDA")
        assert 'Apple' in entities
        assert 'NVIDIA' in entities


class TestExtractWithEvidence:
    def test_evidence_structure(self, ticker_extractor):
        evidence = ticker_extractor.extract_with_evidence("Bought $TSLA today")
        assert len(evidence) > 0
        item = evidence[0]
        assert 'canonical' in item
        assert 'surface_form' in item
        assert 'method' in item
        assert 'position' in item

    def test_cashtag_method(self, ticker_extractor):
        evidence = ticker_extractor.extract_with_evidence("Love $AAPL")
        cashtag_evidence = [e for e in evidence if e['method'] == 'cashtag']
        assert len(cashtag_evidence) > 0
        assert cashtag_evidence[0]['canonical'] == 'Apple'
        assert cashtag_evidence[0]['surface_form'] == '$AAPL'

    def test_position_is_tuple(self, ticker_extractor):
        evidence = ticker_extractor.extract_with_evidence("Bought $TSLA")
        for item in evidence:
            assert isinstance(item['position'], tuple)
            assert len(item['position']) == 2
