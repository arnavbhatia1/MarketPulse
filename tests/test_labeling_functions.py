"""
Tests for individual labeling functions.

Each of the 16 text-only labeling functions is tested for:
- Known positive class match
- ABSTAIN on irrelevant text
- Edge cases (empty, emoji-only, sarcasm)
"""

import pytest
from src.labeling.functions import (
    BULLISH, BEARISH, NEUTRAL, MEME, ABSTAIN,
    lf_keyword_bullish, lf_keyword_bearish, lf_keyword_neutral, lf_keyword_meme,
    lf_emoji_bullish, lf_emoji_bearish, lf_emoji_meme,
    lf_question_structure, lf_short_post, lf_all_caps_ratio,
    lf_options_directional, lf_price_target_mention, lf_loss_reporting,
    lf_news_language, lf_sarcasm_indicators, lf_self_deprecating,
    lf_stocktwits_user_sentiment, lf_reddit_flair,
    LABELING_FUNCTIONS, METADATA_FUNCTIONS,
)


# ── Keyword functions ──────────────────────────────────────────

class TestKeywordBullish:
    def test_buying_keyword(self):
        assert lf_keyword_bullish("Just buying AAPL here") == BULLISH

    def test_bullish_keyword(self):
        assert lf_keyword_bullish("Very bullish on NVDA long term") == BULLISH

    def test_no_match(self):
        assert lf_keyword_bullish("The weather is nice today") == ABSTAIN

    def test_empty_string(self):
        assert lf_keyword_bullish("") == ABSTAIN


class TestKeywordBearish:
    def test_selling_keyword(self):
        assert lf_keyword_bearish("Selling everything before crash") == BEARISH

    def test_puts_keyword(self):
        assert lf_keyword_bearish("Loaded puts on SPY") == BEARISH

    def test_no_match(self):
        assert lf_keyword_bearish("The weather is nice today") == ABSTAIN

    def test_empty_string(self):
        assert lf_keyword_bearish("") == ABSTAIN


class TestKeywordNeutral:
    def test_question_pattern(self):
        assert lf_keyword_neutral("What do you think about TSLA?") == NEUTRAL

    def test_news_report(self):
        assert lf_keyword_neutral("AAPL announces new product line") == NEUTRAL

    def test_no_match(self):
        assert lf_keyword_neutral("TSLA is going up forever") == ABSTAIN

    def test_empty_string(self):
        assert lf_keyword_neutral("") == ABSTAIN


class TestKeywordMeme:
    def test_apes_keyword(self):
        assert lf_keyword_meme("Apes together strong") == MEME

    def test_tendies(self):
        assert lf_keyword_meme("Give me those tendies") == MEME

    def test_diamond_hands(self):
        assert lf_keyword_meme("Diamond hands only 💎") == MEME

    def test_no_match(self):
        assert lf_keyword_meme("AAPL is a solid investment") == ABSTAIN

    def test_empty_string(self):
        assert lf_keyword_meme("") == ABSTAIN


# ── Emoji functions ────────────────────────────────────────────

class TestEmojiBullish:
    def test_rocket(self):
        assert lf_emoji_bullish("$TSLA 🚀🚀🚀") == BULLISH

    def test_chart_up(self):
        assert lf_emoji_bullish("Looking good 📈") == BULLISH

    def test_no_emoji(self):
        assert lf_emoji_bullish("No emojis here") == ABSTAIN


class TestEmojiBearish:
    def test_chart_down(self):
        assert lf_emoji_bearish("Oh no 📉📉") == BEARISH

    def test_bear(self):
        assert lf_emoji_bearish("Bear market 🐻") == BEARISH

    def test_no_emoji(self):
        assert lf_emoji_bearish("No emojis here") == ABSTAIN


class TestEmojiMeme:
    def test_diamond(self):
        assert lf_emoji_meme("💎🙌 forever") == MEME

    def test_ape(self):
        assert lf_emoji_meme("🦍🦍🦍") == MEME

    def test_clown(self):
        assert lf_emoji_meme("Great idea 🤡") == MEME

    def test_no_emoji(self):
        assert lf_emoji_meme("No emojis here") == ABSTAIN


# ── Structural functions ───────────────────────────────────────

class TestQuestionStructure:
    def test_ends_with_question_mark(self):
        assert lf_question_structure("Is TSLA overvalued?") == NEUTRAL

    def test_multiple_question_marks(self):
        assert lf_question_structure("Why? How? When??") == NEUTRAL

    def test_no_question(self):
        assert lf_question_structure("TSLA is going up.") == ABSTAIN


class TestShortPost:
    def test_very_short(self):
        assert lf_short_post("GUH") == MEME

    def test_short_emoji(self):
        assert lf_short_post("🚀🚀🚀") == MEME

    def test_longer_post(self):
        assert lf_short_post("This is a longer post about stocks and such") == ABSTAIN


class TestAllCapsRatio:
    def test_bullish_caps(self):
        assert lf_all_caps_ratio("BUY BUY BUY THE MOON IS NEAR") == BULLISH

    def test_bearish_caps(self):
        assert lf_all_caps_ratio("SELL SELL SELL CRASH IS HERE") == BEARISH

    def test_low_caps_ratio(self):
        assert lf_all_caps_ratio("just a normal post about stocks") == ABSTAIN

    def test_too_short(self):
        assert lf_all_caps_ratio("BUY NOW") == ABSTAIN


# ── Financial pattern functions ────────────────────────────────

class TestOptionsDirectional:
    def test_bought_calls(self):
        assert lf_options_directional("Bought calls on AAPL today") == BULLISH

    def test_bought_puts(self):
        assert lf_options_directional("Buying puts on SPY") == BEARISH

    def test_no_options(self):
        assert lf_options_directional("Just bought some shares") == ABSTAIN


class TestPriceTargetMention:
    def test_pt_pattern(self):
        assert lf_price_target_mention("PT $250 on NVDA") == BULLISH

    def test_price_target_words(self):
        assert lf_price_target_mention("Price target $300 for TSLA") == BULLISH

    def test_no_target(self):
        assert lf_price_target_mention("I like this stock") == ABSTAIN


class TestLossReporting:
    def test_down_percent(self):
        assert lf_loss_reporting("Down 80% on my calls") == MEME

    def test_lost_dollars(self):
        assert lf_loss_reporting("Lost $5,000 this week") == MEME

    def test_no_loss(self):
        assert lf_loss_reporting("Made some money today") == ABSTAIN


class TestNewsLanguage:
    def test_news_patterns(self):
        assert lf_news_language("Company announces acquisition, analysts upgrade") == NEUTRAL

    def test_single_news_word(self):
        # Requires >= 2 matches
        assert lf_news_language("Company announces something") == ABSTAIN

    def test_no_news(self):
        assert lf_news_language("YOLO into GME") == ABSTAIN


# ── Sarcasm/irony functions ────────────────────────────────────

class TestSarcasmIndicators:
    def test_sarcasm_with_bullish(self):
        assert lf_sarcasm_indicators("Oh yeah WISH is definitely going to moon") == BEARISH

    def test_clown_with_bullish(self):
        assert lf_sarcasm_indicators("Going to the moon 🤡") == BEARISH

    def test_no_sarcasm(self):
        assert lf_sarcasm_indicators("I think AAPL is a good buy") == ABSTAIN


class TestSelfDeprecating:
    def test_wifes_boyfriend(self):
        assert lf_self_deprecating("My wife's boyfriend picks better stocks") == MEME

    def test_smooth_brain(self):
        assert lf_self_deprecating("Smooth brain play here") == MEME

    def test_no_self_deprecation(self):
        assert lf_self_deprecating("Smart investment decision") == ABSTAIN


# ── Metadata functions ─────────────────────────────────────────

class TestStocktwitsUserSentiment:
    def test_bullish_tag(self):
        assert lf_stocktwits_user_sentiment("text", metadata={'user_sentiment': 'Bullish'}) == BULLISH

    def test_bearish_tag(self):
        assert lf_stocktwits_user_sentiment("text", metadata={'user_sentiment': 'Bearish'}) == BEARISH

    def test_no_metadata(self):
        assert lf_stocktwits_user_sentiment("text") == ABSTAIN

    def test_no_sentiment(self):
        assert lf_stocktwits_user_sentiment("text", metadata={}) == ABSTAIN


class TestRedditFlair:
    def test_yolo_flair(self):
        assert lf_reddit_flair("text", metadata={'flair': 'YOLO'}) == MEME

    def test_dd_flair(self):
        assert lf_reddit_flair("text", metadata={'flair': 'DD'}) == NEUTRAL

    def test_no_flair(self):
        assert lf_reddit_flair("text", metadata={}) == ABSTAIN

    def test_no_metadata(self):
        assert lf_reddit_flair("text") == ABSTAIN


# ── Registry sanity ────────────────────────────────────────────

class TestRegistry:
    def test_labeling_functions_count(self):
        assert len(LABELING_FUNCTIONS) == 16

    def test_metadata_functions_count(self):
        assert len(METADATA_FUNCTIONS) == 2

    def test_all_return_valid_values(self, sample_posts):
        valid = {BULLISH, BEARISH, NEUTRAL, MEME, ABSTAIN}
        for func in LABELING_FUNCTIONS:
            for post in sample_posts:
                result = func(post)
                assert result in valid, f"{func.__name__} returned {result}"
