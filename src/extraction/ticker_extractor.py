"""
Ticker Symbol and Company Name Extraction

Extracts company/ticker mentions from financial social media posts.
Returns canonical company names with extraction evidence.

Challenges in this domain:
- $AAPL is a ticker but $5 is a price
- Some tickers are common words: $F (Ford), $T (AT&T), $AI, $ALL
- Informal references: "Papa Musk" → Tesla
- Emoji references: 🍎 → Apple
- Abbreviated names: "GOOG" without $ prefix
- Product names: "iPhone" → Apple (do we extract this?)
- Multiple tickers in one post
- Tickers in hashtags: #NVDA
"""

import re
from src.utils.logger import get_logger
from src.extraction.ticker_universe import (
    TICKER_UNIVERSE,
    AMBIGUOUS_TICKERS,
    COMMON_WORD_NAMES,
)

logger = get_logger(__name__)


class TickerExtractor:
    """
    Rule-based extraction of ticker symbols and company names from text.

    Supports multiple extraction methods:
    - Cashtag patterns ($AAPL)
    - Bare ticker patterns (NVDA without $)
    - Company name aliases (Apple, Google)
    - Informal references (Elon → Tesla)
    - Emoji mappings (🍎 → Apple)
    """

    def __init__(self):
        # Source of truth: the bundled ticker universe (symbol -> company name).
        self.ticker_map = dict(TICKER_UNIVERSE)

        # Tickers that are also common words — require $ prefix to match.
        self.ambiguous_tickers = set(AMBIGUOUS_TICKERS)

        # Company-name aliases, auto-derived from the universe so any company in
        # it is recognized in headlines (the dominant signal in news text).
        # Skip names that are common English words (match those via $ cashtag) and
        # very short names (<=2 chars) to avoid false positives.
        self.company_aliases = {}
        for symbol, name in self.ticker_map.items():
            alias = name.lower()
            if len(alias) <= 2 or alias in COMMON_WORD_NAMES:
                continue
            self.company_aliases.setdefault(alias, name)

        # A few extra hand-curated aliases the universe names don't cover.
        self.company_aliases.update({
            'alphabet': 'Google', 'facebook': 'Meta', 'meta platforms': 'Meta',
            'advanced micro devices': 'AMD', 'taiwan semiconductor': 'TSMC',
            'lucid motors': 'Lucid', 'rivian automotive': 'Rivian',
            'jp morgan': 'JPMorgan', 'jpmorgan chase': 'JPMorgan',
        })

        self.informal_aliases = {
            'papa musk': 'Tesla', 'elon': 'Tesla',
            'zuck': 'Meta', 'zuckerberg': 'Meta',
            'tim cook': 'Apple', 'cook': 'Apple',
            'bezos': 'Amazon', 'jensen': 'NVIDIA',
            'satya': 'Microsoft', 'nadella': 'Microsoft',
            'lisa su': 'AMD',
        }

        self.emoji_map = {
            '\U0001f34e': 'Apple',  # 🍎
        }

    def extract(self, text):
        """
        Extract all company entities from text.
        Returns list of canonical company names (deduplicated).

        Args:
            text: str, raw post text

        Returns:
            list of str, canonical company names
        """
        evidence = self.extract_with_evidence(text)
        seen = set()
        result = []
        for e in evidence:
            if e['canonical'] not in seen:
                seen.add(e['canonical'])
                result.append(e['canonical'])
        return result

    def extract_with_evidence(self, text):
        """
        Extract entities with provenance tracking.

        Returns list of dicts:
        [
            {
                'canonical': 'Tesla',
                'surface_form': '$TSLA',
                'method': 'cashtag',
                'position': (15, 20)  # character span
            },
            {
                'canonical': 'Tesla',
                'surface_form': 'Elon',
                'method': 'informal_alias',
                'position': (45, 49)
            }
        ]
        """
        results = []
        results.extend(self._extract_cashtags(text))
        results.extend(self._extract_bare_tickers(text))
        results.extend(self._extract_company_names(text))
        results.extend(self._extract_informal(text))
        results.extend(self._extract_emoji(text))
        return results

    def _extract_cashtags(self, text):
        """
        Extract $TICKER patterns. Handle $5 vs $AAPL.
        Only match known tickers.
        """
        results = []
        for match in re.finditer(r'\$([A-Z]{1,5})\b', text):
            ticker = match.group(1)
            if ticker in self.ticker_map:
                results.append({
                    'canonical': self.ticker_map[ticker],
                    'surface_form': match.group(0),
                    'method': 'cashtag',
                    'position': (match.start(), match.end()),
                })
        return results

    def _extract_bare_tickers(self, text):
        """
        Extract ALL-CAPS ticker-like words without $ prefix.
        Only match tickers NOT in the ambiguous set.
        Must be standalone words (not part of a sentence in caps).
        """
        results = []
        for match in re.finditer(r'\b([A-Z]{2,5})\b', text):
            ticker = match.group(1)

            # Skip ambiguous tickers (need $ prefix)
            if ticker in self.ambiguous_tickers:
                continue

            # Skip if not in known ticker map
            if ticker not in self.ticker_map:
                continue

            # Check it's not part of an all-caps sentence
            # by verifying surrounding words aren't all caps too
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end]
            context_words = [w for w in context.split() if len(w) > 1 and w != ticker]

            if context_words:
                caps_ratio = sum(1 for w in context_words if w.isupper()) / len(context_words)
                if caps_ratio > 0.5:
                    continue  # Likely all-caps sentence, not a ticker

            results.append({
                'canonical': self.ticker_map[ticker],
                'surface_form': ticker,
                'method': 'bare_ticker',
                'position': (match.start(), match.end()),
            })
        return results

    def _extract_company_names(self, text):
        """Extract full company names via alias lookup."""
        results = []
        text_lower = text.lower()
        for alias, canonical in self.company_aliases.items():
            pattern = r'\b' + re.escape(alias) + r'\b'
            for match in re.finditer(pattern, text_lower):
                results.append({
                    'canonical': canonical,
                    'surface_form': text[match.start():match.end()],
                    'method': 'company_name',
                    'position': (match.start(), match.end()),
                })
        return results

    def _extract_informal(self, text):
        """Extract informal/people references."""
        results = []
        text_lower = text.lower()
        for alias, canonical in self.informal_aliases.items():
            pattern = r'\b' + re.escape(alias) + r'\b'
            for match in re.finditer(pattern, text_lower):
                results.append({
                    'canonical': canonical,
                    'surface_form': text[match.start():match.end()],
                    'method': 'informal_alias',
                    'position': (match.start(), match.end()),
                })
        return results

    def _extract_emoji(self, text):
        """Extract emoji-to-company mappings."""
        results = []
        for emoji, canonical in self.emoji_map.items():
            idx = text.find(emoji)
            if idx >= 0:
                results.append({
                    'canonical': canonical,
                    'surface_form': emoji,
                    'method': 'emoji',
                    'position': (idx, idx + len(emoji)),
                })
        return results
