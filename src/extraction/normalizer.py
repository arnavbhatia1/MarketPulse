"""
Entity Normalization

Maps all entity variations to a single canonical form
for fair evaluation comparison.

This module demonstrates a key Snorkel concept:
evaluation quality depends on normalization quality.
Metrics computed WITHOUT normalization are artificially low.
"""

import re
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EntityNormalizer:
    """
    Normalizes company entity mentions to canonical forms.

    Handles all known variations:
    - Ticker symbols: $AAPL, AAPL, #AAPL
    - Company names: Apple, Apple Inc, Apple Inc.
    - Different case and spacing
    - Aliases and informal references mapped to canonical form

    Key insight: Without normalization, entity extraction metrics are
    artificially low because multiple surface forms refer to the same
    company. Normalization reveals the true performance.
    """

    def __init__(self):
        self.canonical_map = {}
        self._build_map()

    def _build_map(self):
        """Build comprehensive normalization map. All variations → canonical form."""
        mappings = {
            'apple': [
                'apple', 'apple inc', 'apple inc.', 'apple incorporated',
                'aapl', '$aapl', '#aapl',
            ],
            'tesla': [
                'tesla', 'tesla inc', 'tesla motors', 'tsla',
                '$tsla', '#tsla', 'tesla inc.',
            ],
            'microsoft': [
                'microsoft', 'msft', '$msft', 'microsoft corp',
                'microsoft corporation',
            ],
            'google': [
                'google', 'alphabet', 'goog', 'googl',
                '$goog', '$googl', 'alphabet inc',
            ],
            'amazon': [
                'amazon', 'amzn', '$amzn', 'amazon.com',
                'amazon inc',
            ],
            'nvidia': [
                'nvidia', 'nvda', '$nvda', '#nvda',
                'nvidia corp', 'nvidia corporation',
            ],
            'meta': [
                'meta', 'meta platforms', 'facebook', 'fb',
                '$meta', 'meta inc',
            ],
            'gamestop': [
                'gamestop', 'gme', '$gme', '#gme',
                'game stop', 'gamestop inc',
            ],
            'amc': [
                'amc', 'amc entertainment', '$amc', '#amc',
                'amc entertainment inc',
            ],
            'palantir': [
                'palantir', 'pltr', '$pltr',
                'palantir technologies', 'palantir tech',
            ],
            's&p 500 etf': [
                'spy', '$spy', 's&p 500', 's&p500',
                'sp500', 's&p 500 etf', 'sp 500',
            ],
            'nasdaq etf': [
                'qqq', '$qqq', 'nasdaq', 'nasdaq etf',
                'nasdaq 100',
            ],
            'amd': [
                'amd', '$amd', '#amd', 'amd inc',
                'advanced micro devices',
            ],
            'intel': [
                'intel', 'intc', '$intc', 'intel corp',
                'intel corporation',
            ],
            'netflix': [
                'netflix', 'nflx', '$nflx', 'netflix inc',
            ],
            'disney': [
                'disney', 'dis', '$dis', 'walt disney',
                'disney company',
            ],
            'jpmorgan': [
                'jpmorgan', 'jpm', '$jpm', 'jp morgan',
                'jpmorgan chase', 'jpmc',
            ],
            'bank of america': [
                'bank of america', 'bac', '$bac', 'bofa',
                'boa', 'bank america',
            ],
            'ford': [
                'ford', 'f', '$f', 'ford motor',
                'ford motors',
            ],
            'at&t': [
                'at&t', 't', '$t', 'att', 'at&t inc',
                'american telephone telegraph',
            ],
            'coinbase': [
                'coinbase', 'coin', '$coin', 'coinbase inc',
            ],
            'robinhood': [
                'robinhood', 'hood', '$hood', 'robinhood markets',
            ],
            'paypal': [
                'paypal', 'pypl', '$pypl', 'paypal inc',
            ],
            'visa': [
                'visa', 'v', '$v', 'visa inc',
            ],
            'mastercard': [
                'mastercard', 'ma', '$ma', 'mastercard inc',
            ],
            'uber': [
                'uber', '$uber', 'uber tech',
            ],
            'airbnb': [
                'airbnb', 'abnb', '$abnb', 'airbnb inc',
            ],
            'salesforce': [
                'salesforce', 'crm', '$crm', 'salesforce inc',
            ],
            'oracle': [
                'oracle', 'orcl', '$orcl', 'oracle corp',
                'oracle corporation',
            ],
            'blackberry': [
                'blackberry', 'bb', '$bb', 'rim',
                'research in motion',
            ],
            'sofi': [
                'sofi', '$sofi', 'social finance',
            ],
            'block': [
                'block', 'sq', '$sq', 'square',
                'block inc', 'square inc',
            ],
            'snap': [
                'snap', '$snap', 'snapchat', 'snap inc',
            ],
            'alibaba': [
                'alibaba', 'baba', '$baba', 'alibaba group',
                'alibaba group holding',
            ],
            'tsmc': [
                'tsmc', 'tsm', '$tsm', 'taiwan semiconductor',
                'taiwan semi',
            ],
            'rivian': [
                'rivian', 'rivn', '$rivn', 'rivian automotive',
            ],
            'lucid': [
                'lucid', 'lcid', '$lcid', 'lucid motors',
            ],
            'nio': [
                'nio', '$nio', 'nio inc',
            ],
            'contextlogic': [
                'contextlogic', 'wish', '$wish', 'wish inc',
            ],
            'ge': [
                'ge', '$ge', 'general electric',
                'general electric co',
            ],
            'lyft': [
                'lyft', '$lyft', 'lyft inc',
            ],
        }

        for canonical, variations in mappings.items():
            for variation in variations:
                self.canonical_map[variation.lower()] = canonical

    def normalize(self, entity):
        """
        Normalize single entity to canonical form.

        Args:
            entity: str, raw entity text (may include $ or #, different case)

        Returns:
            str, canonical form (lowercase company name or alias)
        """
        if not entity:
            return entity

        entity_lower = entity.strip().lower()
        # Strip $ and # prefixes
        entity_lower = re.sub(r'^[\$#]', '', entity_lower)
        # Normalize whitespace
        entity_lower = re.sub(r'\s+', ' ', entity_lower).strip()

        return self.canonical_map.get(entity_lower, entity_lower)

    def normalize_set(self, entities):
        """
        Normalize list of entities, deduplicate after normalization.

        Args:
            entities: list of str, raw entity texts

        Returns:
            list of str, sorted canonical forms (deduplicated)
        """
        if not entities:
            return []

        normalized = set()
        for entity in entities:
            canonical = self.normalize(entity)
            if canonical:
                normalized.add(canonical)
        return sorted(normalized)

    def entities_match(self, entity_a, entity_b):
        """
        Check if two entity strings refer to the same company.

        Args:
            entity_a: str, first entity
            entity_b: str, second entity

        Returns:
            bool, True if both normalize to the same canonical form
        """
        return self.normalize(entity_a) == self.normalize(entity_b)
