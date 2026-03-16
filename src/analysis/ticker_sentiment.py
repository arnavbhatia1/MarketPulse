"""
Per-Ticker Sentiment Aggregation

Takes labeled + extracted DataFrame and produces per-ticker
sentiment summaries for the dashboard.
"""

import pandas as pd
from collections import Counter
from src.extraction.ticker_extractor import TickerExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

SENTIMENT_COLORS = {
    'bullish': '#00C853',
    'bearish': '#FF1744',
    'neutral': '#78909C',
    'meme': '#FFD600',
}


class TickerSentimentAnalyzer:

    def __init__(self):
        self.extractor = TickerExtractor()
        self._ticker_to_symbol = {}
        for symbol, company in self.extractor.ticker_map.items():
            if company not in self._ticker_to_symbol:
                self._ticker_to_symbol[company] = symbol

    def analyze(self, df):
        """
        Analyze per-ticker sentiment from labeled DataFrame.

        Returns dict mapping company name to ticker summary dict.
        Fields: news_sentiment, sentiment_by_day, top_posts.
        """
        df = df.copy()

        if 'tickers' not in df.columns:
            df['tickers'] = df['text'].apply(self.extractor.extract)

        labeled = df[df['programmatic_label'].notna()].copy()

        # Normalize timestamp to date string for grouping
        labeled['_date'] = pd.to_datetime(
            labeled['timestamp'], errors='coerce'
        ).dt.strftime('%Y-%m-%d').fillna('unknown')

        ticker_data = {}

        for _, row in labeled.iterrows():
            tickers = row['tickers']
            if not tickers or not isinstance(tickers, list):
                continue

            for company in tickers:
                if company not in ticker_data:
                    ticker_data[company] = {
                        'company': company,
                        'symbol': self._ticker_to_symbol.get(company, ''),
                        'all_posts': [],
                    }

                ticker_data[company]['all_posts'].append({
                    'post_id': str(row.get('post_id', '')),
                    'text': str(row['text']),
                    'sentiment': row['programmatic_label'],
                    'confidence': float(row.get('label_confidence', 0)),
                    'source': str(row.get('source', 'unknown')),
                    'timestamp': str(row.get('timestamp', '')),
                    'date': str(row['_date']),
                    'author': str(row.get('author', 'unknown')),
                    'url': str(row.get('url', '')),
                })

        results = {}
        for company, data in ticker_data.items():
            posts = data['all_posts']
            total = len(posts)
            if total == 0:
                continue

            all_sentiments = [p['sentiment'] for p in posts]
            sentiment_counts = Counter(all_sentiments)
            dominant = sentiment_counts.most_common(1)[0][0]

            # Per-source dominant sentiment
            source_sentiments = {'news': None}
            news_posts = [p for p in posts if p['source'] == 'news']
            if news_posts:
                news_counts = Counter(p['sentiment'] for p in news_posts)
                source_sentiments['news'] = news_counts.most_common(1)[0][0]

            # Dominant sentiment per day
            day_groups = {}
            for p in posts:
                day_groups.setdefault(p['date'], []).append(p['sentiment'])
            sentiment_by_day = {
                day: Counter(sentiments).most_common(1)[0][0]
                for day, sentiments in day_groups.items()
            }

            # Top posts sorted by confidence desc
            top_posts = {'news': []}
            top_posts['news'] = sorted(
                [p for p in posts if p['source'] == 'news'],
                key=lambda p: p['confidence'], reverse=True
            )[:5]

            confidences = [p['confidence'] for p in posts]
            results[company] = {
                'company': company,
                'symbol': data['symbol'],
                'mention_count': total,
                'sentiment': dict(sentiment_counts),
                'dominant_sentiment': dominant,
                'dominant_color': SENTIMENT_COLORS.get(dominant, '#78909C'),
                'bullish_ratio': sentiment_counts.get('bullish', 0) / total,
                'bearish_ratio': sentiment_counts.get('bearish', 0) / total,
                'avg_confidence': sum(confidences) / len(confidences),
                'reddit_sentiment': None,
                'news_sentiment': source_sentiments['news'],
                'stocktwits_sentiment': None,
                'sentiment_by_day': sentiment_by_day,
                'top_posts': top_posts,
                'posts': sorted(posts, key=lambda p: p['confidence'], reverse=True),
            }

        results = dict(sorted(
            results.items(), key=lambda x: x[1]['mention_count'], reverse=True
        ))
        logger.info(f"Ticker analysis: {len(results)} tickers, {len(labeled)} labeled posts")
        return results

    def get_market_summary(self, ticker_results):
        """
        Aggregate market-level summary from ticker results.

        Returns dict with overall sentiment distribution across tickers.
        """
        sentiment_counts = Counter()
        total_mentions = 0
        total_tickers = len(ticker_results)

        for company, data in ticker_results.items():
            sentiment_counts[data['dominant_sentiment']] += 1
            total_mentions += data['mention_count']

        bullish_count = sentiment_counts.get('bullish', 0)
        bearish_count = sentiment_counts.get('bearish', 0)
        bullish_pct = bullish_count / total_tickers if total_tickers else 0
        bearish_pct = bearish_count / total_tickers if total_tickers else 0

        if bullish_count > bearish_count:
            overall_sentiment = 'bullish'
        elif bearish_count > bullish_count:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'mixed'

        return {
            'total_tickers': total_tickers,
            'total_mentions': total_mentions,
            'ticker_sentiment_distribution': dict(sentiment_counts),
            'overall_sentiment': overall_sentiment,
            'bullish_pct': bullish_pct,
            'bearish_pct': bearish_pct,
            'top_bullish': [
                t for t in ticker_results.values()
                if t['dominant_sentiment'] == 'bullish'
            ][:5],
            'top_bearish': [
                t for t in ticker_results.values()
                if t['dominant_sentiment'] == 'bearish'
            ][:5],
        }
