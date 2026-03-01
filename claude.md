# CLAUDE.md — MarketPulse: Data-Centric Sentiment Intelligence for Financial Social Media

## PROJECT OVERVIEW

MarketPulse is a data-centric AI pipeline that classifies market sentiment from financial social media posts (WallStreetBets-style), extracts ticker symbol entities, and demonstrates that **labeling strategy and data quality drive model performance more than model complexity.**

This project showcases the Snorkel AI philosophy: programmatic labeling → label quality assessment → classical ML vs LLM comparison → evaluation framework design, all visualized through an interactive Streamlit dashboard.

### Core Thesis
> "A logistic regression trained on high-quality, programmatically labeled data outperforms an LLM on noisy, inconsistently labeled data. The data is the model."

### What This Project Demonstrates
1. **Programmatic Labeling (Weak Supervision):** Writing labeling functions that encode domain heuristics, combining noisy label sources into probabilistic training labels
2. **Label Quality Assessment:** Measuring inter-function agreement, coverage, conflict rates, and label confidence before training any model
3. **Classical ML Pipeline:** TF-IDF + Logistic Regression trained on programmatically labeled data
4. **LLM-Based Classification:** Prompt-engineered GPT-based classification for comparison
5. **Entity Extraction:** Ticker symbol and company name extraction with normalization
6. **Evaluation Framework:** Comprehensive metrics with entity-level P/R/F1, normalization logic, and error analysis
7. **Interactive Dashboard:** Streamlit app visualizing the entire pipeline from data exploration through model comparison

---

## ARCHITECTURE
marketpulse/
│
├── CLAUDE.md                          # This file — project spec
├── README.md                          # Project documentation for GitHub
├── requirements.txt                   # Dependencies
├── .env.example                       # Template for API keys
│
├── data/
│   ├── raw/
│   │   └── wsb_posts_raw.csv          # Raw social media posts dataset
│   ├── labeled/
│   │   └── wsb_posts_labeled.csv      # Posts with programmatic labels
│   ├── gold/
│   │   └── gold_standard.csv          # 100 hand-labeled posts for evaluation
│   └── README.md                      # Data documentation and schema
│
├── src/
│   ├── init.py
│   │
│   ├── data/
│   │   ├── init.py
│   │   ├── dataset.py                 # Dataset loading, cleaning, splitting
│   │   └── explorer.py                # Data exploration and profiling
│   │
│   ├── labeling/
│   │   ├── init.py
│   │   ├── labeling_functions.py      # All programmatic labeling functions
│   │   ├── label_aggregator.py        # Majority vote / weighted combination
│   │   └── label_quality.py           # Coverage, conflict, agreement metrics
│   │
│   ├── models/
│   │   ├── init.py
│   │   ├── classical_ml.py            # TF-IDF + LogReg pipeline
│   │   └── llm_classifier.py          # OpenAI-based classification
│   │
│   ├── extraction/
│   │   ├── init.py
│   │   ├── ticker_extractor.py        # Rule-based ticker extraction
│   │   ├── llm_extractor.py           # LLM-based entity extraction
│   │   └── normalizer.py              # Entity normalization logic
│   │
│   ├── evaluation/
│   │   ├── init.py
│   │   ├── classification_eval.py     # Classification P/R/F1 + error analysis
│   │   ├── extraction_eval.py         # Entity-level P/R/F1 with normalization
│   │   └── comparison.py              # Head-to-head model comparison
│   │
│   └── utils/
│       ├── init.py
│       └── config.py                  # Configuration and constants
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA notebook
│   ├── 02_labeling_pipeline.ipynb     # Programmatic labeling walkthrough
│   ├── 03_model_training.ipynb        # Classical ML training
│   ├── 04_llm_comparison.ipynb        # LLM vs classical comparison
│   └── 05_full_pipeline.ipynb         # End-to-end demo notebook
│
├── app/
│   ├── streamlit_app.py               # Main Streamlit dashboard
│   ├── pages/
│   │   ├── 1_data_exploration.py      # Data profiling page
│   │   ├── 2_labeling_studio.py       # Labeling function visualization
│   │   ├── 3_label_quality.py         # Label quality assessment page
│   │   ├── 4_model_comparison.py      # Classical vs LLM comparison
│   │   ├── 5_entity_extraction.py     # Ticker extraction + eval
│   │   └── 6_error_analysis.py        # Deep error analysis page
│   └── components/
│       ├── charts.py                  # Reusable Plotly chart components
│       └── metrics.py                 # Metric display components
│
└── tests/
├── test_labeling_functions.py
├── test_evaluation.py
├── test_normalizer.py
└── test_extraction.py
Copy
---

## DATA SPECIFICATION

### Raw Dataset Schema: `wsb_posts_raw.csv`

Generate a synthetic dataset of 500 financial social media posts. Each post should feel authentic to WallStreetBets / FinTwit culture. The dataset must include intentional messiness and edge cases.

```python
# Schema
{
    'post_id': int,                    # Unique identifier
    'text': str,                       # Raw post text
    'timestamp': str,                  # ISO format datetime
    'source': str,                     # 'reddit' | 'twitter' | 'stocktwits'
    'upvotes': int,                    # Engagement metric
    'sentiment_label': str | None,     # Ground truth (only for gold set)
    'ticker_labels': list | None       # Ground truth entities (only for gold set)
}
Sentiment Categories (4 classes)
CopyBULLISH:  Positive market sentiment. Expecting price to go up.
          Buying, holding, optimistic about a stock/market.
          Examples: "TSLA to the moon 🚀", "Loading up on NVDA calls"

BEARISH:  Negative market sentiment. Expecting price to go down.
          Selling, shorting, pessimistic about a stock/market.
          Examples: "This bubble is about to pop", "Puts on SPY printing"

NEUTRAL:  Factual, informational, asking questions, sharing news 
          without strong directional sentiment.
          Examples: "AAPL reports earnings Thursday", "What's everyone 
          thinking about the Fed meeting?"

MEME:     Hype-driven, joke-heavy, ironic, or self-deprecating content 
          that doesn't carry reliable directional signal. 
          Examples: "apes together strong 🦍", "wife's boyfriend 
          bought me WISH shares", "I just like the stock"
Why 4 Categories (Not 3)
The MEME category is critical. WallStreetBets content frequently contains extreme language ("YOLO everything into GME") that SOUNDS bullish but is actually meme culture. Classifying memes as bullish would corrupt any downstream signal. This category tests whether the system can distinguish genuine conviction from internet culture.
Data Generation Requirements
Generate posts across these distributions:

Bullish: 30% (150 posts)
Bearish: 20% (100 posts)
Neutral: 25% (125 posts)
Meme: 25% (125 posts)

Each post MUST include realistic messiness:
CopyTEXT PATTERNS TO INCLUDE:
- Ticker symbols: $TSLA, $AAPL, $GME, $AMC, $NVDA, $SPY, $QQQ, $PLTR, $BB, $SOFI
- Cashtags and hashtags: $TSLA, #YOLO, #diamondhands
- Reddit/Twitter slang: DD, YOLO, tendies, apes, diamond hands, paper hands, 
  wife's boyfriend, to the moon, bagholder, FUD, HODL, stonks
- Emojis: 🚀🌙💎🙌🦍📈📉🐻🐂💀🤡
- ALL CAPS for emphasis: "TSLA IS GOING TO MARS"
- Mixed case and typos: "nvda gunna moon tmrw"
- Sarcasm: "great job buying at the top genius 🤡"
- Numbers and prices: "bought 100 shares at $180", "PT $500"
- Options language: "calls", "puts", "IV crush", "theta gang", "FDs"
- Abbreviated language: "ngl", "imo", "tbh", "lmao", "smh"
- Multiple tickers in one post: "rotating from AAPL to MSFT and GOOG"
- Implicit sentiment: "just sold everything" (bearish but no negative words)
- Sarcastic bullish: "sure this time the squeeze will definitely happen 🙄"
Edge Cases to Include (At Least 50 Posts)
CopyAMBIGUOUS POSTS (must exist in dataset):

1. Sarcastic bullish → actually bearish:
   "oh yeah WISH is definitely going to $100 🤡🤡🤡"

2. Meme that contains real sentiment:
   "apes not selling GME. diamond hands forever 💎🙌 (also I genuinely 
   believe in the company's transformation)"

3. Bearish language, bullish position:
   "everyone says PLTR is going to crash but I just bought 500 more shares"

4. Question with embedded sentiment:
   "is anyone else worried about NVDA's valuation at these levels?"

5. News mixed with opinion:
   "AAPL beat earnings by 15%. This stock is unstoppable 🚀"

6. Post about losses (could be meme or bearish):
   "down 80% on my GME calls. at least I have my cardboard box 📦"

7. Contradictory sentiment:
   "I think TSLA is overvalued but I keep buying every dip lol"

8. Pure meme with ticker mentions:
   "what if we just all bought SPY puts at the same time haha jk... unless? 👀"

9. Neutral analysis that sounds bearish:
   "NVDA P/E ratio is 65x. Historical average for semis is 20x. 
   Draw your own conclusions."

10. Extremely short ambiguous posts:
    "TSLA 🚀", "GUH", "RIP my portfolio", "calls or puts?", "moon soon"
Gold Standard Dataset: gold_standard.csv
Hand-label 100 posts (subset of the 500) with:

sentiment_gold: The correct sentiment label
tickers_gold: List of all ticker symbols/companies mentioned
ambiguity_score: 1 (clear) to 5 (very ambiguous) — how hard is this to label
labeling_notes: Brief explanation of why this label was chosen

This gold set is used to evaluate BOTH the programmatic labels AND the model predictions.

PROGRAMMATIC LABELING PIPELINE
Philosophy
Instead of manually labeling 500 posts, we write labeling functions — heuristic rules that each vote on what the label should be. Each function is noisy and incomplete on its own, but combined they produce high-quality probabilistic labels. This is the Snorkel approach.
File: src/labeling/labeling_functions.py
Implement these labeling functions. Each function takes a post text and returns one of: BULLISH, BEARISH, NEUTRAL, MEME, or ABSTAIN (when the function can't decide).
pythonCopy# Constants
BULLISH = "bullish"
BEARISH = "bearish"  
NEUTRAL = "neutral"
MEME = "meme"
ABSTAIN = -1  # Function doesn't vote on this post

# ============================================
# KEYWORD-BASED LABELING FUNCTIONS
# ============================================

def lf_bullish_keywords(text):
    """
    Vote BULLISH if post contains buying/positive keywords.
    Expected: High precision for obvious cases, low recall.
    """
    bullish_words = ['moon', 'rocket', 'buy', 'calls', 'long', 
                     'bullish', 'loading up', 'going up', 'breakout',
                     'undervalued', 'buy the dip', 'price target',
                     'upside', 'accumulating']
    text_lower = text.lower()
    if any(word in text_lower for word in bullish_words):
        return BULLISH
    return ABSTAIN


def lf_bearish_keywords(text):
    """
    Vote BEARISH if post contains selling/negative keywords.
    Expected: High precision for obvious cases, misses subtle bears.
    """
    bearish_words = ['puts', 'short', 'sell', 'crash', 'dump', 
                     'overvalued', 'bubble', 'bearish', 'going down',
                     'bagholder', 'dead cat', 'rug pull', 'top is in',
                     'exit', 'taking profits']
    text_lower = text.lower()
    if any(word in text_lower for word in bearish_words):
        return BEARISH
    return ABSTAIN


def lf_neutral_keywords(text):
    """
    Vote NEUTRAL if post is asking questions or sharing facts.
    Expected: Catches questions well, may miss factual statements.
    """
    text_lower = text.lower()
    neutral_patterns = [
        r'\?$', r'\?["\s]', 'what do you think', 'thoughts on',
        'anyone know', 'when is', 'how does', 'eli5',
        'earnings report', 'announces', 'reports'
    ]
    if any(re.search(p, text_lower) for p in neutral_patterns):
        return NEUTRAL
    return ABSTAIN


def lf_meme_keywords(text):
    """
    Vote MEME if post contains WSB meme culture language.
    Expected: Good at catching obvious memes, may over-trigger on 
    posts that use meme language but have real sentiment.
    """
    meme_words = ['apes', 'tendies', 'wife\'s boyfriend', 'wendy\'s',
                  'diamond hands', 'paper hands', 'yolo', 'guh',
                  'to the moon', 'stonks', 'smooth brain', 'retard',
                  'degen', 'casino', 'gambling', 'loss porn',
                  'gain porn', 'i just like the stock']
    text_lower = text.lower()
    if any(word in text_lower for word in meme_words):
        return MEME
    return ABSTAIN


# ============================================
# EMOJI-BASED LABELING FUNCTIONS
# ============================================

def lf_bullish_emoji(text):
    """
    Vote BULLISH if post contains rocket/moon/chart-up emojis.
    Expected: Tricky — 🚀 is used both genuinely and ironically.
    This function will have lower precision because of meme usage.
    """
    bullish_emojis = ['🚀', '🌙', '📈', '🐂', '💰', '🤑', '⬆️']
    if any(emoji in text for emoji in bullish_emojis):
        return BULLISH
    return ABSTAIN


def lf_bearish_emoji(text):
    """
    Vote BEARISH if post contains bear/down/skull emojis.
    """
    bearish_emojis = ['📉', '🐻', '💀', '🔻', '⬇️', '😱']
    if any(emoji in text for emoji in bearish_emojis):
        return BEARISH
    return ABSTAIN


def lf_meme_emoji(text):
    """
    Vote MEME if post contains diamond hands or ape emojis.
    Expected: High precision for meme content.
    """
    meme_emojis = ['💎', '🙌', '🦍', '🤡', '🎰', '🍗']
    if any(emoji in text for emoji in meme_emojis):
        return MEME
    return ABSTAIN


# ============================================
# STRUCTURAL LABELING FUNCTIONS
# ============================================

def lf_question_mark(text):
    """
    Vote NEUTRAL if post ends with or contains multiple question marks.
    Expected: Simple but effective for identifying questions.
    """
    if text.strip().endswith('?') or text.count('?') >= 2:
        return NEUTRAL
    return ABSTAIN


def lf_all_caps_intensity(text):
    """
    Vote based on caps ratio — high caps often indicates strong 
    emotion (bullish hype or bearish panic).
    Does NOT vote directly — abstains. Used as a signal amplifier.
    """
    words = text.split()
    if len(words) == 0:
        return ABSTAIN
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)
    if caps_ratio > 0.5:
        # High caps — could be bullish hype or bearish panic
        # Check for bearish keywords in caps context
        text_lower = text.lower()
        if any(w in text_lower for w in ['crash', 'sell', 'dump', 'dead']):
            return BEARISH
        if any(w in text_lower for w in ['moon', 'buy', 'rocket', 'mars']):
            return BULLISH
    return ABSTAIN


def lf_short_post_meme(text):
    """
    Vote MEME if post is very short (under 20 chars).
    Short posts like "GUH", "stonks", "🚀🚀🚀" are usually memes.
    Expected: Decent precision but will miss longer meme posts.
    """
    if len(text.strip()) < 20:
        return MEME
    return ABSTAIN


# ============================================
# FINANCIAL PATTERN LABELING FUNCTIONS
# ============================================

def lf_options_bullish(text):
    """
    Vote BULLISH if post mentions buying calls or selling puts.
    Expected: High precision — options language is directional.
    """
    text_lower = text.lower()
    bullish_options = ['bought calls', 'buying calls', 'loaded calls',
                       'call options', 'selling puts', 'sold puts',
                       'bull spread', 'long calls']
    if any(phrase in text_lower for phrase in bullish_options):
        return BULLISH
    return ABSTAIN


def lf_options_bearish(text):
    """
    Vote BEARISH if post mentions buying puts or selling calls.
    Expected: High precision — options language is directional.
    """
    text_lower = text.lower()
    bearish_options = ['bought puts', 'buying puts', 'loaded puts',
                       'put options', 'selling calls', 'sold calls',
                       'bear spread', 'long puts']
    if any(phrase in text_lower for phrase in bearish_options):
        return BEARISH
    return ABSTAIN


def lf_price_target(text):
    """
    Vote BULLISH if post mentions a price target (usually bullish).
    Pattern: "PT $XXX" or "price target $XXX"
    Expected: Decent precision, may miss bearish price targets.
    """
    if re.search(r'(PT|price target)\s*\$?\d+', text, re.IGNORECASE):
        return BULLISH
    return ABSTAIN


def lf_loss_mention(text):
    """
    Vote MEME if post mentions specific losses (loss porn culture).
    Pattern: "down XX%", "lost $XXX"
    Expected: Could be genuinely bearish OR meme loss porn.
    """
    text_lower = text.lower()
    if re.search(r'down \d+%', text_lower) or re.search(r'lost \$[\d,]+', text_lower):
        return MEME
    return ABSTAIN


# ============================================
# SARCASM/IRONY DETECTION FUNCTIONS
# ============================================

def lf_sarcasm_clown(text):
    """
    Vote BEARISH if post contains 🤡 — usually mocking bullish thesis.
    Expected: Interesting function — clown emoji almost always 
    indicates the author thinks someone is wrong/foolish.
    """
    if '🤡' in text:
        return BEARISH
    return ABSTAIN


def lf_definitely_sarcasm(text):
    """
    Vote BEARISH if post uses "definitely" or "surely" with bullish words.
    Pattern: sarcastic construction like "yeah WISH is definitely 
    going to $100" — actually bearish.
    Expected: Low coverage, but interesting precision dynamics.
    """
    text_lower = text.lower()
    sarcasm_markers = ['definitely', 'surely', 'of course', 'totally',
                       'oh yeah', 'sure thing', 'trust me bro']
    bullish_words = ['moon', 'rocket', '$100', '$1000', 'going up',
                     'squeeze', 'to the moon']
    has_sarcasm = any(m in text_lower for m in sarcasm_markers)
    has_bullish = any(w in text_lower for w in bullish_words)
    if has_sarcasm and has_bullish:
        return BEARISH
    return ABSTAIN
File: src/labeling/label_aggregator.py
Implement label aggregation that combines votes from all labeling functions:
pythonCopydef aggregate_labels(text, labeling_functions):
    """
    Run all labeling functions on a post and combine votes.
    
    Returns:
        dict with:
        - 'final_label': str — the aggregated label
        - 'votes': dict — {function_name: vote} for all non-abstaining functions
        - 'confidence': float — agreement ratio among voting functions
        - 'coverage': bool — whether any function voted
        - 'conflict': bool — whether functions disagreed
    """
    # Collect votes from all functions
    # Filter out ABSTAIN votes
    # Implement majority vote for final label
    # Calculate confidence as (majority_count / total_votes)
    # Flag conflicts where multiple labels received votes
    pass
Implement THREE aggregation strategies:

Majority Vote: Simple — most common non-abstain label wins
Weighted Vote: Weight each function by its estimated accuracy (set manually based on domain knowledge)
Confidence-Weighted: Only assign a label if confidence exceeds a threshold (e.g., 0.6), otherwise mark as "uncertain"

File: src/labeling/label_quality.py
Implement label quality assessment metrics:
pythonCopydef compute_label_quality_report(df, labeling_functions):
    """
    Comprehensive report on labeling function performance.
    
    Metrics per labeling function:
    - Coverage: What % of posts does this function vote on?
    - Polarity: What labels does it assign (distribution)?
    - Conflict rate: How often does it disagree with the majority?
    - Accuracy on gold set: If gold labels exist, how accurate is it?
    
    Aggregate metrics:
    - Overall coverage: What % of posts received at least one vote?
    - Average votes per post: How many functions vote per post?
    - Conflict rate: What % of posts have disagreeing functions?
    - Label distribution: Does the output distribution match expected?
    - Uncertain posts: How many posts couldn't be confidently labeled?
    """
    pass


def compare_labels_to_gold(programmatic_labels, gold_labels):
    """
    Compare programmatically generated labels against gold standard.
    This is the KEY metric — how good is our labeling pipeline?
    
    Returns:
    - Per-class precision, recall, F1
    - Confusion matrix
    - List of disagreements with analysis of why
    """
    pass

CLASSICAL ML PIPELINE
File: src/models/classical_ml.py
pythonCopyclass ClassicalSentimentPipeline:
    """
    TF-IDF + Logistic Regression pipeline for sentiment classification.
    
    This class should support:
    1. Training on programmatically labeled data
    2. Training on gold-labeled data (for comparison)
    3. Prediction with confidence scores
    4. Feature importance extraction
    5. Error analysis on misclassified examples
    """
    
    def __init__(self, max_features=500, ngram_range=(1, 2), C=1.0):
        pass
    
    def preprocess(self, text):
        """
        MINIMAL preprocessing — lowercase + whitespace only.
        Do NOT strip emojis, tickers, or punctuation — they carry signal.
        """
        pass
    
    def train(self, X_train, y_train):
        """Train TF-IDF + LogReg pipeline."""
        pass
    
    def predict(self, X_test):
        """Return predictions and confidence scores."""
        pass
    
    def get_feature_importance(self):
        """
        Return top predictive features per class.
        This is important for explainability and validation 
        that the model learned meaningful patterns.
        """
        pass
    
    def error_analysis(self, X_test, y_test, y_pred):
        """
        Return detailed error analysis:
        - Misclassified examples with text
        - Per-error: which class it was confused with
        - Per-error: what features drove the wrong prediction
        - Categorize errors: labeling issue vs model limitation
        """
        pass
Key Experiment: Label Quality Impact
The most important experiment in the project. Train the SAME model on:

Noisy labels: Random label noise added to gold labels (simulate bad labeling)
Programmatic labels: Output of the labeling function pipeline
Gold labels: Hand-labeled ground truth

Compare F1 across all three. The expected result:

Gold labels → highest F1 (best data = best model)
Programmatic labels → close to gold (Snorkel's thesis)
Noisy labels → worst F1 (bad data = bad model regardless of model complexity)

This single experiment IS the thesis of the project. Make it prominent in the dashboard.

LLM CLASSIFICATION
File: src/models/llm_classifier.py
pythonCopyclass LLMSentimentClassifier:
    """
    OpenAI-based sentiment classifier for comparison.
    
    Supports:
    1. Zero-shot classification
    2. Few-shot classification (with examples)
    3. Structured output parsing
    4. Cost and latency tracking per prediction
    """
    
    def build_prompt(self, text, few_shot_examples=None):
        """
        Build classification prompt.
        
        The prompt should:
        - Define all 4 categories with financial context
        - Handle the meme vs genuine sentiment distinction explicitly
        - Handle sarcasm detection instruction
        - Request structured JSON output
        - Optionally include few-shot examples
        """
        pass
    
    def classify(self, text):
        """
        Classify a single post. Return:
        - label: str
        - confidence: float (if available from API)
        - raw_response: str
        - latency_ms: float
        - token_count: int (for cost calculation)
        """
        pass
    
    def classify_batch(self, texts, batch_size=10):
        """
        Classify multiple posts with rate limiting.
        Track total cost and average latency.
        """
        pass
Important: Include a mock/cached mode so the dashboard works without burning API credits. Cache LLM responses in a JSON file after first run.

ENTITY EXTRACTION
File: src/extraction/ticker_extractor.py
pythonCopyclass TickerExtractor:
    """
    Extract ticker symbols and company names from financial posts.
    
    Challenges specific to this domain:
    - $AAPL is a ticker, but $ can also mean dollar amounts
    - "Apple" could be the company or the fruit (context matters)
    - WSB uses informal names: "Papa Musk" → Tesla, "Zuck" → Meta
    - Some tickers are common words: $AI, $IT, $ALL, $F
    - Emojis as company references: 🍎 → Apple
    """
    
    def __init__(self):
        self.ticker_to_company = {
            'AAPL': 'Apple', 'TSLA': 'Tesla', 'MSFT': 'Microsoft',
            'GOOG': 'Google', 'GOOGL': 'Google', 'AMZN': 'Amazon',
            'NVDA': 'NVIDIA', 'META': 'Meta', 'GME': 'GameStop',
            'AMC': 'AMC Entertainment', 'SPY': 'S&P 500 ETF',
            'QQQ': 'Nasdaq ETF', 'PLTR': 'Palantir', 'BB': 'BlackBerry',
            'SOFI': 'SoFi', 'WISH': 'ContextLogic', 'NFLX': 'Netflix',
            'DIS': 'Disney', 'AMD': 'AMD', 'INTC': 'Intel',
            'JPM': 'JPMorgan', 'BAC': 'Bank of America',
            'F': 'Ford', 'GE': 'GE', 'T': 'AT&T',
        }
        
        self.company_aliases = {
            'apple': 'Apple', 'tesla': 'Tesla', 'microsoft': 'Microsoft',
            'google': 'Google', 'alphabet': 'Google', 'amazon': 'Amazon',
            'nvidia': 'NVIDIA', 'jensen': 'NVIDIA', 'meta': 'Meta',
            'facebook': 'Meta', 'zuck': 'Meta', 'gamestop': 'GameStop',
            'palantir': 'Palantir', 'blackberry': 'BlackBerry',
            'papa musk': 'Tesla', 'elon': 'Tesla',
        }
        
        self.emoji_to_company = {
            '🍎': 'Apple',
        }
    
    def extract(self, text):
        """
        Extract all company/ticker entities.
        Return list of canonical company names.
        """
        pass
    
    def extract_with_evidence(self, text):
        """
        Extract entities with the surface form that triggered extraction.
        Return list of dicts: {canonical: str, surface_form: str, method: str}
        method = 'ticker' | 'company_name' | 'alias' | 'emoji'
        """
        pass
File: src/extraction/normalizer.py
pythonCopyclass EntityNormalizer:
    """
    Normalize entity mentions to canonical forms for fair evaluation.
    
    This is critical because:
    - Rule-based extractor might return "Apple" 
    - LLM might return "Apple Inc."
    - Ground truth might say "AAPL"
    - All three refer to the same entity
    
    Without normalization, evaluation metrics are artificially low.
    """
    
    def __init__(self):
        self.canonical_map = {
            # All variations → single canonical form
            'apple': 'apple', 'apple inc': 'apple', 'apple inc.': 'apple',
            'aapl': 'apple', '$aapl': 'apple',
            'tesla': 'tesla', 'tesla inc': 'tesla', 'tsla': 'tesla',
            '$tsla': 'tesla', 'tesla motors': 'tesla',
            # ... etc for all companies
        }
    
    def normalize(self, entity):
        """Normalize a single entity to canonical form."""
        pass
    
    def entities_match(self, pred_entity, gt_entity):
        """Check if two entity strings refer to the same company."""
        pass
    
    def normalize_set(self, entities):
        """Normalize a list of entities, deduplicate after normalization."""
        pass
File: src/extraction/llm_extractor.py
pythonCopyclass LLMTickerExtractor:
    """
    LLM-based entity extraction for comparison with rule-based.
    
    The prompt should handle:
    - Explicit tickers ($AAPL)
    - Company names (Apple, Tesla)
    - Informal references (Papa Musk → Tesla)
    - Emoji references (🍎 → Apple)
    - Distinguishing ticker symbols from regular words ($F = Ford vs "f")
    - Returning canonical company names, not raw surface forms
    """
    
    def build_extraction_prompt(self, text):
        pass
    
    def extract(self, text):
        """Return list of canonical company names."""
        pass

EVALUATION FRAMEWORK
File: src/evaluation/classification_eval.py
pythonCopydef evaluate_sentiment_classification(y_true, y_pred, texts=None):
    """
    Comprehensive classification evaluation.
    
    Returns:
    - Overall accuracy, weighted F1
    - Per-class precision, recall, F1, support
    - Confusion matrix
    - If texts provided: misclassified examples with analysis
    - Specific analysis of meme vs bullish confusion
    - Specific analysis of question vs complaint confusion
    """
    pass


def evaluate_label_quality_impact(gold_labels, programmatic_labels, 
                                   noisy_labels, X_train, X_test, y_test):
    """
    THE KEY EXPERIMENT: Train same model on different quality labels.
    Compare performance to prove data quality > model complexity.
    
    Returns comparison table:
    | Training Data      | F1    | Precision | Recall |
    | Gold labels        | 0.XX  | 0.XX      | 0.XX   |
    | Programmatic labels| 0.XX  | 0.XX      | 0.XX   |
    | Noisy labels       | 0.XX  | 0.XX      | 0.XX   |
    """
    pass
File: src/evaluation/extraction_eval.py
pythonCopydef evaluate_entity_extraction(predictions, ground_truths, normalizer):
    """
    Entity-level evaluation with normalization.
    
    Args:
        predictions: list of lists of extracted entities
        ground_truths: list of lists of true entities
        normalizer: EntityNormalizer instance
    
    Returns:
        - Entity-level precision, recall, F1
        - Per-company performance breakdown
        - False positive analysis (over-extraction patterns)
        - False negative analysis (what entity types are missed)
        - Normalization impact: metrics WITH vs WITHOUT normalization
          (this demonstrates why normalization matters)
    """
    pass
File: src/evaluation/comparison.py
pythonCopydef compare_classification_approaches(classical_results, llm_results, 
                                       gold_labels):
    """
    Head-to-head comparison of classical ML vs LLM.
    
    Comparison dimensions:
    - F1 by class
    - Performance on ambiguous posts (high ambiguity_score in gold set)
    - Performance on meme detection specifically
    - Latency (ms per prediction)
    - Cost ($ per 1000 predictions)
    - Consistency (run LLM twice, measure agreement)
    
    The narrative should show:
    - LLM is better on ambiguous/sarcastic posts
    - Classical ML is better on clear-cut posts
    - Classical ML is 1000x cheaper and faster
    - Hybrid approach (LLM for labeling, classical for production) is optimal
    """
    pass


def compare_extraction_approaches(rules_results, llm_results, gold_entities):
    """
    Head-to-head comparison of rule-based vs LLM extraction.
    
    With and without normalization to show normalization impact.
    """
    pass

STREAMLIT DASHBOARD
File: app/streamlit_app.py
pythonCopy"""
MarketPulse Dashboard — Data-Centric Sentiment Intelligence

Multi-page Streamlit app with 6 sections.
Use st.set_page_config(layout="wide") for full-width layout.
Use a dark theme that feels financial/terminal-like.
Sidebar should show project description and navigation.
"""
Page 1: Data Exploration (pages/1_data_exploration.py)
Display:

Dataset overview (row count, column types, sample posts)
Label distribution bar chart (Plotly)
Text length distribution histogram
Word cloud per sentiment category
Post source distribution (reddit/twitter/stocktwits)
Interactive table: filter by label, search text, sort by upvotes
"Messiness audit": count of emojis, tickers, ALL CAPS, question marks per class

Page 2: Labeling Studio (pages/2_labeling_studio.py)
Display:

List of all labeling functions with descriptions
For each function: coverage %, label distribution pie chart, example posts it labeled
Interactive demo: enter a post → see which functions fire and how they vote
Vote aggregation visualization: show how individual votes combine into final label
Highlight conflicts: posts where functions disagree, show the votes side by side

Page 3: Label Quality (pages/3_label_quality.py)
Display:

THE KEY CHART: Bar chart comparing F1 of models trained on gold vs programmatic vs noisy labels. This is the thesis visualization.
Coverage heatmap: which functions cover which posts
Conflict matrix: which function pairs disagree most often
Confidence distribution: histogram of label confidence scores
Uncertain posts: table of posts that couldn't be confidently labeled
Gold comparison: programmatic labels vs gold labels confusion matrix

Page 4: Model Comparison (pages/4_model_comparison.py)
Display:

Side-by-side classification reports (classical ML vs LLM)
Per-class F1 comparison bar chart
Performance by difficulty: how each model performs on easy vs ambiguous posts
Cost comparison: time and money per 1000 predictions
Meme detection deep dive: which model handles memes better
Sarcasm handling: examples of sarcastic posts and how each model classified them
Interactive: enter a post → see both models' predictions + confidence

Page 5: Entity Extraction (pages/5_entity_extraction.py)
Display:

Entity-level P/R/F1 for rule-based vs LLM extraction
Most mentioned companies bar chart
Normalization impact: metrics with vs without normalization (side by side)
Extraction evidence viewer: for each post, show what was extracted and how (ticker match, alias, emoji, etc.)
Error analysis: false positives and false negatives with explanations
Interactive: enter a post → see extracted entities from both approaches

Page 6: Error Analysis (pages/6_error_analysis.py)
Display:

Filterable table of all misclassified posts across both models
Error categorization: "labeling ambiguity" vs "model limitation" vs "data quality issue"
Most confused class pairs
Feature analysis: what words/features drove incorrect predictions
Recommended data improvements: specific suggestions for labeling guideline refinements
Interactive error explorer: click an error → see full context, model scores, labeling function votes


VISUAL DESIGN
Use Plotly for all charts. Financial/dark theme:
pythonCopy# Color scheme
COLORS = {
    'bullish': '#00C853',     # Green
    'bearish': '#FF1744',     # Red
    'neutral': '#90A4AE',     # Gray
    'meme': '#FFD600',        # Yellow/Gold
    'background': '#0D1117',  # Dark
    'surface': '#161B22',     # Slightly lighter dark
    'text': '#E6EDF3',        # Light text
    'accent': '#58A6FF',      # Blue accent
}

REQUIREMENTS
Copy# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
streamlit>=1.29.0
openai>=1.0.0
python-dotenv>=1.0.0
wordcloud>=1.9.0

README.md STRUCTURE
Write a professional README with:

Hero section: Project name, one-line description, screenshot of dashboard
The Thesis: "Data quality > model complexity" with brief explanation
Key Results: The comparison table showing gold vs programmatic vs noisy label performance
Architecture diagram: ASCII or Mermaid diagram of the pipeline
Quick Start: How to install, set up API keys, run the dashboard
Pipeline Walkthrough: Brief description of each stage
Built With: Tech stack badges
Author: Link to LinkedIn, GitHub


IMPLEMENTATION ORDER
Build in this sequence:
CopyPhase 1: Data & Foundation
  1. Generate synthetic dataset (data/raw/wsb_posts_raw.csv)
  2. Create gold standard labels (data/gold/gold_standard.csv)
  3. Implement data loading and exploration (src/data/)

Phase 2: Labeling Pipeline  
  4. Implement all labeling functions (src/labeling/labeling_functions.py)
  5. Implement label aggregation (src/labeling/label_aggregator.py)
  6. Implement label quality metrics (src/labeling/label_quality.py)
  7. Generate programmatic labels for full dataset

Phase 3: Models
  8. Implement classical ML pipeline (src/models/classical_ml.py)
  9. Implement LLM classifier (src/models/llm_classifier.py)
  10. Run the key experiment: gold vs programmatic vs noisy labels

Phase 4: Entity Extraction
  11. Implement rule-based ticker extraction (src/extraction/)
  12. Implement entity normalization (src/extraction/normalizer.py)
  13. Implement LLM extraction (src/extraction/llm_extractor.py)

Phase 5: Evaluation
  14. Implement classification evaluation (src/evaluation/)
  15. Implement extraction evaluation with normalization
  16. Implement head-to-head comparison

Phase 6: Dashboard
  17. Build Streamlit app shell and navigation
  18. Build each page in order (1 through 6)
  19. Polish visual design and interactivity

Phase 7: Polish
  20. Write README.md
  21. Add tests
  22. Clean up code and add docstrings
  23. Take dashboard screenshots for README

TESTING REQUIREMENTS
pythonCopy# tests/test_labeling_functions.py
# Test each labeling function with clear positive, negative, 
# and edge case examples

def test_lf_bullish_keywords():
    assert lf_bullish_keywords("loading up on TSLA calls") == BULLISH
    assert lf_bullish_keywords("this stock is terrible") == ABSTAIN
    assert lf_bullish_keywords("definitely not buying this") == BULLISH  
    # ^ This is a known limitation — keyword match doesn't understand negation
    # Document this as a known weakness

def test_lf_meme_vs_bullish():
    """Test that meme function catches meme language even with bullish words."""
    assert lf_meme_keywords("apes together strong TSLA to the moon") == MEME
    # But note: lf_bullish_keywords would ALSO fire on this post
    # This is expected — that's why we need aggregation

# tests/test_normalizer.py
# Test entity normalization thoroughly
def test_normalize_ticker_variations():
    normalizer = EntityNormalizer()
    assert normalizer.normalize("AAPL") == "apple"
    assert normalizer.normalize("$AAPL") == "apple"
    assert normalizer.normalize("Apple") == "apple"
    assert normalizer.normalize("Apple Inc.") == "apple"
    # All should resolve to same canonical form

# tests/test_evaluation.py
# Test that evaluation correctly handles normalization
def test_evaluation_with_normalization():
    preds = [["AAPL", "TSLA"]]
    gts = [["Apple", "Tesla"]]
    results = evaluate_entity_extraction(preds, gts, normalizer)
    assert results['f1'] == 1.0  # Should match after normalization

KEY PRINCIPLES FOR IMPLEMENTATION
Copy1. DATA-CENTRIC FIRST
   Every design decision should reinforce that data quality 
   matters more than model complexity. The dashboard should 
   make this visually obvious.

2. SHOW THE CONFLICTS
   Don't hide when labeling functions disagree. The conflicts 
   are the interesting part — they reveal ambiguity in the data.

3. NORMALIZATION IS NOT OPTIONAL
   Always show metrics WITH and WITHOUT normalization 
   side by side. The difference proves why normalization matters.

4. ERRORS ARE INSIGHTS
   Every misclassification should be categorized as either 
   a data/labeling issue or a genuine model limitation.

5. MAKE IT INTERACTIVE
   The dashboard should let users type in their own posts 
   and see the full pipeline process them in real time.

6. FINANCIAL DOMAIN AUTHENTICITY
   Posts should feel real. Use actual WSB language, real 
   ticker symbols, realistic financial scenarios. Don't 
   sanitize the data — messiness is the point.

7. THE THESIS CHART
   The single most important visualization: bar chart 
   showing model performance on gold vs programmatic vs 
   noisy labels. This chart IS the project.