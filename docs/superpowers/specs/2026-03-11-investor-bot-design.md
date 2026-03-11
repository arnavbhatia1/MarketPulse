# MarketPulse Investor Bot — Design Spec

**Date:** 2026-03-11
**Status:** Approved
**Feature:** AI-powered investment analyst bot with proprietary quantitative formula, Claude review layer, and paper trading execution.

---

## Overview

A premium feature that adds an autonomous investment bot to MarketPulse. The bot combines the existing sentiment pipeline with public financial data (yfinance) to make paper trades, manage risk, and build long-term wealth for customers. It uses a hybrid decision engine: a proprietary quantitative formula generates trade candidates, and Claude reviews/approves them.

**Key decisions:**
- Paper trading v1, designed for real brokerage swap-in (Broker abstraction)
- Stocks + ETFs (39 ETFs, all ERs < 0.10%)
- yfinance for market data (free, no API key)
- Hybrid brain: quantitative formula + Claude review
- Auto-pilot with customer override (advisory mode)
- Weekly rebalance + event-driven triggers
- Simple auth gate (`is_premium` flag, no Stripe in v1)
- Configurable portfolios: starting capital, risk profile, investment horizon

---

## Section 1: The Proprietary Formula ("Alpha Engine")

### Signal Inputs (4 dimensions)

**1. Sentiment Signal** (from existing MarketPulse pipeline)
- `sentiment_score`: Maps dominant sentiment to numeric (-1.0 bearish to +1.0 bullish)
- `sentiment_momentum`: Direction of change over the last 7 days
- `confidence`: Average labeling confidence for this ticker
- `source_agreement`: Do Reddit, Stocktwits, and News agree? Consensus = stronger signal

**2. Valuation Signal** (from yfinance)
- `pe_ratio`: Price-to-earnings relative to sector median
- `price_to_book`: Asset value backing
- `dividend_yield`: Income component
- `market_cap_category`: Large/mid/small cap (affects risk weighting)

**3. Momentum Signal** (from yfinance price history)
- `price_momentum_30d`: 30-day price return
- `price_momentum_90d`: 90-day price return
- `volatility`: Standard deviation of daily returns
- `relative_strength`: Performance vs SPY benchmark

**4. Risk Signal** (computed)
- `sector_concentration`: How much of the portfolio is in this ticker's sector
- `geographic_concentration`: US vs international exposure
- `correlation_to_portfolio`: Diversification contribution
- `drawdown_risk`: Max drawdown in recent history

### Sub-Composite Calculations

Each sub-composite is normalized to a 0-100 scale before weighting.

**Sentiment Composite (0-100):**
```
sentiment_raw = (
    0.40 * normalize(sentiment_score, -1.0, 1.0)     # current sentiment
  + 0.25 * normalize(sentiment_momentum, -1.0, 1.0)  # 7-day trend direction
  + 0.20 * normalize(confidence, 0.0, 1.0)           # labeling confidence
  + 0.15 * normalize(source_agreement, 0.0, 1.0)     # cross-source consensus
) * 100
```

**Valuation Composite (0-100):**
```
pe_score = percentile_rank(sector_median_pe / pe_ratio)  # lower PE = higher score
pb_score = percentile_rank(1.0 / price_to_book)          # lower P/B = higher score
div_score = percentile_rank(dividend_yield)               # higher yield = higher score
cap_score = {large: 70, mid: 50, small: 30}              # large cap = safer

valuation_raw = (0.35 * pe_score + 0.30 * pb_score + 0.20 * div_score + 0.15 * cap_score)
```

Special cases:
- Negative PE (unprofitable): `pe_score = 0`
- PE/PB unavailable (common for ETFs): skip valuation signal, redistribute weight equally to sentiment and momentum. See "Missing Data Handling" below.

**Momentum Composite (0-100):**
```
mom_raw = (
    0.35 * percentile_rank(price_momentum_30d)
  + 0.30 * percentile_rank(price_momentum_90d)
  + 0.20 * percentile_rank(relative_strength)     # vs SPY
  + 0.15 * percentile_rank(-volatility)            # lower vol = higher score
) * 100
```

**Risk Penalty (0-100):**
```
risk_raw = (
    0.35 * normalize(sector_concentration / sector_limit, 0.0, 1.5)
  + 0.25 * normalize(geographic_concentration_deviation, 0.0, 0.5)
  + 0.25 * normalize(correlation_to_portfolio, -1.0, 1.0)
  + 0.15 * normalize(drawdown_risk, 0.0, 0.5)
) * 100
```

Where `normalize(value, min, max)` clamps to [0, 1] then scales to context. `percentile_rank()` ranks across all tickers in the current scoring universe, returning 0-100.

### Missing Data Handling (yfinance gaps)

yfinance frequently returns `None`/`NaN` for fundamentals, especially on ETFs.

**Rules:**
- If a signal dimension has **no data at all** (e.g., ETF with no PE/PB): set that composite to `None` and redistribute its weight equally among the remaining dimensions.
- If a signal dimension has **partial data** (e.g., PE available but PB missing): score only the available sub-signals, re-normalizing their internal weights to sum to 1.0.
- If **yfinance is entirely down** (no price data): skip the ticker for this cycle. Log a warning. Do not sell existing holdings just because data is temporarily unavailable.

Example: scoring an ETF (no valuation data) with Moderate weights:
- Original: sentiment 0.25, valuation 0.30, momentum 0.25, risk 0.20
- Adjusted: sentiment 0.40, momentum 0.40, risk 0.20 (valuation's 0.30 split equally)

### Composite Score Formula

Each ticker gets a composite score (0-100):

```
raw_score = (
    w_sentiment  * sentiment_composite
  + w_valuation  * valuation_composite
  + w_momentum   * momentum_composite
  - w_risk       * risk_penalty
)

score = clamp(raw_score, 0, 100)
```

The subtraction of risk_penalty means raw scores can theoretically go negative (high-risk ticker with poor fundamentals). Clamping to [0, 100] handles this. In practice, a score near 0 is a strong sell signal; near 100 is a strong buy.

**Weights by risk profile:**

| Weight | Conservative | Moderate | Aggressive |
|--------|-------------|----------|------------|
| `w_sentiment` | 0.15 | 0.25 | 0.35 |
| `w_valuation` | 0.40 | 0.30 | 0.20 |
| `w_momentum` | 0.15 | 0.25 | 0.30 |
| `w_risk` | 0.30 | 0.20 | 0.15 |

### Ticker Universe

The scoring universe consists of two fixed sets:

1. **ETF Universe** (39 ETFs): Always scored. Defined in `etf_universe` table. Static set seeded on first run.
2. **Stock Watchlist**: The tickers from `ticker_extractor.py`'s `ticker_map` (~42 tickers). Always scored regardless of whether they appear in recent sentiment data.

**Universe rules:**
- A stock does NOT leave the universe just because sentiment data is absent. It gets scored on valuation + momentum only (sentiment composite = neutral 50).
- A stock enters the portfolio only via a buy signal. It exits only via a sell signal or manual override — never because it fell out of the ingestion window.
- The watchlist can be extended via config. Future: dynamic watchlist expansion based on trending tickers.

### Trade Generation

1. Score all tickers in the universe (~42 stocks + 39 ETFs)
2. Rank by score
3. **Buy signal**: Score >= 65 AND not in portfolio, or existing holding with score >= 65 that is underweight (current weight < target weight by >2%)
4. **Sell signal**: Score < 35 for existing holdings, or holdings that are overweight (current weight > target weight by >3%)
5. **Hold**: Score between 35-65, or within 2-3% of target weight
6. Apply position sizing (see below)
7. Pass candidates to Claude for review

### Position Sizing

For buy orders, allocation is **score-proportional within budget**:

```
buy_budget = min(available_cash * 0.80, portfolio_value * max_position_pct)

For N buy candidates ranked by score:
  weight_i = score_i / sum(all_buy_scores)
  allocation_i = buy_budget * weight_i
  shares_i = floor(allocation_i / price_i)
```

- Never allocate more than `max_position_pct` to a single holding (5%/8%/12% by profile)
- Reserve 20% of cash as buffer (don't deploy 100% of available cash in one cycle)
- Round down to whole shares (no fractional shares in v1)

For sell orders, the sell amount is the lesser of:
- Enough shares to bring the position back to target weight
- All shares if score < 20 (strong sell — full exit)

---

## Section 2: Risk Management Framework

Five layers of protection.

### Layer 1: Position Limits

| Constraint | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| Max single position | 5% | 8% | 12% |
| Max sector exposure | 20% | 30% | 40% |
| Min cash reserve | 15% | 10% | 5% |
| Max individual stocks (vs ETFs) | 50% | 70% | 85% |

Any trade that would violate these limits gets blocked.

### Layer 2: Sector Concentration Risk

- Track each holding's GICS sector via yfinance
- ETFs decomposed by primary sector exposure (XLK = Technology, SPY = diversified)
- When a sector approaches its limit, the engine favors other sectors or broad ETFs
- **Rebalance trigger**: If any sector exceeds limit by 5%+ due to price appreciation, flag for trimming

### Layer 3: Geographic Diversification

| Allocation | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| US Domestic | 60% | 70% | 80% |
| International Developed | 30% | 20% | 15% |
| Emerging Markets | 10% | 10% | 5% |

International ETFs (VEA, VWO, VXUS) are the primary tool for geographic exposure.

### Layer 4: Recession Stress Testing

Apply historical recession drawdowns to current portfolio with sector-specific sensitivity multipliers:

**Sector Sensitivity Coefficients:**

| Sector | 2008 Multiplier | 2020 Multiplier | 2022 Multiplier |
|--------|----------------|----------------|----------------|
| Technology | 1.20 | 0.85 | 1.40 |
| Consumer Discretionary | 1.30 | 1.10 | 1.15 |
| Financials | 1.50 | 1.00 | 0.90 |
| Energy | 1.10 | 1.40 | 0.60 |
| Healthcare | 0.75 | 0.70 | 0.85 |
| Consumer Staples | 0.60 | 0.65 | 0.80 |
| Utilities | 0.55 | 0.70 | 0.75 |
| Industrials | 1.15 | 1.05 | 1.00 |
| Real Estate | 1.40 | 0.90 | 1.20 |
| Materials | 1.10 | 0.95 | 1.00 |
| Communication Services | 0.90 | 0.80 | 1.30 |
| Bonds/Fixed Income | 0.20 | 0.15 | 0.90 |
| Broad Market ETFs | 1.00 | 1.00 | 1.00 |

**Stress score calculation:**
```
For each scenario (2008, 2020, 2022):
  portfolio_drawdown = sum(holding_weight * base_drawdown * sector_multiplier)

stress_score = average(portfolio_drawdown_2008, portfolio_drawdown_2020, portfolio_drawdown_2022)
```

**Stress score thresholds by risk profile:**

| Profile | Warning | Action Required |
|---------|---------|-----------------|
| Conservative | > 0.20 | > 0.25 |
| Moderate | > 0.28 | > 0.33 |
| Aggressive | > 0.35 | > 0.40 |

When "Action Required" threshold is breached, the engine shifts toward defensive ETFs (XLU, XLP, VIG, BND, VGIT).

### Layer 5: Hedging Strategies (ETF-based)

- **Inverse correlation**: When portfolio is equity-heavy, allocate to bond ETFs (BND, VGIT) and treasury ETFs (VGSH, VGLT)
- **Defensive rotation**: In high-stress environments, rotate from growth ETFs to value/dividend ETFs
- **Cash as hedge**: Raise cash reserve above minimum when aggregate sentiment turns bearish
- **Inflation protection**: VTIP (TIPS) allocation when inflation signals emerge

### Event-Driven Triggers (mid-cycle)

| Trigger | Condition | Action |
|---------|-----------|--------|
| Sentiment flip | Holding's sentiment shifts bullish to bearish (or vice versa) with confidence > 0.7 | Flag for review |
| Drawdown alert | Position drops >10% from purchase price | Evaluate stop-loss |
| Concentration breach | Sector exceeds limit due to price moves | Flag for trimming |
| Stress spike | Multiple holdings in recession-sensitive sectors trending bearish simultaneously | Defensive rotation |

---

## Section 3: Portfolio Management & Trade Execution

### Portfolio State Model

```
Portfolio:
  portfolio_id, user_id, starting_capital, current_cash
  risk_profile (conservative | moderate | aggressive)
  investment_horizon (1yr | 5yr | 10yr+)
  mode (autopilot | advisory)
  created_at, last_rebalanced_at

Holdings:
  portfolio_id, symbol, shares, avg_cost_basis
  current_price, current_value, weight
  sector, geography, asset_type (stock | etf)
  acquired_at

Trades:
  trade_id, portfolio_id, symbol
  action (buy | sell), shares, price, total_value
  reason, claude_review
  status (proposed | approved | executed | rejected | overridden)
  trigger (weekly_rebalance | event_driven | manual)
  proposed_at, executed_at
```

### Trade Lifecycle

```
Formula scores tickers
        |
Generate trade candidates (buy/sell/rebalance)
        |
Risk checks -- block any that violate limits
        |
Claude reviews remaining candidates
  -> Approves with commentary
  -> Vetoes with reason
        |
    Mode = autopilot?
      YES -> execute immediately
      NO  -> queue for user approval
        |
Execute: update holdings, cash, log trade
        |
Record performance snapshot
```

### Paper Trading Execution

- Look up current price via yfinance
- Deduct/add cash, update shares held
- Record trade with timestamp and price
- No slippage modeling in v1 (assume market price fills)

### Brokerage Abstraction

```python
class Broker(ABC):
    def get_price(symbol) -> float
    def execute_buy(symbol, shares) -> TradeResult
    def execute_sell(symbol, shares) -> TradeResult
    def get_positions() -> list[Position]

class PaperBroker(Broker):
    # SQLite-backed paper trading

class AlpacaBroker(Broker):
    # Future: real brokerage
```

### Performance Tracking (daily snapshots)

- `total_value` (cash + holdings at market price)
- `daily_return`, `cumulative_return`
- `benchmark_return` (vs SPY)
- `sharpe_ratio` (rolling risk-adjusted return)
- `max_drawdown` (worst peak-to-trough)

---

## Section 4: Claude Review Layer

### Review Prompt Structure

Each rebalancing cycle, Claude receives:
- Current portfolio state (value, cash, allocation percentages)
- Risk profile and constraints
- Proposed trades with formula scores and reasons
- Market context from MarketPulse (overall sentiment, trending tickers)
- Risk check summary (stress score, sector exposure, cash level)

Trades are referenced by `trade_id` (UUID) in both the prompt and the response, ensuring unambiguous mapping.

### Review Output (structured JSON)

```json
{
  "decisions": [
    {
      "trade_id": "abc-123",
      "action": "APPROVE",
      "reasoning": "1-2 sentence explanation"
    }
  ],
  "portfolio_commentary": "Overall assessment for customer display"
}
```

### Guardrails

- Claude can only APPROVE or VETO — cannot invent new trades
- Vetoed trades are skipped (or replaced if Claude suggests an alternative that's already in the scored universe)
- **API failure** (network error, 5xx, timeout) = formula-only mode, logged as `review: "unavailable"`
- **Malformed JSON response** (200 but unparseable): retry once with same prompt. If second attempt also fails, fall back to formula-only mode. Log the raw response for debugging.
- One API call per rebalance cycle (not per trade)
- `max_tokens=500`
- `portfolio_commentary` becomes user-facing explanation on dashboard

---

## Section 5: Authentication & Paywall

### Auth Model

```sql
users (
    user_id TEXT PRIMARY KEY,       -- UUID
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,     -- bcrypt
    is_premium INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
)
```

- Registration: email + password, bcrypt hash
- Login: verify hash, set `st.session_state.user`
- Session: Streamlit session state (no JWT)
- Premium toggle: manual database flag (Stripe later)
- **Password reset: out of scope for v1.** Manual process via direct DB update.
- **Email verification: out of scope for v1.**

### Access Control

| Page | Unauthenticated | Free User | Premium User |
|------|----------------|-----------|-------------|
| Login/Register | Yes | N/A | N/A |
| MarketPulse Home | Yes (via "Continue as Guest") | Yes | Yes |
| Ticker Detail | Yes | Yes | Yes |
| Portfolio Dashboard | No | Locked overlay | Full access |
| Trade Activity | No | Locked overlay | Full access |

### App Entry Point Architecture

`MarketPulse.py` remains the Streamlit entry point (`streamlit run app/MarketPulse.py`). Auth is embedded as an inline gate:

```
MarketPulse.py loads:
  if not authenticated and not guest:
    show login/register form (inline, full-page)
    "Continue as Guest" button sets session_state.guest = True
  else:
    show existing MarketPulse home page
```

This is **not** a separate `Login.py` page — it is a conditional render within the existing entry point. This avoids breaking the Streamlit multi-page architecture and the existing launch command.

**UX change acknowledged:** Existing users who previously saw the dashboard immediately will now see a login screen first. The "Continue as Guest" button provides a one-click bypass to the existing free experience. This is an intentional trade-off for the paywall.

- `require_auth(premium=False)` helper at the top of protected pages (Portfolio, Trades)

---

## Section 6: UI & Dashboard

### New Pages

**Portfolio Dashboard** (`app/pages/2_Portfolio.py`)
- Header: portfolio value, daily change ($ and %), cash, risk profile badge
- Performance chart: cumulative return vs SPY, selectable timeframes (1W, 1M, 3M, ALL)
- Holdings table: symbol, shares, avg cost, current price, gain/loss, weight %, sector (sortable)
- Allocation rings: sector breakdown + geographic breakdown (donut charts)
- Risk panel: recession stress gauge, sector concentration bars, top risk flags
- Bot status: last rebalance, mode (autopilot/advisory), next scheduled rebalance
- **"Rebalance Now" button**: triggers an immediate rebalance cycle (same as weekly, but on-demand)

**Trade Activity** (`app/pages/3_Trades.py`)
- Pending trades (advisory mode): cards with formula reasoning + Claude commentary, Approve/Reject buttons
- Trade history: filterable/sortable table
- Bot commentary: Claude's portfolio_commentary styled like existing briefing card

### Navigation Flow

```
App opens -> MarketPulse.py
  -> Not authenticated? Show login/register form
    -> "Continue as Guest" -> MarketPulse Home (free tier, unchanged)
    -> Sign In/Register -> MarketPulse Home
                            |-- Ticker Detail (unchanged)
                            |-- Portfolio Dashboard (premium only)
                            |-- Trade Activity (premium only)
```

### Styling

- Same dark theme (#0D1117 background, green/red colors)
- Portfolio gain = green (#00C853), loss = red (#FF1744)
- New components follow existing patterns in styles.py and charts.py
- Portfolio onboarding: "Set Up Your Portfolio" card with starting capital slider, risk profile radios, horizon dropdown

---

## Section 7: Database Schema

### Migration Strategy

- New tables use `CREATE TABLE IF NOT EXISTS` — same pattern as existing `init_db()`.
- Added to the existing `init_db()` function in `db.py` as additional `CREATE TABLE` statements.
- **Foreign keys**: Enable `PRAGMA foreign_keys = ON` at connection time for new code paths (investor module). Existing code paths are unaffected — they don't reference the new tables.
- **No destructive migrations**: All changes are additive. Existing `data/marketpulse.db` files gain new tables on next `init_db()` call. No data loss.
- ETF universe table is seeded via an `_seed_etf_universe()` function called after table creation (idempotent — skips if data exists).

### New Tables

```sql
users (
    user_id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_premium INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
)

portfolios (
    portfolio_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    name TEXT DEFAULT 'My Portfolio',
    starting_capital REAL NOT NULL,
    current_cash REAL NOT NULL,
    risk_profile TEXT NOT NULL,
    investment_horizon TEXT NOT NULL,
    mode TEXT DEFAULT 'autopilot',
    is_active INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    last_rebalanced_at TEXT
)

holdings (
    id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
    symbol TEXT NOT NULL,
    company_name TEXT,
    shares REAL NOT NULL,
    avg_cost_basis REAL NOT NULL,
    asset_type TEXT NOT NULL,
    sector TEXT,
    geography TEXT,
    acquired_at TEXT NOT NULL,
    UNIQUE(portfolio_id, symbol)
)

trades (
    trade_id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    total_value REAL NOT NULL,
    formula_score REAL,
    reason TEXT,
    claude_review TEXT,
    status TEXT NOT NULL,
    trigger TEXT,
    proposed_at TEXT NOT NULL,
    executed_at TEXT
)

portfolio_snapshots (
    id TEXT PRIMARY KEY,
    portfolio_id TEXT NOT NULL REFERENCES portfolios(portfolio_id),
    snapshot_date TEXT NOT NULL,
    total_value REAL NOT NULL,
    cash_value REAL NOT NULL,
    holdings_value REAL NOT NULL,
    daily_return REAL,
    cumulative_return REAL,
    benchmark_return REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    sector_allocation TEXT,
    geo_allocation TEXT,
    stress_score REAL,
    UNIQUE(portfolio_id, snapshot_date)
)

etf_universe (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    sector TEXT,
    geography TEXT,
    expense_ratio REAL NOT NULL,
    description TEXT
)
```

### ETF Universe (39 ETFs, all ERs < 0.10%)

**Broad US Market:**

| Symbol | Name | ER | Purpose |
|--------|------|-----|---------|
| VOO | Vanguard S&P 500 | 0.03% | Core US large cap |
| VTI | Vanguard Total Stock Market | 0.03% | Full US market |
| SPY | SPDR S&P 500 | 0.09% | Benchmark / liquidity |
| IVV | iShares Core S&P 500 | 0.03% | S&P 500 alternative |

**Size & Style:**

| Symbol | Name | ER | Purpose |
|--------|------|-----|---------|
| VV | Vanguard Large Cap | 0.04% | Large cap core |
| VO | Vanguard Mid Cap | 0.04% | Mid cap exposure |
| VB | Vanguard Small Cap | 0.05% | Small cap |
| VXF | Vanguard Extended Market | 0.06% | Mid + small ex-S&P 500 |
| MGK | Vanguard Mega Cap Growth | 0.07% | Growth tilt |
| MGV | Vanguard Mega Cap Value | 0.07% | Value tilt |
| VBR | Vanguard Small Cap Value | 0.07% | Small value factor |
| VOE | Vanguard Mid Cap Value | 0.07% | Mid value factor |

**Dividend / Income:**

| Symbol | Name | ER | Purpose |
|--------|------|-----|---------|
| VIG | Vanguard Dividend Appreciation | 0.06% | Dividend growth |
| VYM | Vanguard High Dividend Yield | 0.06% | High yield income |
| SCHD | Schwab US Dividend | 0.06% | Quality dividend |

**International:**

| Symbol | Name | ER | Purpose |
|--------|------|-----|---------|
| VEA | Vanguard Developed Markets | 0.05% | Intl developed |
| VWO | Vanguard Emerging Markets | 0.08% | Emerging markets |
| VXUS | Vanguard Total International | 0.07% | All ex-US |
| VT | Vanguard Total World | 0.07% | Global single-fund |
| SCHF | Schwab International Equity | 0.06% | Developed alt |
| IEMG | iShares Core Emerging | 0.09% | Emerging alt |

**Sector (SPDR Select, all 0.09%):**

XLK (Technology), XLF (Financials), XLV (Healthcare), XLE (Energy), XLI (Industrials), XLP (Consumer Staples), XLU (Utilities), XLC (Communication Services), XLRE (Real Estate), XLB (Materials), XLY (Consumer Discretionary)

**Fixed Income:**

| Symbol | Name | ER | Purpose |
|--------|------|-----|---------|
| BND | Vanguard Total Bond | 0.03% | Core bond hedge |
| BNDX | Vanguard International Bond | 0.07% | Intl fixed income |
| VGSH | Vanguard Short-Term Treasury | 0.04% | Cash-like safety |
| VGIT | Vanguard Intermediate Treasury | 0.04% | Duration hedge |
| VGLT | Vanguard Long-Term Treasury | 0.04% | Rate-sensitive hedge |
| VTIP | Vanguard TIPS | 0.04% | Inflation protection |
| AGG | iShares Core Aggregate Bond | 0.03% | Bond alternative |

**Notable exclusions (ER >= 0.10%):** QQQ (0.20%), IWM (0.19%), GLD (0.40%), Vanguard sector ETFs (all exactly 0.10%)

---

## Section 8: File Structure

### New Files

```
src/
  auth/
    auth.py                  # Registration, login, password hashing, session
  investor/
    engine.py                # Alpha Engine -- composite scoring formula
    portfolio.py             # Portfolio CRUD, holdings, position sizing
    risk.py                  # 5-layer risk framework
    market_data.py           # yfinance wrapper
    rebalancer.py            # Weekly + event-driven orchestration
    broker.py                # Broker ABC + PaperBroker
  agent/
    reviewer.py              # Claude trade review layer (shares Anthropic client setup with briefing.py)

app/
  pages/
    2_Portfolio.py            # Portfolio dashboard
    3_Trades.py              # Trade activity
  components/
    auth_guard.py            # require_auth(), premium gate

scripts/
  rebalance.py               # CLI: run rebalance cycle (cron target)
```

Note: `Login.py` is removed as a separate file. Auth UI is inline in `MarketPulse.py` (see Section 5).

### Modified Files

| File | Change | Impact |
|------|--------|--------|
| `src/storage/db.py` | New table creation, migration, portfolio CRUD queries, `PRAGMA foreign_keys = ON` | Medium |
| `app/MarketPulse.py` | Inline auth gate (login/register form), nav updates, guest mode | Medium |
| `app/components/charts.py` | Portfolio performance, allocation, stress charts | Additive |
| `app/components/styles.py` | Portfolio card, trade card, auth form CSS | Additive |
| `config/default.yaml` | `investor:` section with weights, limits, ETFs, triggers | Additive |

### Rebalance Scheduling

The weekly rebalance is triggered via two mechanisms:

1. **CLI script** (`scripts/rebalance.py`): Runs a full rebalance cycle for all active portfolios. Designed to be called by an OS-level cron job (e.g., `0 9 * * 1 python scripts/rebalance.py` for Monday 9 AM). This is the primary production mechanism.

2. **"Rebalance Now" button** on the Portfolio Dashboard: Triggers the same rebalance logic for the current user's portfolio, on-demand. Used for testing, first-time setup, or when a user wants to act on new information immediately.

Event-driven triggers are checked whenever the sentiment pipeline runs (`refresh_pipeline()`). If a trigger fires, it creates proposed trades (status: `proposed`) that are either auto-executed (autopilot) or queued for user review (advisory). This piggybacks on the existing pipeline refresh — no separate background process needed.

### Config Additions (config/default.yaml)

```yaml
investor:
  formula_weights:
    conservative: {sentiment: 0.15, valuation: 0.40, momentum: 0.15, risk: 0.30}
    moderate:     {sentiment: 0.25, valuation: 0.30, momentum: 0.25, risk: 0.20}
    aggressive:   {sentiment: 0.35, valuation: 0.20, momentum: 0.30, risk: 0.15}
  position_limits:
    conservative: {max_position: 0.05, max_sector: 0.20, min_cash: 0.15, max_stocks: 0.50}
    moderate:     {max_position: 0.08, max_sector: 0.30, min_cash: 0.10, max_stocks: 0.70}
    aggressive:   {max_position: 0.12, max_sector: 0.40, min_cash: 0.05, max_stocks: 0.85}
  geo_targets:
    conservative: {us: 0.60, intl_developed: 0.30, emerging: 0.10}
    moderate:     {us: 0.70, intl_developed: 0.20, emerging: 0.10}
    aggressive:   {us: 0.80, intl_developed: 0.15, emerging: 0.05}
  scoring:
    buy_threshold: 65
    sell_threshold: 35
    strong_sell_threshold: 20
    underweight_tolerance: 0.02
    overweight_tolerance: 0.03
    cash_deploy_ratio: 0.80
  rebalance:
    schedule: "weekly"
    event_triggers:
      sentiment_flip_confidence: 0.7
      drawdown_threshold: 0.10
      concentration_breach_buffer: 0.05
  stress_scenarios:
    recession_2008: -0.38
    covid_2020: -0.34
    rate_hike_2022: -0.25
  stress_thresholds:
    conservative: {warning: 0.20, action: 0.25}
    moderate:     {warning: 0.28, action: 0.33}
    aggressive:   {warning: 0.35, action: 0.40}
```

### Module Dependency Flow

```
market_data.py --> engine.py --> rebalancer.py --> broker.py
                      ^               ^
ticker_cache (SQLite) |       reviewer.py (Claude)
                                      |
                              portfolio.py --> db.py
```

No circular dependencies. Each module has a single responsibility. The rebalancer orchestrates the full cycle. `reviewer.py` shares the Anthropic client initialization pattern from `briefing.py` (same `ANTHROPIC_API_KEY` env var, same fallback behavior).
