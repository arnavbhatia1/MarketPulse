from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    trade_id: str
    symbol: str
    action: str  # 'buy' or 'sell'
    shares: float
    price: float
    total_value: float
    executed_at: str
    success: bool
    error: str = None

class Broker(ABC):
    @abstractmethod
    def get_price(self, symbol: str) -> float | None:
        pass

    @abstractmethod
    def execute_buy(self, symbol: str, shares: float) -> TradeResult:
        pass

    @abstractmethod
    def execute_sell(self, symbol: str, shares: float) -> TradeResult:
        pass

class PaperBroker(Broker):
    """Paper trading broker backed by yfinance prices and SQLite storage."""

    def __init__(self, portfolio_id: str):
        self.portfolio_id = portfolio_id

    def get_price(self, symbol: str) -> float | None:
        from src.investor.market_data import get_current_price
        return get_current_price(symbol)

    def execute_buy(self, symbol: str, shares: float) -> TradeResult:
        """Execute a paper buy. Deducts cash, adds/updates holding."""
        price = self.get_price(symbol)
        if price is None:
            return TradeResult(
                trade_id=str(uuid.uuid4()), symbol=symbol, action='buy',
                shares=shares, price=0, total_value=0,
                executed_at=datetime.utcnow().isoformat(), success=False,
                error=f"Could not get price for {symbol}"
            )

        total_value = price * shares
        trade_id = str(uuid.uuid4())

        from src.storage import db
        # Update portfolio cash
        portfolio = db.get_portfolio(self.portfolio_id)
        if portfolio is None:
            return TradeResult(trade_id=trade_id, symbol=symbol, action='buy',
                             shares=shares, price=price, total_value=total_value,
                             executed_at=datetime.utcnow().isoformat(), success=False,
                             error="Portfolio not found")

        if portfolio['current_cash'] < total_value:
            return TradeResult(trade_id=trade_id, symbol=symbol, action='buy',
                             shares=shares, price=price, total_value=total_value,
                             executed_at=datetime.utcnow().isoformat(), success=False,
                             error="Insufficient cash")

        new_cash = portfolio['current_cash'] - total_value
        db.update_portfolio(self.portfolio_id, current_cash=new_cash)

        # Update holding (compute new average cost basis if already holding)
        existing = [h for h in db.get_holdings(self.portfolio_id) if h['symbol'] == symbol]
        if existing:
            old = existing[0]
            new_shares = old['shares'] + shares
            new_avg_cost = (old['avg_cost_basis'] * old['shares'] + price * shares) / new_shares
            db.upsert_holding(self.portfolio_id, symbol, new_shares, new_avg_cost,
                            old['asset_type'], old.get('company_name'), old.get('sector'),
                            old.get('geography'))
        else:
            # Determine asset type from ETF universe
            etfs = {e['symbol'] for e in db.get_etf_universe()}
            asset_type = 'etf' if symbol in etfs else 'stock'
            db.upsert_holding(self.portfolio_id, symbol, shares, price, asset_type)

        now = datetime.utcnow().isoformat()
        db.save_trade({
            'trade_id': trade_id, 'portfolio_id': self.portfolio_id,
            'symbol': symbol, 'action': 'buy', 'shares': shares,
            'price': price, 'total_value': total_value,
            'status': 'executed', 'proposed_at': now, 'executed_at': now,
        })

        logger.info("Paper BUY: %s x%.1f @ $%.2f = $%.2f", symbol, shares, price, total_value)
        return TradeResult(trade_id=trade_id, symbol=symbol, action='buy',
                         shares=shares, price=price, total_value=total_value,
                         executed_at=now, success=True)

    def execute_sell(self, symbol: str, shares: float) -> TradeResult:
        """Execute a paper sell. Adds cash, removes/reduces holding."""
        price = self.get_price(symbol)
        if price is None:
            return TradeResult(
                trade_id=str(uuid.uuid4()), symbol=symbol, action='sell',
                shares=shares, price=0, total_value=0,
                executed_at=datetime.utcnow().isoformat(), success=False,
                error=f"Could not get price for {symbol}")

        total_value = price * shares
        trade_id = str(uuid.uuid4())

        from src.storage import db
        existing = [h for h in db.get_holdings(self.portfolio_id) if h['symbol'] == symbol]
        if not existing:
            return TradeResult(trade_id=trade_id, symbol=symbol, action='sell',
                             shares=shares, price=price, total_value=total_value,
                             executed_at=datetime.utcnow().isoformat(), success=False,
                             error=f"No holding found for {symbol}")

        holding = existing[0]
        if holding['shares'] < shares:
            shares = holding['shares']  # sell what we have
            total_value = price * shares

        # Update cash
        portfolio = db.get_portfolio(self.portfolio_id)
        new_cash = portfolio['current_cash'] + total_value
        db.update_portfolio(self.portfolio_id, current_cash=new_cash)

        # Update or remove holding
        remaining = holding['shares'] - shares
        if remaining <= 0.001:  # effectively zero
            db.delete_holding(self.portfolio_id, symbol)
        else:
            db.upsert_holding(self.portfolio_id, symbol, remaining, holding['avg_cost_basis'],
                            holding['asset_type'], holding.get('company_name'),
                            holding.get('sector'), holding.get('geography'))

        now = datetime.utcnow().isoformat()
        db.save_trade({
            'trade_id': trade_id, 'portfolio_id': self.portfolio_id,
            'symbol': symbol, 'action': 'sell', 'shares': shares,
            'price': price, 'total_value': total_value,
            'status': 'executed', 'proposed_at': now, 'executed_at': now,
        })

        logger.info("Paper SELL: %s x%.1f @ $%.2f = $%.2f", symbol, shares, price, total_value)
        return TradeResult(trade_id=trade_id, symbol=symbol, action='sell',
                         shares=shares, price=price, total_value=total_value,
                         executed_at=now, success=True)
