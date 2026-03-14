"""Base strategy class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Position:
    ticker: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


@dataclass
class Trade:
    ticker: str
    date: pd.Timestamp
    action: str  # "BUY" or "SELL"
    shares: int
    price: float
    commission: float
    reason: str = ""


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        sells = [t for t in self.trades if t.action == "SELL"]
        if not sells:
            return 0.0
        # Simplified: count sells with positive PnL (from trade records)
        wins = sum(1 for t in sells if "profit" in t.reason.lower())
        return wins / len(sells)


class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        """Generate a trading signal for the given row index."""
        pass

    def get_stop_loss(self, entry_price: float) -> float:
        """Override for custom stop loss logic."""
        from config import STOP_LOSS_PCT
        return entry_price * (1 - STOP_LOSS_PCT)

    def get_take_profit(self, entry_price: float) -> float:
        """Override for custom take profit logic."""
        from config import TAKE_PROFIT_PCT
        return entry_price * (1 + TAKE_PROFIT_PCT)

    def __repr__(self) -> str:
        return f"<{self.name}>"
