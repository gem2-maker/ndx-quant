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
    entry_commission: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    bars_held: int = 0
    highest_price: float = 0.0
    lowest_price: float = 0.0

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
class CompletedTrade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    entry_commission: float
    exit_commission: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    hold_period_bars: int
    hold_period_days: int
    max_favorable_excursion_pct: float
    max_adverse_excursion_pct: float
    exit_reason: str = ""

    @property
    def total_commission(self) -> float:
        return self.entry_commission + self.exit_commission

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    completed_trades: list[CompletedTrade] = field(default_factory=list)
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
        completed = self.completed_trades
        if not completed:
            return 0.0
        wins = sum(1 for trade in completed if trade.is_win)
        return wins / len(completed)

    @property
    def metrics(self) -> dict[str, float]:
        eq = self.equity_curve
        if eq.empty:
            return {
                "initial_equity": 0.0,
                "final_equity": 0.0,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "annualized_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "calmar_ratio": 0.0,
                "buy_trades": 0,
                "sell_trades": 0,
                "completed_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_profit": 0.0,
                "average_trade_pnl": 0.0,
                "average_trade_return": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "expectancy": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "average_hold_bars": 0.0,
                "average_hold_days": 0.0,
                "total_commission": 0.0,
                "exposure_ratio": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
            }

        daily_returns = eq.pct_change().dropna()
        downside_returns = daily_returns[daily_returns < 0]
        periods = max(len(eq) - 1, 1)
        total_return = self.total_return
        annualized_return = (eq.iloc[-1] / eq.iloc[0]) ** (252 / periods) - 1 if eq.iloc[0] > 0 else 0.0
        annualized_volatility = daily_returns.std() * (252 ** 0.5) if not daily_returns.empty else 0.0
        sharpe_ratio = 0.0
        if not daily_returns.empty and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        sortino_ratio = 0.0
        if not downside_returns.empty and downside_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() / downside_returns.std()) * (252 ** 0.5)
        max_drawdown = float(((eq / eq.cummax()) - 1).min())
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        buys = sum(1 for trade in self.trades if trade.action == "BUY")
        sells = sum(1 for trade in self.trades if trade.action == "SELL")
        completed = self.completed_trades
        commissions = float(sum(trade.commission for trade in self.trades))

        gross_profit = float(sum(trade.net_pnl for trade in completed if trade.net_pnl > 0))
        gross_loss = float(sum(trade.net_pnl for trade in completed if trade.net_pnl < 0))
        profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else (float("inf") if gross_profit > 0 else 0.0)
        average_trade_pnl = float(sum(trade.net_pnl for trade in completed) / len(completed)) if completed else 0.0
        average_trade_return = float(sum(trade.return_pct for trade in completed) / len(completed)) if completed else 0.0
        wins = [trade.net_pnl for trade in completed if trade.net_pnl > 0]
        losses = [trade.net_pnl for trade in completed if trade.net_pnl < 0]
        average_win = float(sum(wins) / len(wins)) if wins else 0.0
        average_loss = float(sum(losses) / len(losses)) if losses else 0.0
        expectancy = (self.win_rate * average_win) + ((1 - self.win_rate) * average_loss) if completed else 0.0
        best_trade = float(max((trade.net_pnl for trade in completed), default=0.0))
        worst_trade = float(min((trade.net_pnl for trade in completed), default=0.0))
        average_hold_bars = float(sum(trade.hold_period_bars for trade in completed) / len(completed)) if completed else 0.0
        average_hold_days = float(sum(trade.hold_period_days for trade in completed) / len(completed)) if completed else 0.0
        exposure_bars = sum(trade.hold_period_bars for trade in completed)
        exposure_ratio = exposure_bars / len(eq) if len(eq) > 0 else 0.0

        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        for trade in completed:
            if trade.is_win:
                current_wins += 1
                current_losses = 0
            elif trade.net_pnl < 0:
                current_losses += 1
                current_wins = 0
            else:
                current_wins = 0
                current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
            max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return {
            "initial_equity": float(eq.iloc[0]),
            "final_equity": float(eq.iloc[-1]),
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_volatility),
            "sharpe_ratio": float(sharpe_ratio),
            "sortino_ratio": float(sortino_ratio),
            "max_drawdown": max_drawdown,
            "calmar_ratio": float(calmar_ratio),
            "buy_trades": buys,
            "sell_trades": sells,
            "completed_trades": len(completed),
            "win_rate": float(self.win_rate),
            "profit_factor": float(profit_factor),
            "gross_profit": gross_profit,
            "gross_loss": float(gross_loss),
            "net_profit": float(sum(trade.net_pnl for trade in completed)),
            "average_trade_pnl": average_trade_pnl,
            "average_trade_return": average_trade_return,
            "average_win": average_win,
            "average_loss": average_loss,
            "expectancy": float(expectancy),
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "average_hold_bars": average_hold_bars,
            "average_hold_days": average_hold_days,
            "total_commission": commissions,
            "exposure_ratio": float(exposure_ratio),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
        }


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
