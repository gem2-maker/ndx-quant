"""Livermore Trend-Following Strategy.

Based on Jesse Livermore's trading principles:
1. Ride the trend — don't predict, follow
2. Pyramiding — add to winners at key breakout points
3. 0.618 Fibonacci retracement — pullback entry in established trends
4. Trailing stop — exit when price drops 6-7% from peak (CTA stop-loss zone)
5. Key pivot points — identify support/resistance for entries
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from strategies.base import BaseStrategy, Signal


@dataclass
class LivermorePosition:
    """A position that supports pyramiding (multiple entries)."""
    ticker: str
    layers: list[dict] = field(default_factory=list)  # [{shares, price, date, commission}]
    highest_price: float = 0.0
    stop_loss: float = 0.0
    bars_held: int = 0

    @property
    def total_shares(self) -> int:
        return sum(l["shares"] for l in self.layers)

    @property
    def avg_entry_price(self) -> float:
        total_cost = sum(l["shares"] * l["price"] for l in self.layers)
        total_shares = self.total_shares
        return total_cost / total_shares if total_shares > 0 else 0.0

    @property
    def total_commission(self) -> float:
        return sum(l["commission"] for l in self.layers)

    def market_value(self, price: float) -> float:
        return self.total_shares * price

    def pnl_pct(self, price: float) -> float:
        avg = self.avg_entry_price
        return (price - avg) / avg if avg > 0 else 0.0

    def add_layer(self, shares: int, price: float, date: pd.Timestamp, commission: float):
        self.layers.append({
            "shares": shares,
            "price": price,
            "date": date,
            "commission": commission,
        })


class LivermoreStrategy(BaseStrategy):
    """
    Livermore Trend-Following Strategy.

    Core rules:
    - Trend identified by higher highs/higher lows (or lower highs/lower lows)
    - Entry at 0.618 Fibonacci retracement of the current swing
    - Add to winners (pyramid) on breakout above prior swing high
    - Stop loss at 6-7% trailing from the highest price
    - Only trade in direction of the primary trend
    """

    def __init__(
        self,
        lookback: int = 120,           # Bars to identify trend structure
        fib_level: float = 0.618,      # Fibonacci retracement level for entry
        stop_loss_pct: float = 0.065,  # 6.5% trailing stop from peak
        max_pyramids: int = 3,         # Max number of pyramid additions
        atr_period: int = 14,          # ATR for volatility context
        trend_sma: int = 200,          # SMA for primary trend filter
    ):
        super().__init__("Livermore")
        self.lookback = lookback
        self.fib_level = fib_level
        self.stop_loss_pct = stop_loss_pct
        self.max_pyramids = max_pyramids
        self.atr_period = atr_period
        self.trend_sma = trend_sma

    def _calc_atr(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate ATR at a given index."""
        start = max(0, idx - self.atr_period + 1)
        high = df["High"].iloc[start:idx+1]
        low = df["Low"].iloc[start:idx+1]
        close = df["Close"].iloc[start:idx+1]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.mean()

    def _find_swing_high(self, df: pd.DataFrame, idx: int, lookback: int) -> tuple[float, int]:
        """Find the highest high and its index within the lookback window."""
        start = max(0, idx - lookback)
        highs = df["High"].iloc[start:idx+1]
        max_idx = highs.idxmax()
        return float(df["High"].loc[max_idx]), df.index.get_loc(max_idx)

    def _find_swing_low(self, df: pd.DataFrame, idx: int, lookback: int) -> tuple[float, int]:
        """Find the lowest low and its index within the lookback window."""
        start = max(0, idx - lookback)
        lows = df["Low"].iloc[start:idx+1]
        min_idx = lows.idxmin()
        return float(df["Low"].loc[min_idx]), df.index.get_loc(min_idx)

    def _is_uptrend(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if in primary uptrend (price above trend SMA)."""
        if idx < self.trend_sma:
            return False
        sma_val = df["Close"].iloc[max(0, idx-self.trend_sma):idx+1].mean()
        return df["Close"].iloc[idx] > sma_val

    def _calc_fibonacci_levels(self, swing_high: float, swing_low: float) -> dict:
        """Calculate Fibonacci retracement levels."""
        diff = swing_high - swing_low
        return {
            "0.0": swing_low,
            "0.236": swing_low + 0.236 * diff,
            "0.382": swing_low + 0.382 * diff,
            "0.5": swing_low + 0.5 * diff,
            "0.618": swing_low + 0.618 * diff,
            "0.786": swing_low + 0.786 * diff,
            "1.0": swing_high,
        }

    def get_stop_loss(self, entry_price: float) -> float:
        """Override base — trailing stop calculated dynamically, not at entry."""
        return 0.0  # Handled in engine

    def get_take_profit(self, entry_price: float) -> float:
        """No fixed take profit — let winners run."""
        return 0.0

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        """
        Generate signal based on Livermore rules:
        - BUY: Uptrend + price at 0.618 Fib retracement + bounce confirmation
        - SELL: Price drops stop_loss_pct from peak
        - HOLD: Everything else
        """
        if idx < max(self.lookback, self.trend_sma) + 1:
            return Signal.HOLD

        close = df["Close"].iloc[idx]
        prev_close = df["Close"].iloc[idx - 1]

        # Only trade in direction of primary trend
        if not self._is_uptrend(df, idx):
            return Signal.SELL if prev_close > close else Signal.HOLD

        # Find recent swing high and low
        swing_high, high_idx = self._find_swing_high(df, idx, self.lookback)
        swing_low, low_idx = self._find_swing_low(df, idx, self.lookback)

        # Ensure swing low comes before swing high (uptrend structure)
        if low_idx >= high_idx:
            return Signal.HOLD

        # Calculate Fibonacci levels
        fib = self._calc_fibonacci_levels(swing_high, swing_low)

        # BUY signal: price pulled back to 0.618 area and is bouncing
        # (previous bar was near or below 0.618, current bar is rising)
        fib_entry = fib[f"{self.fib_level}"]
        prev_at_fib = prev_close <= fib_entry * 1.01  # Within 1% of fib level
        bouncing = close > prev_close  # Bounce confirmation

        if prev_at_fib and bouncing:
            return Signal.BUY

        return Signal.HOLD


class LivermoreEngine:
    """
    Backtest engine with pyramiding support for Livermore strategy.

    Key differences from standard engine:
    - Supports adding to winning positions (pyramiding)
    - Dynamic trailing stop based on highest price
    - No fixed take profit
    """

    def __init__(
        self,
        strategy: LivermoreStrategy,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position_pct: float = 0.25,  # Allow larger positions for pyramiding
        pyramid_profit_trigger: float = 0.05,  # Add after 5% profit
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        self.pyramid_profit_trigger = pyramid_profit_trigger

    def run(self, df: pd.DataFrame, ticker: str = "QQQ") -> dict:
        """Run backtest with Livermore rules."""
        cash = self.initial_capital
        position: LivermorePosition | None = None
        trades = []
        completed_trades = []
        equity = []
        pyramid_count = 0

        for i in range(len(df)):
            close = df["Close"].iloc[i]
            date = df.index[i]
            high = df["High"].iloc[i]

            # === Update position ===
            if position:
                position.bars_held += 1
                position.highest_price = max(position.highest_price, high)

                # Dynamic trailing stop
                trailing_stop = position.highest_price * (1 - self.strategy.stop_loss_pct)

                # === Check trailing stop ===
                if close <= trailing_stop:
                    sell_price = close * (1 - self.slippage)
                    total_shares = position.total_shares
                    comm = total_shares * sell_price * self.commission
                    cash += total_shares * sell_price - comm

                    # Calculate PnL
                    total_cost = sum(l["shares"] * l["price"] for l in position.layers)
                    total_comm = position.total_commission + comm
                    net_pnl = (total_shares * sell_price - comm) - total_cost - position.total_commission
                    return_pct = (sell_price - position.avg_entry_price) / position.avg_entry_price

                    trades.append({
                        "date": date, "action": "SELL", "shares": total_shares,
                        "price": sell_price, "commission": comm, "reason": "trailing_stop",
                        "layers": len(position.layers),
                    })
                    completed_trades.append({
                        "entry_date": position.layers[0]["date"],
                        "exit_date": date,
                        "avg_entry": position.avg_entry_price,
                        "exit_price": sell_price,
                        "shares": total_shares,
                        "net_pnl": net_pnl,
                        "return_pct": return_pct,
                        "hold_days": (date - position.layers[0]["date"]).days,
                        "layers": len(position.layers),
                        "max_price": position.highest_price,
                        "exit_reason": "trailing_stop",
                    })
                    position = None
                    pyramid_count = 0

                # === Check pyramid opportunity ===
                elif (position.total_shares > 0
                      and pyramid_count < self.strategy.max_pyramids
                      and position.pnl_pct(close) > self.pyramid_profit_trigger):

                    # Add to winner — use same sizing as initial
                    max_value = cash * self.max_position_pct
                    buy_price = close * (1 + self.slippage)
                    add_shares = int(max_value / buy_price)
                    if add_shares > 0:
                        comm = add_shares * buy_price * self.commission
                        cost = add_shares * buy_price + comm
                        if cost <= cash:
                            cash -= cost
                            position.add_layer(add_shares, buy_price, date, comm)
                            pyramid_count += 1
                            # Raise trailing stop on pyramid
                            position.stop_loss = close * (1 - self.strategy.stop_loss_pct)
                            trades.append({
                                "date": date, "action": "PYRAMID", "shares": add_shares,
                                "price": buy_price, "commission": comm,
                                "reason": f"pyramid_{pyramid_count}",
                                "avg_entry": position.avg_entry_price,
                                "total_shares": position.total_shares,
                            })

            # === Check signal ===
            signal = self.strategy.generate_signal(df, i)

            if signal == Signal.BUY and position is None:
                max_value = cash * self.max_position_pct
                buy_price = close * (1 + self.slippage)
                shares = int(max_value / buy_price)
                if shares > 0:
                    comm = shares * buy_price * self.commission
                    cost = shares * buy_price + comm
                    if cost <= cash:
                        cash -= cost
                        position = LivermorePosition(ticker=ticker)
                        position.add_layer(shares, buy_price, date, comm)
                        position.highest_price = close
                        position.stop_loss = close * (1 - self.strategy.stop_loss_pct)
                        pyramid_count = 0
                        trades.append({
                            "date": date, "action": "BUY", "shares": shares,
                            "price": buy_price, "commission": comm, "reason": "fib_entry",
                        })

            # Record equity
            total_value = cash
            if position:
                total_value += position.market_value(close)
            equity.append(total_value)

        # === Close open position at end ===
        if position:
            close = df["Close"].iloc[-1]
            date = df.index[-1]
            sell_price = close * (1 - self.slippage)
            total_shares = position.total_shares
            comm = total_shares * sell_price * self.commission
            cash += total_shares * sell_price - comm

            total_cost = sum(l["shares"] * l["price"] for l in position.layers)
            net_pnl = (total_shares * sell_price - comm) - total_cost - position.total_commission
            return_pct = (sell_price - position.avg_entry_price) / position.avg_entry_price

            trades.append({
                "date": date, "action": "SELL", "shares": total_shares,
                "price": sell_price, "commission": comm, "reason": "end_of_data",
            })
            completed_trades.append({
                "entry_date": position.layers[0]["date"],
                "exit_date": date,
                "avg_entry": position.avg_entry_price,
                "exit_price": sell_price,
                "shares": total_shares,
                "net_pnl": net_pnl,
                "return_pct": return_pct,
                "hold_days": (date - position.layers[0]["date"]).days,
                "layers": len(position.layers),
                "max_price": position.highest_price,
                "exit_reason": "end_of_data",
            })

        equity_series = pd.Series(equity, index=df.index[:len(equity)])
        return {
            "trades": trades,
            "completed_trades": completed_trades,
            "equity_curve": equity_series,
            "metrics": self._calc_metrics(equity_series, trades, completed_trades),
        }

    def _calc_metrics(self, equity: pd.Series, trades: list, completed: list) -> dict:
        """Calculate performance metrics."""
        if equity.empty:
            return {}

        daily_returns = equity.pct_change().dropna()
        downside = daily_returns[daily_returns < 0]
        periods = max(len(equity) - 1, 1)
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / periods) - 1
        ann_vol = daily_returns.std() * (252 ** 0.5) if not daily_returns.empty else 0.0
        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0.0
        sortino = (daily_returns.mean() / downside.std()) * (252 ** 0.5) if not downside.empty and downside.std() > 0 else 0.0
        max_dd = float(((equity / equity.cummax()) - 1).min())
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

        wins = [t for t in completed if t["net_pnl"] > 0]
        losses = [t for t in completed if t["net_pnl"] <= 0]
        win_rate = len(wins) / len(completed) if completed else 0.0
        gross_profit = sum(t["net_pnl"] for t in wins)
        gross_loss = abs(sum(t["net_pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        total_comm = sum(t["commission"] for t in trades)

        # Pyramiding stats
        pyramid_trades = [t for t in trades if t["action"] == "PYRAMID"]
        multi_layer = [t for t in completed if t["layers"] > 1]

        return {
            "initial_equity": self.initial_capital,
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(total_return),
            "annualized_return": float(ann_return),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": max_dd,
            "calmar_ratio": float(calmar),
            "completed_trades": len(completed),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "gross_profit": float(gross_profit),
            "gross_loss": float(-sum(t["net_pnl"] for t in losses)),
            "net_profit": float(sum(t["net_pnl"] for t in completed)),
            "average_trade_return": float(sum(t["return_pct"] for t in completed) / len(completed)) if completed else 0.0,
            "average_hold_days": float(sum(t["hold_days"] for t in completed) / len(completed)) if completed else 0.0,
            "best_trade": float(max((t["return_pct"] for t in completed), default=0.0)),
            "worst_trade": float(min((t["return_pct"] for t in completed), default=0.0)),
            "total_commission": float(total_comm),
            "exposure_ratio": float(sum(t["hold_days"] for t in completed) / len(equity)) if completed else 0.0,
            "pyramid_adds": len(pyramid_trades),
            "multi_layer_trades": len(multi_layer),
            "avg_layers": float(sum(t["layers"] for t in completed) / len(completed)) if completed else 0.0,
        }

    def summary(self, result: dict, ticker: str = "QQQ") -> str:
        """Generate human-readable summary."""
        m = result["metrics"]
        eq = result["equity_curve"]

        return f"""
{'='*55}
Livermore Trend Strategy on {ticker}
{'='*55}
Period: {eq.index[0].strftime('%Y-%m-%d')} → {eq.index[-1].strftime('%Y-%m-%d')}
Initial Capital: ${m['initial_equity']:,.2f}
Final Equity: ${m['final_equity']:,.2f}
Total Return: {m['total_return']:+.2%}
Annualized Return: {m['annualized_return']:+.2%}
Sharpe Ratio: {m['sharpe_ratio']:.2f}
Sortino Ratio: {m['sortino_ratio']:.2f}
Max Drawdown: {m['max_drawdown']:.2%}
Calmar Ratio: {m['calmar_ratio']:.2f}
{'─'*55}
Completed Trades: {m['completed_trades']}
Win Rate: {m['win_rate']:.2%}
Profit Factor: {m['profit_factor']:.2f}
Net Profit: ${m['net_profit']:,.2f}
Best Trade: {m['best_trade']:+.2%}
Worst Trade: {m['worst_trade']:+.2%}
Avg Return/Trade: {m['average_trade_return']:+.2%}
Avg Hold: {m['average_hold_days']:.0f} days
{'─'*55}
Pyramid Adds: {m['pyramid_adds']}
Multi-Layer Trades: {m['multi_layer_trades']}
Avg Layers/Trade: {m['avg_layers']:.1f}
{'─'*55}
Total Commission: ${m['total_commission']:,.2f}
Exposure: {m['exposure_ratio']:.2%}
{'='*55}
"""
