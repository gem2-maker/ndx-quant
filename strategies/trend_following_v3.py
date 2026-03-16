"""Trend Following V3 - Improved based on V1 lessons.

Key improvements over V1:
1. Removed volume filter (QQQ has consistent volume, not needed for indices)
2. Removed pullback filter (too restrictive, misses strong breakouts)
3. Added 63-day momentum as an entry confirmation filter
4. Lowered ADX thresholds slightly (entry >= 21, exit <= 17) for more entries
5. Optional monthly rebalance for entries (reduce overtrading, monthly check only)
6. Simplified code by removing unused tracking

Core remains: SMA50/200 crossover + ADX trend strength + trailing stop.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import timedelta

from strategies.base import BaseStrategy, Signal


@dataclass
class TrendPositionV3:
    ticker: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    highest_price: float = 0.0
    commission: float = 0.0
    bars_held: int = 0

    def pnl_pct(self, price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (price - self.entry_price) / self.entry_price


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Vectorized ADX computation."""
    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder's smoothing
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=alpha, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=low.index).ewm(alpha=alpha, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period).mean()
    return adx.fillna(0.0)


class TrendFollowingV3(BaseStrategy):
    """V3: SMA crossover with ADX filter + momentum confirmation + monthly rebalance entry.

    Entry conditions (ALL must be true):
    1. SMA50 crosses above SMA200 (golden cross) OR already above
    2. ADX >= entry_threshold (default 21) - trend strength sufficient
    3. 63-day momentum > 0 (price above its 63-day lag)
    4. If rebalance_monthly=True, only check on first trading day of month (or every 21 bars)

    Exit conditions (ANY triggers):
    1. Death cross: SMA50 crosses below SMA200
    2. ADX drops below exit_threshold (default 17) - trend weakening
    3. Trailing stop: price drops X% from highest since entry
    """

    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        adx_period: int = 14,
        adx_entry: float = 21.0,      # ADX must exceed this to enter
        adx_exit: float = 17.0,       # ADX below this triggers exit
        momentum_period: int = 63,    # 63-day momentum check
        trailing_stop_pct: float = 0.10,
        min_bars_between_trades: int = 10,
        rebalance_monthly: bool = True,  # Only enter on first bar of month
    ):
        super().__init__("TrendFollowingV3")
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.adx_period = adx_period
        self.adx_entry = adx_entry
        self.adx_exit = adx_exit
        self.momentum_period = momentum_period
        self.trailing_stop_pct = trailing_stop_pct
        self.min_bars = min_bars_between_trades
        self.rebalance_monthly = rebalance_monthly

        # Precomputed arrays (set on reset)
        self._sma_fast_arr: Optional[np.ndarray] = None
        self._sma_slow_arr: Optional[np.ndarray] = None
        self._adx_arr: Optional[np.ndarray] = None
        self._momentum_arr: Optional[np.ndarray] = None

        # State
        self._in_position = False
        self._last_trade_idx = -999
        self._last_rebalance_month: Optional[int] = None
        self._highest_since_entry = 0.0

    def reset(self):
        """Reset state. Precompute indicators for faster backtest."""
        self._in_position = False
        self._last_trade_idx = -999
        self._last_rebalance_month = None
        self._highest_since_entry = 0.0

    def _precompute(self, df: pd.DataFrame):
        """Precompute all needed arrays for O(1) access."""
        close = df["Close"].values
        high = df["High"].values if "High" in df.columns else close
        low = df["Low"].values if "Low" in df.columns else close

        # SMAs
        self._sma_fast_arr = pd.Series(close).rolling(window=self.sma_fast, min_periods=self.sma_fast).mean().values
        self._sma_slow_arr = pd.Series(close).rolling(window=self.sma_slow, min_periods=self.sma_slow).mean().values

        # ADX
        self._adx_arr = compute_adx(
            pd.Series(high), pd.Series(low), pd.Series(close), self.adx_period
        ).values

        # Momentum = (current price / price N periods ago) - 1
        mom = np.full_like(close, np.nan, dtype=float)
        for i in range(self.momentum_period, len(close)):
            mom[i] = close[i] / close[i - self.momentum_period] - 1.0
        self._momentum_arr = np.where(np.isnan(mom), 0.0, mom)

    def _is_rebalance_day(self, idx: int, date: pd.Timestamp) -> bool:
        """Check if this is a rebalance day (monthly)."""
        if not self.rebalance_monthly:
            return True
        current_month = (date.year, date.month)
        if self._last_rebalance_month != current_month:
            # First rebalance of the month, also ensure we're past min_bars
            if idx >= self.sma_slow and idx - self._last_trade_idx >= self.min_bars:
                return True
        return False

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        n = len(df)
        if idx >= n:
            return Signal.HOLD

        # Precompute on first bar where we have enough data
        if self._sma_fast_arr is None and idx >= self.sma_fast:
            self._precompute(df)

        # Not enough data yet
        required = max(self.sma_slow, self.momentum_period) + 5
        if idx < required:
            return Signal.HOLD

        # --- Get current values ---
        sma_fast = self._sma_fast_arr[idx]
        sma_slow = self._sma_slow_arr[idx]
        adx = self._adx_arr[idx]
        momentum = self._momentum_arr[idx]
        date = df.index[idx]

        # --- EXIT LOGIC (checked every bar, risk first) ---
        if self._in_position:
            # 1. Death cross
            if sma_fast < sma_slow:
                self._in_position = False
                self._last_trade_idx = idx
                return Signal.SELL
            # 2. ADX exit
            if adx < self.adx_exit:
                self._in_position = False
                self._last_trade_idx = idx
                return Signal.SELL
            return Signal.HOLD

        # --- ENTRY LOGIC ---
        # 1. Rebalance day check
        if not self._is_rebalance_day(idx, date):
            return Signal.HOLD

        # 2. Golden cross
        if sma_fast <= sma_slow:
            return Signal.HOLD

        # 3. ADX entry threshold
        if adx < self.adx_entry:
            return Signal.HOLD

        # 4. Momentum confirmation (price above N-day lag)
        if momentum <= 0:
            return Signal.HOLD

        # 5. Min bars between trades
        if idx - self._last_trade_idx < self.min_bars:
            return Signal.HOLD

        self._in_position = True
        self._last_trade_idx = idx
        self._last_rebalance_month = (date.year, date.month)
        self._highest_since_entry = 0.0
        return Signal.BUY


class TrendFollowingV3Engine:
    """Engine for V3 strategy."""

    def __init__(
        self,
        strategy: TrendFollowingV3,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_pct: float = 0.95,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_pct = position_pct

    def run(self, df: pd.DataFrame, ticker: str = "QQQ") -> dict:
        self.strategy.reset()

        cash = self.initial_capital
        position: Optional[TrendPositionV3] = None
        trades = []
        equity = []

        for i in range(len(df)):
            close = df["Close"].iloc[i]
            date = df.index[i]
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]

            # Update trailing stop
            if position and position.shares > 0:
                position.bars_held += 1
                position.highest_price = max(position.highest_price, high)
                if position.highest_price > 0:
                    drawdown = (low - position.highest_price) / position.highest_price
                    if drawdown <= -self.strategy.trailing_stop_pct:
                        # Sell at close with slippage
                        sell_price = close * (1 - self.slippage)
                        comm = position.shares * sell_price * self.commission
                        cash += position.shares * sell_price - comm
                        trades.append({
                            "date": date, "action": "SELL", "shares": position.shares,
                            "price": sell_price, "commission": comm,
                            "reason": "trailing_stop",
                            "pnl": f"{position.pnl_pct(close):+.1%}",
                        })
                        position = None
                        self.strategy._in_position = False
                        self.strategy._last_trade_idx = i

            # --- Get signal ---
            signal = self.strategy.generate_signal(df, i)

            # --- Execute trades ---
            if signal == Signal.BUY and position is None:
                investable = cash * self.position_pct
                buy_price = close * (1 + self.slippage)
                shares = int(investable / buy_price)
                if shares > 0:
                    comm = shares * buy_price * self.commission
                    cost = shares * buy_price + comm
                    if cost <= cash:
                        cash -= cost
                        position = TrendPositionV3(
                            ticker=ticker, shares=shares,
                            entry_price=buy_price, entry_date=date,
                            highest_price=close, commission=comm,
                        )
                        # Record trade with context
                        sma_fast = self.strategy._sma_fast_arr[i] if self.strategy._sma_fast_arr is not None else None
                        sma_slow = self.strategy._sma_slow_arr[i] if self.strategy._sma_slow_arr is not None else None
                        adx = self.strategy._adx_arr[i] if self.strategy._adx_arr is not None else None
                        mom = self.strategy._momentum_arr[i] if self.strategy._momentum_arr is not None else None
                        trades.append({
                            "date": date, "action": "BUY", "shares": shares,
                            "price": buy_price, "commission": comm,
                            "reason": "golden_cross",
                            "sma50": f"{sma_fast:.1f}" if sma_fast else "",
                            "sma200": f"{sma_slow:.1f}" if sma_slow else "",
                            "adx": f"{adx:.1f}" if adx else "",
                            "mom": f"{mom:.1%}" if mom else "",
                        })

            elif signal == Signal.SELL and position and position.shares > 0:
                sell_price = close * (1 - self.slippage)
                comm = position.shares * sell_price * self.commission
                cash += position.shares * sell_price - comm
                sma_fast = self.strategy._sma_fast_arr[i] if self.strategy._sma_fast_arr is not None else None
                sma_slow = self.strategy._sma_slow_arr[i] if self.strategy._sma_slow_arr is not None else None
                exit_reason = "death_cross" if sma_fast and sma_slow and sma_fast < sma_slow else "adx_weak"
                trades.append({
                    "date": date, "action": "SELL", "shares": position.shares,
                    "price": sell_price, "commission": comm,
                    "reason": exit_reason,
                    "pnl": f"{position.pnl_pct(close):+.1%}",
                })
                position = None

            # Record equity
            total_value = cash
            if position and position.shares > 0:
                total_value += position.shares * close
            equity.append(total_value)

        # Force close at end
        if position and position.shares > 0:
            close = df["Close"].iloc[-1]
            date = df.index[-1]
            sell_price = close * (1 - self.slippage)
            comm = position.shares * sell_price * self.commission
            cash += position.shares * sell_price - comm
            trades.append({
                "date": date, "action": "SELL", "shares": position.shares,
                "price": sell_price, "commission": comm,
                "reason": "end_of_data",
                "pnl": f"{position.pnl_pct(close):+.1%}",
            })

        equity_series = pd.Series(equity, index=df.index[:len(equity)])
        return {
            "trades": trades,
            "equity_curve": equity_series,
            "metrics": self._calc_metrics(equity_series, trades),
        }

    def _calc_metrics(self, equity: pd.Series, trades: list) -> dict:
        if equity.empty:
            return {}

        daily_returns = equity.pct_change().dropna()
        downside = daily_returns[daily_returns < 0]
        periods = max(len(equity) - 1, 1)
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / periods) - 1 if equity.iloc[0] > 0 else 0.0
        ann_vol = daily_returns.std() * (252 ** 0.5) if not daily_returns.empty else 0.0
        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0.0
        sortino = (daily_returns.mean() / downside.std()) * (252 ** 0.5) if not downside.empty and downside.std() > 0 else 0.0
        max_dd = float(((equity / equity.cummax()) - 1).min())
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

        buys = [t for t in trades if t["action"] == "BUY"]
        sells = [t for t in trades if t["action"] == "SELL"]
        total_comm = sum(t["commission"] for t in trades)

        # Compute round-trips win/loss
        round_trips = []
        for i in range(min(len(buys), len(sells))):
            b, s = buys[i], sells[i]
            pnl_pct = (s["price"] - b["price"]) / b["price"]
            round_trips.append(pnl_pct)
        wins = [p for p in round_trips if p > 0]
        losses = [p for p in round_trips if p < 0]
        win_rate = len(wins) / len(round_trips) if round_trips else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf") if wins else 0.0

        # Time in market
        bars_in_market = 0
        if not equity.empty:
            cash_eq = self.initial_capital
            for i in range(len(equity)):
                if equity.iloc[i] > cash_eq:
                    bars_in_market += 1
        exposure_ratio = bars_in_market / len(equity) if len(equity) > 0 else 0.0

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
            "total_buys": len(buys),
            "total_sells": len(sells),
            "round_trips": len(round_trips),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "exposure_ratio": float(exposure_ratio),
            "total_commission": float(total_comm),
        }

    def summary(self, result: dict, ticker: str = "QQQ") -> str:
        m = result["metrics"]
        eq = result["equity_curve"]
        trades = result["trades"]

        exit_reasons = {}
        for t in trades:
            if t["action"] == "SELL":
                reason = t.get("reason", "unknown")
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        exit_summary = " | ".join(f"{k}: {v}" for k, v in exit_reasons.items())

        return f"""
{'='*60}
Trend Following V3 (SMA50/200 + ADX + 63d Momentum + Monthly Entry) on {ticker}
{'='*60}
Period: {eq.index[0].strftime('%Y-%m-%d')} -> {eq.index[-1].strftime('%Y-%m-%d')}
Initial: ${m['initial_equity']:,.0f} -> Final: ${m['final_equity']:,.0f}
Total Return: {m['total_return']:+.2%}
Annualized: {m['annualized_return']:+.2%} | Vol: {m['annualized_volatility']:.2%}
Sharpe: {m['sharpe_ratio']:.2f} | Sortino: {m['sortino_ratio']:.2f}
Max Drawdown: {m['max_drawdown']:.2%} | Calmar: {m['calmar_ratio']:.2f}
Exposure Ratio: {m['exposure_ratio']:.1%}
{'─'*60}
Round Trips: {m['round_trips']} | Win Rate: {m['win_rate']:.1%}
Avg Win: {m['avg_win']:+.2%} | Avg Loss: {m['avg_loss']:+.2%}
Profit Factor: {m['profit_factor']:.2f}
Exits: {exit_summary}
Commission: ${m['total_commission']:,.0f}
{'='*60}
"""
