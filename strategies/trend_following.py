"""Long-Term Trend Following Strategy.

A buy-and-hold style strategy that enters on golden cross (SMA50 > SMA200),
filters by ADX trend strength, confirms with volume, and waits for pullback
entries. Holds until death cross or trailing stop exits.

Designed for: low-frequency trading, most of the time in cash or fully invested.
Typical holding period: months to years.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from strategies.base import BaseStrategy, Signal


@dataclass
class TrendPosition:
    """Tracks a long-term trend position with trailing stop."""
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


class TrendFollowingStrategy(BaseStrategy):
    """Long-term trend following with SMA crossover + ADX filter.

    Entry conditions (ALL must be true):
    1. Golden cross: SMA50 crosses above SMA200 (or already above)
    2. ADX > threshold (default 25): trend is strong enough
    3. Volume confirmation: current volume > average volume * factor
    4. Pullback entry: price near SMA50 (within tolerance) OR just crossed

    Exit conditions (ANY triggers):
    1. Death cross: SMA50 crosses below SMA200
    2. ADX drops below exit threshold: trend weakening
    3. Trailing stop: price drops X% from highest since entry
    """

    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 25.0,       # Minimum ADX to enter
        adx_exit_threshold: float = 20.0,  # ADX below this = exit
        volume_factor: float = 1.0,        # Volume must be >= avg * factor
        volume_lookback: int = 20,         # Volume average window
        pullback_pct: float = 0.03,        # Price within 3% of SMA50 = pullback
        trailing_stop_pct: float = 0.12,   # 12% trailing stop from peak
        min_bars_between_trades: int = 10, # Prevent whipsaw
    ):
        super().__init__("TrendFollowing")
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.adx_exit_threshold = adx_exit_threshold
        self.volume_factor = volume_factor
        self.volume_lookback = volume_lookback
        self.pullback_pct = pullback_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.min_bars_between_trades = min_bars_between_trades

        # State tracking (reset per backtest run)
        self._last_trade_bar: int = -999
        self._in_position: bool = False

    def reset(self):
        """Reset state for a new backtest run."""
        self._last_trade_bar = -999
        self._in_position = False

    def _compute_sma(self, df: pd.DataFrame, idx: int, period: int) -> Optional[float]:
        """Compute SMA at given index."""
        if idx < period - 1:
            return None
        return float(df["Close"].iloc[idx - period + 1:idx + 1].mean())

    def _compute_adx(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """Compute ADX (Average Directional Index) at given index."""
        period = self.adx_period
        if idx < period * 2:
            return None

        high = df["High"].iloc[:idx + 1]
        low = df["Low"].iloc[:idx + 1]
        close = df["Close"].iloc[:idx + 1]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # Smooth with Wilder's method (equivalent to EMA with alpha=1/period)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
            alpha=1/period, min_periods=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
            alpha=1/period, min_periods=period).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, min_periods=period).mean()

        val = adx.iloc[-1]
        return float(val) if not pd.isna(val) else None

    def _sma_fast(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        return self._compute_sma(df, idx, self.sma_fast)

    def _sma_slow(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        return self._compute_sma(df, idx, self.sma_slow)

    def _volume_ok(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if current volume confirms the trend."""
        if "Volume" not in df.columns:
            return True  # No volume data = assume OK
        if idx < self.volume_lookback:
            return True
        vol = df["Volume"].iloc[idx]
        avg_vol = df["Volume"].iloc[idx - self.volume_lookback + 1:idx + 1].mean()
        if avg_vol <= 0:
            return True
        return vol >= avg_vol * self.volume_factor

    def _is_pullback(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if price is near SMA50 (pullback zone)."""
        sma50 = self._sma_fast(df, idx)
        if sma50 is None:
            return False
        close = df["Close"].iloc[idx]
        deviation = abs(close - sma50) / sma50
        # Price within pullback_pct of SMA50, OR price just broke above SMA50
        if deviation <= self.pullback_pct:
            return True
        # Also allow entry when price just crossed above SMA50
        if idx >= 1:
            prev_close = df["Close"].iloc[idx - 1]
            prev_sma50 = self._sma_fast(df, idx - 1)
            if prev_sma50 and prev_close <= prev_sma50 and close > sma50:
                return True
        return False

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        if idx < self.sma_slow:
            return Signal.HOLD

        sma50 = self._sma_fast(df, idx)
        sma200 = self._sma_slow(df, idx)

        if sma50 is None or sma200 is None:
            return Signal.HOLD

        close = df["Close"].iloc[idx]
        adx = self._compute_adx(df, idx)

        # === EXIT SIGNALS ===
        if self._in_position:
            # Death cross: SMA50 < SMA200
            if sma50 < sma200:
                self._in_position = False
                self._last_trade_bar = idx
                return Signal.SELL

            # ADX collapse: trend too weak
            if adx is not None and adx < self.adx_exit_threshold:
                self._in_position = False
                self._last_trade_bar = idx
                return Signal.SELL

            return Signal.HOLD

        # === ENTRY SIGNALS ===
        # 1. Golden cross: SMA50 > SMA200
        if sma50 <= sma200:
            return Signal.HOLD

        # 2. ADX threshold
        if adx is None or adx < self.adx_threshold:
            return Signal.HOLD

        # 3. Volume confirmation
        if not self._volume_ok(df, idx):
            return Signal.HOLD

        # 4. Pullback entry (or initial cross)
        if not self._is_pullback(df, idx):
            return Signal.HOLD

        # 5. Minimum bars between trades (whipsaw protection)
        if idx - self._last_trade_bar < self.min_bars_between_trades:
            return Signal.HOLD

        self._in_position = True
        self._last_trade_bar = idx
        return Signal.BUY


class TrendFollowingEngine:
    """Engine for the Trend Following strategy.

    Handles position sizing, trailing stops, and metrics.
    Unlike the generic BacktestEngine, this is tailored for
    long-term holding with minimal trading.
    """

    def __init__(
        self,
        strategy: TrendFollowingStrategy,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_pct: float = 0.95,  # Invest 95% when in market (buy-and-hold style)
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_pct = position_pct

    def run(self, df: pd.DataFrame, ticker: str = "QQQ") -> dict:
        self.strategy.reset()

        cash = self.initial_capital
        position: Optional[TrendPosition] = None
        trades = []
        equity = []

        for i in range(len(df)):
            close = df["Close"].iloc[i]
            date = df.index[i]
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]

            # === Update position & trailing stop ===
            if position and position.shares > 0:
                position.bars_held += 1
                position.highest_price = max(position.highest_price, high)

                # Trailing stop check
                if position.highest_price > 0:
                    drawdown = (low - position.highest_price) / position.highest_price
                    if drawdown <= -self.strategy.trailing_stop_pct:
                        # Trailing stop hit
                        sell_price = close * (1 - self.slippage)
                        comm = position.shares * sell_price * self.commission
                        cash += position.shares * sell_price - comm

                        trades.append({
                            "date": date, "action": "SELL", "shares": position.shares,
                            "price": sell_price, "commission": comm,
                            "reason": "trailing_stop",
                            "drawdown": f"{drawdown:.1%}",
                            "highest": f"${position.highest_price:.2f}",
                        })
                        position = None
                        self.strategy._in_position = False
                        self.strategy._last_trade_bar = i

            # === Signal ===
            signal = self.strategy.generate_signal(df, i)

            if signal == Signal.BUY and position is None:
                # Full position entry
                investable = cash * self.position_pct
                buy_price = close * (1 + self.slippage)
                shares = int(investable / buy_price)
                if shares > 0:
                    comm = shares * buy_price * self.commission
                    cost = shares * buy_price + comm
                    if cost <= cash:
                        cash -= cost
                        position = TrendPosition(
                            ticker=ticker, shares=shares,
                            entry_price=buy_price, entry_date=date,
                            highest_price=close, commission=comm,
                        )
                        # Record ADX for context
                        adx = self.strategy._compute_adx(df, i)
                        sma50 = self.strategy._sma_fast(df, i)
                        sma200 = self.strategy._sma_slow(df, i)
                        trades.append({
                            "date": date, "action": "BUY", "shares": shares,
                            "price": buy_price, "commission": comm,
                            "reason": "golden_cross",
                            "adx": f"{adx:.1f}" if adx else "N/A",
                            "sma50": f"${sma50:.2f}" if sma50 else "N/A",
                            "sma200": f"${sma200:.2f}" if sma200 else "N/A",
                        })

            elif signal == Signal.SELL and position and position.shares > 0:
                sell_price = close * (1 - self.slippage)
                comm = position.shares * sell_price * self.commission
                cash += position.shares * sell_price - comm

                # Determine exit reason
                sma50 = self.strategy._sma_fast(df, i)
                sma200 = self.strategy._sma_slow(df, i)
                if sma50 and sma200 and sma50 < sma200:
                    reason = "death_cross"
                else:
                    reason = "adx_weak"

                trades.append({
                    "date": date, "action": "SELL", "shares": position.shares,
                    "price": sell_price, "commission": comm,
                    "reason": reason,
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

        # Calculate time in market
        in_market_bars = 0
        for i, val in enumerate(equity):
            if i > 0 and val != equity.iloc[i - 1]:
                pass  # simplified: use equity changes
        # Better: count bars where equity > cash equivalent

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
            "round_trips": min(len(buys), len(sells)),
            "total_commission": float(total_comm),
        }

    def summary(self, result: dict, ticker: str = "QQQ") -> str:
        m = result["metrics"]
        eq = result["equity_curve"]
        trades = result["trades"]

        # Count different exit reasons
        exit_reasons = {}
        for t in trades:
            if t["action"] == "SELL":
                reason = t.get("reason", "unknown")
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        exit_summary = " | ".join(f"{k}: {v}" for k, v in exit_reasons.items())

        return f"""
{'='*55}
Trend Following (SMA50/200 + ADX + Volume + Pullback) on {ticker}
{'='*55}
Period: {eq.index[0].strftime('%Y-%m-%d')} -> {eq.index[-1].strftime('%Y-%m-%d')}
Initial: ${m['initial_equity']:,.0f} -> Final: ${m['final_equity']:,.0f}
Total Return: {m['total_return']:+.2%}
Annualized: {m['annualized_return']:+.2%} | Vol: {m['annualized_volatility']:.2%}
Sharpe: {m['sharpe_ratio']:.2f} | Sortino: {m['sortino_ratio']:.2f}
Max Drawdown: {m['max_drawdown']:.2%} | Calmar: {m['calmar_ratio']:.2f}
{'─'*55}
Round Trips: {m['round_trips']} | Buys: {m['total_buys']} | Sells: {m['total_sells']}
Exits: {exit_summary}
Commission: ${m['total_commission']:,.0f}
{'='*55}
"""
