"""Quick In-and-Out Strategy — Short-term mean reversion captures.

Targets oversold bounces and breakout continuations with tight stops.
Holding period: 2-14 days. High win-rate, fast turnover.

Entry signals:
1. RSI oversold bounce (RSI < 30 then crosses back above 30)
2. Bollinger Band squeeze breakout (bandwidth < threshold, then price breaks out)
3. Volume-price divergence (price makes lower low but volume decreases)

Exit signals:
1. RSI overbought (RSI > 70)
2. Hit take-profit (default 5%)
3. Hit stop-loss (default 3%)
4. Max holding period exceeded (default 14 bars)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from strategies.base import BaseStrategy, Signal


@dataclass
class QuickPosition:
    """Short-term position with strict risk controls."""
    ticker: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    entry_reason: str = ""
    bars_held: int = 0
    commission: float = 0.0

    def pnl_pct(self, price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (price - self.entry_price) / self.entry_price


class QuickTradeStrategy(BaseStrategy):
    """Quick in-and-out: RSI bounce + BB breakout + volume divergence.

    Three independent entry signals, each with its own logic.
    All share the same risk controls: tight stop, quick TP, max hold time.
    """

    def __init__(
        self,
        # RSI settings
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        # Bollinger Band settings
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_squeeze_threshold: float = 0.04,  # BB width / price < 4% = squeeze
        # Volume divergence
        volume_lookback: int = 20,
        price_lookback: int = 10,
        # Risk controls
        stop_loss_pct: float = 0.03,    # 3% hard stop
        take_profit_pct: float = 0.05,  # 5% target
        max_hold_bars: int = 14,        # Max 14 trading days
        # Entry filters
        min_rsi_bounce: float = 5.0,    # RSI must rise this much from low
        min_volume_ratio: float = 0.8,  # Volume at low < 80% of average = divergence
    ):
        super().__init__("QuickTrade")
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_squeeze_threshold = bb_squeeze_threshold
        self.volume_lookback = volume_lookback
        self.price_lookback = price_lookback
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_bars = max_hold_bars
        self.min_rsi_bounce = min_rsi_bounce
        self.min_volume_ratio = min_volume_ratio

        # State
        self._last_exit_bar: int = -999
        self._cooldown_bars: int = 5  # Wait N bars after exit before re-entry

    def reset(self):
        self._last_exit_bar = -999

    def _compute_rsi(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """Compute RSI at given index."""
        if idx < self.rsi_period:
            return None
        closes = df["Close"].iloc[:idx + 1]
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if not pd.isna(val) else None

    def _compute_bb(self, df: pd.DataFrame, idx: int) -> Optional[tuple]:
        """Compute Bollinger Bands: (upper, middle, lower, width_pct)."""
        if idx < self.bb_period:
            return None
        closes = df["Close"].iloc[idx - self.bb_period + 1:idx + 1]
        middle = closes.mean()
        std = closes.std()
        upper = middle + self.bb_std * std
        lower = middle - self.bb_std * std
        close = df["Close"].iloc[idx]
        width_pct = (upper - lower) / close if close > 0 else 0
        return (upper, middle, lower, width_pct)

    def _rsi_bounce_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """RSI was oversold, now bouncing back up."""
        if idx < self.rsi_period + 3:
            return False

        rsi_now = self._compute_rsi(df, idx)
        rsi_prev = self._compute_rsi(df, idx - 1)
        rsi_low = self._compute_rsi(df, idx - 2)

        if rsi_now is None or rsi_prev is None or rsi_low is None:
            return False

        # RSI crossed above oversold threshold
        if rsi_prev <= self.rsi_oversold and rsi_now > self.rsi_oversold:
            return True

        # Or RSI bounced significantly from a low
        if rsi_low < self.rsi_oversold + 10 and (rsi_now - rsi_low) >= self.min_rsi_bounce:
            return True

        return False

    def _bb_breakout_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """Bollinger Band squeeze followed by breakout."""
        if idx < self.bb_period + 5:
            return False

        bb_now = self._compute_bb(df, idx)
        if bb_now is None:
            return False

        upper, middle, lower, width_pct = bb_now
        close = df["Close"].iloc[idx]

        # Check if we were in a squeeze recently
        was_squeeze = False
        for lookback in range(2, 6):
            if idx - lookback < self.bb_period:
                break
            bb_past = self._compute_bb(df, idx - lookback)
            if bb_past and bb_past[3] < self.bb_squeeze_threshold:
                was_squeeze = True
                break

        if not was_squeeze:
            return False

        # Price broke above upper band (bullish breakout)
        if close > upper:
            return True

        return False

    def _volume_divergence_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """Price made lower low but volume is declining = divergence."""
        if "Volume" not in df.columns:
            return False
        if idx < max(self.price_lookback, self.volume_lookback) + 1:
            return False

        # Price: check if recent low is lower than previous low
        recent_low = df["Low"].iloc[idx - self.price_lookback // 2:idx + 1].min()
        prev_low = df["Low"].iloc[idx - self.price_lookback:idx - self.price_lookback // 2].min()

        if recent_low >= prev_low:
            return False  # Not a lower low

        # Volume: check if volume at recent low is declining
        recent_vol = df["Volume"].iloc[idx - self.price_lookback // 2:idx + 1].mean()
        prev_vol = df["Volume"].iloc[idx - self.price_lookback:idx - self.price_lookback // 2].mean()

        if prev_vol <= 0:
            return False

        vol_ratio = recent_vol / prev_vol
        return vol_ratio < self.min_volume_ratio

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        if idx < max(self.rsi_period, self.bb_period, self.volume_lookback) + 5:
            return Signal.HOLD

        # Cooldown after exit
        if idx - self._last_exit_bar < self._cooldown_bars:
            return Signal.HOLD

        # Check entry signals (any one triggers entry)
        if self._rsi_bounce_signal(df, idx):
            return Signal.BUY

        if self._bb_breakout_signal(df, idx):
            return Signal.BUY

        if self._volume_divergence_signal(df, idx):
            return Signal.BUY

        return Signal.HOLD


class QuickTradeEngine:
    """Engine for Quick Trade strategy with strict risk management."""

    def __init__(
        self,
        strategy: QuickTradeStrategy,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_pct: float = 0.30,  # 30% per trade (conservative)
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_pct = position_pct

    def run(self, df: pd.DataFrame, ticker: str = "QQQ") -> dict:
        self.strategy.reset()

        cash = self.initial_capital
        position: Optional[QuickPosition] = None
        trades = []
        equity = []

        for i in range(len(df)):
            close = df["Close"].iloc[i]
            date = df.index[i]
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]

            # === Check exits on existing position ===
            if position and position.shares > 0:
                position.bars_held += 1

                # Stop loss
                if low <= position.entry_price * (1 - self.strategy.stop_loss_pct):
                    sell_price = position.entry_price * (1 - self.strategy.stop_loss_pct) * (1 - self.slippage)
                    comm = position.shares * sell_price * self.commission
                    cash += position.shares * sell_price - comm
                    trades.append({
                        "date": date, "action": "SELL", "shares": position.shares,
                        "price": sell_price, "commission": comm,
                        "reason": "stop_loss",
                        "pnl": f"{position.pnl_pct(sell_price):+.1%}",
                        "bars_held": position.bars_held,
                    })
                    position = None
                    self.strategy._last_exit_bar = i

                # Take profit
                elif high >= position.entry_price * (1 + self.strategy.take_profit_pct):
                    sell_price = position.entry_price * (1 + self.strategy.take_profit_pct) * (1 - self.slippage)
                    comm = position.shares * sell_price * self.commission
                    cash += position.shares * sell_price - comm
                    trades.append({
                        "date": date, "action": "SELL", "shares": position.shares,
                        "price": sell_price, "commission": comm,
                        "reason": "take_profit",
                        "pnl": f"{position.pnl_pct(sell_price):+.1%}",
                        "bars_held": position.bars_held,
                    })
                    position = None
                    self.strategy._last_exit_bar = i

                # Max holding period
                elif position.bars_held >= self.strategy.max_hold_bars:
                    sell_price = close * (1 - self.slippage)
                    comm = position.shares * sell_price * self.commission
                    cash += position.shares * sell_price - comm
                    trades.append({
                        "date": date, "action": "SELL", "shares": position.shares,
                        "price": sell_price, "commission": comm,
                        "reason": "max_hold",
                        "pnl": f"{position.pnl_pct(sell_price):+.1%}",
                        "bars_held": position.bars_held,
                    })
                    position = None
                    self.strategy._last_exit_bar = i

                # RSI overbought exit
                elif position.bars_held >= 3:  # Min hold 3 bars
                    rsi = self.strategy._compute_rsi(df, i)
                    if rsi is not None and rsi >= self.strategy.rsi_overbought:
                        sell_price = close * (1 - self.slippage)
                        comm = position.shares * sell_price * self.commission
                        cash += position.shares * sell_price - comm
                        trades.append({
                            "date": date, "action": "SELL", "shares": position.shares,
                            "price": sell_price, "commission": comm,
                            "reason": "rsi_overbought",
                            "pnl": f"{position.pnl_pct(sell_price):+.1%}",
                            "bars_held": position.bars_held,
                        })
                        position = None
                        self.strategy._last_exit_bar = i

            # === Entry signal ===
            if position is None:
                signal = self.strategy.generate_signal(df, i)
                if signal == Signal.BUY:
                    investable = cash * self.position_pct
                    buy_price = close * (1 + self.slippage)
                    shares = int(investable / buy_price)
                    if shares > 0:
                        comm = shares * buy_price * self.commission
                        cost = shares * buy_price + comm
                        if cost <= cash:
                            cash -= cost
                            # Determine entry reason
                            if self.strategy._rsi_bounce_signal(df, i):
                                reason = "rsi_bounce"
                            elif self.strategy._bb_breakout_signal(df, i):
                                reason = "bb_breakout"
                            else:
                                reason = "vol_divergence"

                            rsi = self.strategy._compute_rsi(df, i)
                            position = QuickPosition(
                                ticker=ticker, shares=shares,
                                entry_price=buy_price, entry_date=date,
                                entry_reason=reason, commission=comm,
                            )
                            trades.append({
                                "date": date, "action": "BUY", "shares": shares,
                                "price": buy_price, "commission": comm,
                                "reason": reason,
                                "rsi": f"{rsi:.1f}" if rsi else "N/A",
                            })

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
                "bars_held": position.bars_held,
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

        # Win rate from PnL strings
        wins = sum(1 for t in sells if t.get("pnl", "").startswith("+"))
        losses = sum(1 for t in sells if t.get("pnl", "").startswith("-"))
        total_completed = wins + losses
        win_rate = wins / total_completed if total_completed > 0 else 0.0

        # Exit reason breakdown
        exit_reasons = {}
        for t in sells:
            reason = t.get("reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        # Average hold period
        hold_bars = [t.get("bars_held", 0) for t in sells if t.get("bars_held", 0) > 0]
        avg_hold = sum(hold_bars) / len(hold_bars) if hold_bars else 0.0

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
            "total_trades": len(buys),
            "win_rate": float(win_rate),
            "wins": wins,
            "losses": losses,
            "avg_hold_days": float(avg_hold),
            "exit_reasons": exit_reasons,
            "total_commission": float(total_comm),
        }

    def summary(self, result: dict, ticker: str = "QQQ") -> str:
        m = result["metrics"]
        eq = result["equity_curve"]

        exit_summary = " | ".join(f"{k}: {v}" for k, v in m.get("exit_reasons", {}).items())

        return f"""
{'='*55}
Quick Trade (RSI/BB/Volume) on {ticker}
{'='*55}
Period: {eq.index[0].strftime('%Y-%m-%d')} -> {eq.index[-1].strftime('%Y-%m-%d')}
Initial: ${m['initial_equity']:,.0f} -> Final: ${m['final_equity']:,.0f}
Total Return: {m['total_return']:+.2%}
Annualized: {m['annualized_return']:+.2%} | Vol: {m['annualized_volatility']:.2%}
Sharpe: {m['sharpe_ratio']:.2f} | Sortino: {m['sortino_ratio']:.2f}
Max Drawdown: {m['max_drawdown']:.2%} | Calmar: {m['calmar_ratio']:.2f}
{'─'*55}
Trades: {m['total_trades']} | Win Rate: {m['win_rate']:.0%} ({m['wins']}W / {m['losses']}L)
Avg Hold: {m['avg_hold_days']:.1f} days
Exits: {exit_summary}
Commission: ${m['total_commission']:,.0f}
{'='*55}
"""
