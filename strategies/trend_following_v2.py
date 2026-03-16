"""Enhanced Long-Term Trend Following Strategy (V2).

Incorporates improvements from open-source trend-following research:
- Multi-horizon momentum (63/126/252 days) with sqrt-weighted averaging
- Skip-last-20-days to avoid short-term mean-reversion noise
- Signal normalization to [-1, 1] range
- Lag signal by 1 bar to prevent lookahead bias
- Monthly rebalance mode (reduces whipsaw, lower transaction costs)
- Volatility regime filter (only trade when vol is sufficient)

Designed for: QQQ / NDX-100 trend following, long-only.
Typical holding period: weeks to months.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence

from strategies.base import BaseStrategy, Signal


@dataclass
class TrendPositionV2:
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


def compute_momentum_signal(
    prices: pd.DataFrame,
    lookbacks: Sequence[int] = (63, 126, 252),
    skip_last_n: int = 20,
    strength_clip: float = 3.0,
) -> pd.Series:
    """Multi-horizon time-series momentum signal.
    
    Returns a normalized signal in [-1, 1]:
      > 0 = uptrend (long)
      < 0 = downtrend (short)
      = 0 = no trend
      
    Key idea from systematic-trend-following research:
    - Use multiple lookback periods weighted by 1/sqrt(lookback)
      (shorter periods weighted more as they respond faster)
    - Skip last N days to avoid short-term mean-reversion noise
    """
    close = prices["Close"].copy().sort_index()
    shifted = close.shift(skip_last_n)  # Skip recent noise
    
    weights = np.array([1.0 / np.sqrt(lb) for lb in lookbacks], dtype=float)
    weights = weights / weights.sum()  # Normalize weights
    
    combined = pd.Series(0.0, index=close.index)
    for weight, lookback in zip(weights, lookbacks):
        momentum = shifted / shifted.shift(lookback) - 1.0
        combined = combined.add(momentum * weight, fill_value=0.0)
    
    # Normalize to [-1, 1] using tanh-like clipping
    # Scale by typical momentum magnitude (~0.1 for 1-year lookback)
    normalized = np.clip(combined * 5.0, -strength_clip, strength_clip) / strength_clip
    return normalized.fillna(0.0)


def compute_sma_crossover_signal(
    prices: pd.DataFrame,
    fast: int = 50,
    slow: int = 200,
    strength_clip: float = 3.0,
) -> pd.Series:
    """SMA crossover signal normalized to [-1, 1]."""
    close = prices["Close"].copy()
    sma_fast = close.rolling(window=fast, min_periods=fast).mean()
    sma_slow = close.rolling(window=slow, min_periods=slow).mean()
    
    raw = (sma_fast - sma_slow) / (sma_slow.abs() + 1e-9)
    normalized = np.clip(raw * 10.0, -strength_clip, strength_clip) / strength_clip
    return normalized.fillna(0.0)


def compute_adx_signal(
    prices: pd.DataFrame,
    period: int = 14,
    threshold: float = 20.0,
) -> pd.Series:
    """ADX-based trend strength signal.
    
    Returns:
      > 0 when ADX > threshold (trending market)
      = 0 when ADX <= threshold (choppy market)
    """
    high = prices["High"]
    low = prices["Low"]
    close = prices["Close"]
    
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
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=alpha, min_periods=period).mean() / atr
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period).mean()
    
    # Signal: positive when ADX > threshold (strong trend), 0 otherwise
    signal = pd.Series(0.0, index=adx.index)
    signal[adx > threshold] = 1.0
    # Bonus: directional bias when ADX is strong
    signal = signal * np.sign(plus_di - minus_di).clip(0, 1)  # Only long signals for now
    return signal.fillna(0.0)


def compute_volatility_regime(
    prices: pd.DataFrame,
    vol_window: int = 20,
    vol_threshold: float = 0.015,
) -> pd.Series:
    """Volatility regime filter (from MLM strategy).
    
    Returns 1 when realized vol > threshold (enough movement to trade),
    0 when vol is too low (whipsaw risk).
    """
    close = prices["Close"]
    returns = close.pct_change()
    realized_vol = returns.rolling(window=vol_window).std()
    
    signal = pd.Series(0.0, index=close.index)
    signal[realized_vol > vol_threshold] = 1.0
    return signal.fillna(0.0)


def lag_signal(signal: pd.Series, lag: int = 1) -> pd.Series:
    """Lag signal to prevent lookahead bias.
    
    If we detect a signal on day T, we trade on day T+lag.
    """
    return signal.shift(lag)


class TrendFollowingV2(BaseStrategy):
    """Enhanced trend following with multi-horizon momentum.
    
    Entry conditions:
    1. Combined signal (SMA crossover + momentum) > threshold
    2. ADX confirms trending market
    3. Volatility regime is favorable
    4. (Optional) Monthly rebalance: only check on specific trading days
    
    Exit conditions:
    1. Combined signal < exit threshold
    2. Trailing stop hit
    3. Monthly rebalance: signal turned negative
    """

    def __init__(
        self,
        # SMA parameters
        sma_fast: int = 50,
        sma_slow: int = 200,
        # Momentum parameters
        momentum_lookbacks: tuple = (63, 126, 252),
        momentum_skip_days: int = 20,
        # Signal weights (how much each component contributes)
        sma_weight: float = 0.4,
        momentum_weight: float = 0.4,
        adx_weight: float = 0.2,
        # Thresholds
        entry_threshold: float = 0.15,    # Combined signal must exceed this
        exit_threshold: float = -0.10,    # Exit when signal drops below this
        # ADX
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        # Volatility filter
        vol_window: int = 20,
        vol_threshold: float = 0.015,
        # Rebalance
        rebalance_mode: str = "monthly",  # "daily" or "monthly"
        # Risk management
        trailing_stop_pct: float = 0.10,  # 10% trailing stop
        min_bars_between_trades: int = 10,
        signal_lag: int = 1,
    ):
        super().__init__("TrendFollowingV2")
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.momentum_lookbacks = momentum_lookbacks
        self.momentum_skip_days = momentum_skip_days
        self.sma_weight = sma_weight
        self.momentum_weight = momentum_weight
        self.adx_weight = adx_weight
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold
        self.rebalance_mode = rebalance_mode
        self.trailing_stop_pct = trailing_stop_pct
        self.min_bars_between_trades = min_bars_between_trades
        self.signal_lag = signal_lag
        
        # Pre-computed signals (set during reset)
        self._combined_signal: Optional[pd.Series] = None
        self._vol_regime: Optional[pd.Series] = None
        self._adx_signal: Optional[pd.Series] = None
        
        # State
        self._last_trade_bar: int = -999
        self._in_position: bool = False
        self._last_rebalance_date: Optional[pd.Timestamp] = None

    def reset(self):
        """Reset state and pre-compute all signals."""
        self._last_trade_bar = -999
        self._in_position = False
        self._last_rebalance_date = None
        self._combined_signal = None
        self._vol_regime = None
        self._adx_signal = None

    def precompute_signals(self, df: pd.DataFrame):
        """Pre-compute all signals for efficiency.
        
        This avoids recalculating on every bar during backtest.
        """
        # SMA crossover signal
        sma_signal = compute_sma_crossover_signal(
            df, self.sma_fast, self.sma_slow
        )
        
        # Momentum signal (the key innovation)
        mom_signal = compute_momentum_signal(
            df, self.momentum_lookbacks, self.momentum_skip_days
        )
        
        # ADX signal
        adx_sig = compute_adx_signal(df, self.adx_period, self.adx_threshold)
        
        # Combined signal with weights
        combined = (
            self.sma_weight * sma_signal +
            self.momentum_weight * mom_signal +
            self.adx_weight * adx_sig
        )
        
        # Lag the signal
        combined = lag_signal(combined, self.signal_lag)
        
        # Volatility regime
        vol = compute_volatility_regime(df, self.vol_window, self.vol_threshold)
        
        self._combined_signal = combined.fillna(0.0)
        self._vol_regime = vol.fillna(0.0)
        self._adx_signal = adx_sig.fillna(0.0)

    def _is_rebalance_day(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if today is a rebalance day."""
        if self.rebalance_mode == "daily":
            return True
        
        if self.rebalance_mode == "monthly":
            date = df.index[idx]
            # Rebalance on first trading day of month, or every ~21 bars
            if self._last_rebalance_date is None:
                return True
            # Check if at least 21 trading days have passed (roughly monthly)
            if idx - self._last_trade_bar >= 21:
                return True
            # Or if the month changed
            if date.month != self._last_rebalance_date.month:
                return True
        
        return False

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        # Need enough data for pre-computation
        if idx < max(self.sma_slow, self.momentum_lookbacks[-1] + self.momentum_skip_days, 50):
            return Signal.HOLD
        
        # Pre-compute signals on first call
        if self._combined_signal is None:
            self.precompute_signals(df)
        
        combined = self._combined_signal.iloc[idx]
        vol_ok = self._vol_regime.iloc[idx] > 0
        
        # === EXIT (ALWAYS CHECKED - risk management is daily) ===
        if self._in_position:
            if combined < self.exit_threshold:
                self._in_position = False
                self._last_trade_bar = idx
                return Signal.SELL
            return Signal.HOLD
        
        # === ENTRY (monthly rebalance to reduce whipsaw) ===
        if not self._is_rebalance_day(df, idx):
            return Signal.HOLD
        
        # Combined signal threshold
        if combined < self.entry_threshold:
            return Signal.HOLD
        
        # Volatility regime filter
        if not vol_ok:
            return Signal.HOLD
        
        # Minimum bars between trades
        if idx - self._last_trade_bar < self.min_bars_between_trades:
            return Signal.HOLD
        
        self._in_position = True
        self._last_trade_bar = idx
        self._last_rebalance_date = df.index[idx]
        return Signal.BUY


class TrendFollowingV2Engine:
    """Engine for TrendFollowingV2 strategy.
    
    Similar to TrendFollowingEngine but uses pre-computed signals
    for better performance.
    """

    def __init__(
        self,
        strategy: TrendFollowingV2,
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
        
        # Pre-compute signals for efficiency
        self.strategy.precompute_signals(df)
        
        cash = self.initial_capital
        position: Optional[TrendPositionV2] = None
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
                        sell_price = close * (1 - self.slippage)
                        comm = position.shares * sell_price * self.commission
                        cash += position.shares * sell_price - comm
                        
                        trades.append({
                            "date": date, "action": "SELL", "shares": position.shares,
                            "price": sell_price, "commission": comm,
                            "reason": "trailing_stop",
                            "drawdown": f"{drawdown:.1%}",
                            "pnl": f"{position.pnl_pct(close):+.1%}",
                        })
                        position = None
                        self.strategy._in_position = False
                        self.strategy._last_trade_bar = i

            # Signal
            signal = self.strategy.generate_signal(df, i)

            if signal == Signal.BUY and position is None:
                investable = cash * self.position_pct
                buy_price = close * (1 + self.slippage)
                shares = int(investable / buy_price)
                if shares > 0:
                    comm = shares * buy_price * self.commission
                    cost = shares * buy_price + comm
                    if cost <= cash:
                        cash -= cost
                        position = TrendPositionV2(
                            ticker=ticker, shares=shares,
                            entry_price=buy_price, entry_date=date,
                            highest_price=close, commission=comm,
                        )
                        # Get signal details for logging
                        combined = self.strategy._combined_signal.iloc[i] if self.strategy._combined_signal is not None else 0
                        adx_sig = self.strategy._adx_signal.iloc[i] if self.strategy._adx_signal is not None else 0
                        trades.append({
                            "date": date, "action": "BUY", "shares": shares,
                            "price": buy_price, "commission": comm,
                            "reason": "signal_cross",
                            "signal": f"{combined:.3f}",
                            "adx_signal": f"{adx_sig:.3f}",
                        })

            elif signal == Signal.SELL and position and position.shares > 0:
                sell_price = close * (1 - self.slippage)
                comm = position.shares * sell_price * self.commission
                cash += position.shares * sell_price - comm
                
                combined = self.strategy._combined_signal.iloc[i] if self.strategy._combined_signal is not None else 0
                trades.append({
                    "date": date, "action": "SELL", "shares": position.shares,
                    "price": sell_price, "commission": comm,
                    "reason": "signal_exit",
                    "signal": f"{combined:.3f}",
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
            "signal_series": self.strategy._combined_signal,
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
        
        # Win/loss analysis
        round_trips = []
        for i in range(min(len(buys), len(sells))):
            b, s = buys[i], sells[i]
            pnl = (s["price"] - b["price"]) / b["price"]
            round_trips.append(pnl)
        
        wins = [p for p in round_trips if p > 0]
        losses = [p for p in round_trips if p < 0]
        win_rate = len(wins) / len(round_trips) if round_trips else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf") if wins else 0.0

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
Trend Following V2 (Multi-Horizon Momentum + SMA + ADX) on {ticker}
{'='*60}
Period: {eq.index[0].strftime('%Y-%m-%d')} -> {eq.index[-1].strftime('%Y-%m-%d')}
Initial: ${m['initial_equity']:,.0f} -> Final: ${m['final_equity']:,.0f}
Total Return: {m['total_return']:+.2%}
Annualized: {m['annualized_return']:+.2%} | Vol: {m['annualized_volatility']:.2%}
Sharpe: {m['sharpe_ratio']:.2f} | Sortino: {m['sortino_ratio']:.2f}
Max Drawdown: {m['max_drawdown']:.2%} | Calmar: {m['calmar_ratio']:.2f}
{'─'*60}
Round Trips: {m['round_trips']} | Win Rate: {m['win_rate']:.1%}
Avg Win: {m['avg_win']:+.2%} | Avg Loss: {m['avg_loss']:+.2%}
Profit Factor: {m['profit_factor']:.2f}
Exits: {exit_summary}
Commission: ${m['total_commission']:,.0f}
{'='*60}
"""
