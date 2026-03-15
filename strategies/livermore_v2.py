"""Livermore Trend-Following Strategy v2 — Graduated Stop Loss + Fear & Greed.

Key improvements over v1:
1. Graduated trailing stop: Sell 10% → 25% → 50% → 100% as price drops further
2. Fear & Greed entry: Buy small batches when VIX > 30 (fear)
3. Wider stops (8-10%) with pyramid support intact
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from strategies.base import BaseStrategy, Signal


@dataclass
class LivermorePosition:
    """Position with pyramiding and graduated exit support."""
    ticker: str
    layers: list[dict] = field(default_factory=list)
    highest_price: float = 0.0
    bars_held: int = 0
    # Track graduated stop loss triggers
    stops_hit: list[str] = field(default_factory=list)

    @property
    def total_shares(self) -> int:
        return sum(l["shares"] for l in self.layers)

    @property
    def avg_entry_price(self) -> float:
        total_cost = sum(l["shares"] * l["price"] for l in self.layers)
        return total_cost / self.total_shares if self.total_shares > 0 else 0.0

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


# Graduated stop loss tiers: (pct_from_peak, sell_fraction)
# As price drops further from peak, sell more
DEFAULT_STOP_TIERS = [
    ("stop_1", -0.08, 0.10),   # -8% from peak → sell 10%
    ("stop_2", -0.10, 0.25),   # -10% from peak → sell 25% more
    ("stop_3", -0.12, 0.50),   # -12% from peak → sell 50% more
    ("stop_4", -0.15, 1.00),   # -15% from peak → sell remaining
]


class LivermoreV2Strategy(BaseStrategy):
    """
    Livermore v2: Graduated Stop Loss + Fear & Greed Entry.

    Rules:
    - Trend identified by price > 200 SMA (primary trend filter)
    - Entry at 0.5 Fibonacci retracement (shallower than v1 for strong trends)
    - Breakout entry: price breaks above recent swing high
    - Graduated trailing stop: tiered exits as price drops from peak
    - Fear buying: VIX > 30 triggers small batch accumulation
    - Pyramiding: add to winners on breakout confirmation
    """

    def __init__(
        self,
        lookback: int = 60,
        fib_level: float = 0.5,        # Shallower than v1's 0.618
        stop_tiers: list = None,        # Graduated stop tiers
        max_pyramids: int = 3,
        trend_sma: int = 200,
        vix_fear_threshold: float = 30.0,
        vix_extreme_fear: float = 40.0,
        breakout_lookback: int = 20,    # Days for breakout high
    ):
        super().__init__("LivermoreV2")
        self.lookback = lookback
        self.fib_level = fib_level
        self.stop_tiers = stop_tiers or DEFAULT_STOP_TIERS
        self.max_pyramids = max_pyramids
        self.trend_sma = trend_sma
        self.vix_fear = vix_fear_threshold
        self.vix_extreme = vix_extreme_fear
        self.breakout_lookback = breakout_lookback

    def _find_swing_high(self, df: pd.DataFrame, idx: int, lookback: int) -> tuple[float, int]:
        start = max(0, idx - lookback)
        highs = df["High"].iloc[start:idx+1]
        max_idx = highs.idxmax()
        return float(df["High"].loc[max_idx]), df.index.get_loc(max_idx)

    def _find_swing_low(self, df: pd.DataFrame, idx: int, lookback: int) -> tuple[float, int]:
        start = max(0, idx - lookback)
        lows = df["Low"].iloc[start:idx+1]
        min_idx = lows.idxmin()
        return float(df["Low"].loc[min_idx]), df.index.get_loc(min_idx)

    def _is_uptrend(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < self.trend_sma:
            return False
        sma_val = df["Close"].iloc[max(0, idx-self.trend_sma):idx+1].mean()
        return df["Close"].iloc[idx] > sma_val

    def _get_vix(self, df: pd.DataFrame, idx: int) -> float | None:
        """Get VIX value if available."""
        if "VIX" in df.columns:
            val = df["VIX"].iloc[idx]
            return None if pd.isna(val) else float(val)
        return None

    def _calc_fibonacci_levels(self, swing_high: float, swing_low: float) -> dict:
        diff = swing_high - swing_low
        return {
            "0.382": swing_low + 0.382 * diff,
            "0.5": swing_low + 0.5 * diff,
            "0.618": swing_low + 0.618 * diff,
            "1.0": swing_high,
        }

    def get_stop_loss(self, entry_price: float) -> float:
        return 0.0  # Handled by graduated stop in engine

    def get_take_profit(self, entry_price: float) -> float:
        return 0.0  # Let winners run

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        if idx < max(self.lookback, self.trend_sma) + 1:
            return Signal.HOLD

        close = df["Close"].iloc[idx]
        prev_close = df["Close"].iloc[idx - 1]

        # === FEAR & GREED ENTRY (contrarian) ===
        vix = self._get_vix(df, idx)
        if vix is not None and vix >= self.vix_extreme:
            # Extreme fear → strong buy signal
            return Signal.BUY
        if vix is not None and vix >= self.vix_fear:
            # Fear → buy signal (smaller position handled by engine)
            return Signal.BUY

        # === TREND-FOLLOWING ENTRY ===
        if not self._is_uptrend(df, idx):
            return Signal.HOLD

        swing_high, high_idx = self._find_swing_high(df, idx, self.lookback)
        swing_low, low_idx = self._find_swing_low(df, idx, self.lookback)

        if low_idx >= high_idx:
            return Signal.HOLD

        fib = self._calc_fibonacci_levels(swing_high, swing_low)

        # Fib retracement entry
        fib_entry = fib[f"{self.fib_level}"]
        prev_at_fib = prev_close <= fib_entry * 1.01
        bouncing = close > prev_close

        if prev_at_fib and bouncing:
            return Signal.BUY

        # Breakout entry: price breaks above recent swing high
        recent_high = df["High"].iloc[max(0, idx-self.breakout_lookback):idx].max()
        if close > recent_high and prev_close <= recent_high:
            return Signal.BUY

        return Signal.HOLD


class LivermoreV2Engine:
    """
    Backtest engine with:
    - Pyramiding (add to winners)
    - Graduated trailing stop (tiered exits)
    - VIX-based position sizing (smaller in fear, larger in greed)
    """

    def __init__(
        self,
        strategy: LivermoreV2Strategy,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position_pct: float = 0.25,
        pyramid_profit_trigger: float = 0.05,
        fear_position_scale: float = 0.4,    # 40% of normal size when VIX > 30
        normal_position_scale: float = 1.0,  # Normal size
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        self.pyramid_profit_trigger = pyramid_profit_trigger
        self.fear_scale = fear_position_scale
        self.normal_scale = normal_position_scale

    def _get_position_scale(self, df: pd.DataFrame, idx: int) -> float:
        """Scale position size based on VIX (fear = smaller positions)."""
        if "VIX" not in df.columns:
            return self.normal_scale
        vix = df["VIX"].iloc[idx]
        if pd.isna(vix):
            return self.normal_scale
        if vix >= self.strategy.vix_extreme:
            return self.fear_scale * 0.7  # Even smaller in extreme fear
        if vix >= self.strategy.vix_fear:
            return self.fear_scale
        return self.normal_scale

    def run(self, df: pd.DataFrame, ticker: str = "QQQ") -> dict:
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
            low = df["Low"].iloc[i]

            # === Update position & check graduated stops ===
            if position and position.total_shares > 0:
                position.bars_held += 1
                position.highest_price = max(position.highest_price, high)

                # Check graduated stop loss tiers
                drawdown_from_peak = (low - position.highest_price) / position.highest_price

                for tier_name, trigger_pct, sell_frac in self.strategy.stop_tiers:
                    if tier_name in position.stops_hit:
                        continue  # Already triggered this tier
                    if drawdown_from_peak <= trigger_pct:
                        # This tier triggered — sell the specified fraction
                        position.stops_hit.append(tier_name)

                        if sell_frac >= 1.0:
                            # Final tier — sell everything
                            sell_shares = position.total_shares
                        else:
                            # Partial sell
                            sell_shares = max(1, int(position.total_shares * sell_frac))

                        if sell_shares > 0 and sell_shares <= position.total_shares:
                            sell_price = close * (1 - self.slippage)
                            comm = sell_shares * sell_price * self.commission
                            cash += sell_shares * sell_price - comm

                            # Remove shares from oldest layers (FIFO)
                            remaining = sell_shares
                            layers_to_remove = []
                            for li, layer in enumerate(position.layers):
                                if remaining <= 0:
                                    break
                                if layer["shares"] <= remaining:
                                    remaining -= layer["shares"]
                                    layers_to_remove.append(li)
                                else:
                                    layer["shares"] -= remaining
                                    remaining = 0
                            for li in reversed(layers_to_remove):
                                position.layers.pop(li)

                            trades.append({
                                "date": date, "action": "SELL", "shares": sell_shares,
                                "price": sell_price, "commission": comm,
                                "reason": tier_name, "drawdown": f"{drawdown_from_peak:.1%}",
                            })

                            if position.total_shares == 0:
                                # Fully exited
                                completed_trades.append({
                                    "entry_date": date,  # Will be updated below
                                    "exit_date": date,
                                    "net_pnl": 0,  # Will calc at end
                                    "exit_reason": f"graduated_stop_{tier_name}",
                                    "layers_at_exit": len(position.layers),
                                })
                                position = None
                                pyramid_count = 0
                        break  # Only trigger one tier per bar

            # === Check signal ===
            signal = self.strategy.generate_signal(df, i)
            pos_scale = self._get_position_scale(df, i)

            if signal == Signal.BUY:
                if position is None:
                    # New position
                    max_value = cash * self.max_position_pct * pos_scale
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
                            pyramid_count = 0
                            reason = "fear_buy" if pos_scale < self.normal_scale else "fib_entry"
                            trades.append({
                                "date": date, "action": "BUY", "shares": shares,
                                "price": buy_price, "commission": comm, "reason": reason,
                            })

                elif (position.total_shares > 0
                      and pyramid_count < self.strategy.max_pyramids
                      and position.pnl_pct(close) > self.pyramid_profit_trigger):
                    # Pyramid — add to winner
                    max_value = cash * self.max_position_pct * pos_scale
                    buy_price = close * (1 + self.slippage)
                    add_shares = int(max_value / buy_price)
                    if add_shares > 0:
                        comm = add_shares * buy_price * self.commission
                        cost = add_shares * buy_price + comm
                        if cost <= cash:
                            cash -= cost
                            position.add_layer(add_shares, buy_price, date, comm)
                            pyramid_count += 1
                            # Reset stop tiers on pyramid (new cost basis)
                            position.stops_hit = []
                            position.highest_price = close
                            trades.append({
                                "date": date, "action": "PYRAMID", "shares": add_shares,
                                "price": buy_price, "commission": comm,
                                "reason": f"pyramid_{pyramid_count}",
                                "avg_entry": position.avg_entry_price,
                                "total_shares": position.total_shares,
                            })

            # Record equity
            total_value = cash
            if position and position.total_shares > 0:
                total_value += position.market_value(close)
            equity.append(total_value)

        # Close remaining at end
        if position and position.total_shares > 0:
            close = df["Close"].iloc[-1]
            date = df.index[-1]
            sell_price = close * (1 - self.slippage)
            total_shares = position.total_shares
            comm = total_shares * sell_price * self.commission
            cash += total_shares * sell_price - comm
            trades.append({
                "date": date, "action": "SELL", "shares": total_shares,
                "price": sell_price, "commission": comm, "reason": "end_of_data",
            })

        equity_series = pd.Series(equity, index=df.index[:len(equity)])
        return {
            "trades": trades,
            "completed_trades": completed_trades,
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
        ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / periods) - 1
        ann_vol = daily_returns.std() * (252 ** 0.5) if not daily_returns.empty else 0.0
        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0.0
        sortino = (daily_returns.mean() / downside.std()) * (252 ** 0.5) if not downside.empty and downside.std() > 0 else 0.0
        max_dd = float(((equity / equity.cummax()) - 1).min())
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

        total_comm = sum(t["commission"] for t in trades)
        buys = [t for t in trades if t["action"] == "BUY"]
        sells = [t for t in trades if t["action"] == "SELL"]
        pyramids = [t for t in trades if t["action"] == "PYRAMID"]
        fear_buys = [t for t in trades if t.get("reason") == "fear_buy"]

        # Exposure
        in_position = 0
        for eq_val in equity:
            pass  # Simplified

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
            "total_trades": len(trades),
            "buys": len(buys),
            "sells": len(sells),
            "pyramid_adds": len(pyramids),
            "fear_buys": len(fear_buys),
            "total_commission": float(total_comm),
        }

    def summary(self, result: dict, ticker: str = "QQQ") -> str:
        m = result["metrics"]
        eq = result["equity_curve"]
        return f"""
{'='*55}
Livermore V2 (Graduated Stop + Fear&Greed) on {ticker}
{'='*55}
Period: {eq.index[0].strftime('%Y-%m-%d')} -> {eq.index[-1].strftime('%Y-%m-%d')}
Initial: ${m['initial_equity']:,.0f}  ->  Final: ${m['final_equity']:,.0f}
Total Return: {m['total_return']:+.2%}
Annualized: {m['annualized_return']:+.2%} | Vol: {m['annualized_volatility']:.2%}
Sharpe: {m['sharpe_ratio']:.2f} | Sortino: {m['sortino_ratio']:.2f}
Max Drawdown: {m['max_drawdown']:.2%} | Calmar: {m['calmar_ratio']:.2f}
{'─'*55}
Buys: {m['buys']} | Sells: {m['sells']} | Pyramids: {m['pyramid_adds']}
Fear Buys: {m['fear_buys']}
Commission: ${m['total_commission']:,.0f}
{'='*55}
"""
