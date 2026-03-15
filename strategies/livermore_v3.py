"""Livermore v3 — Value + Momentum + ML Composite Scoring.

Combines:
1. Value factor: RSI oversold, Bollinger Band position, SMA deviation
2. Momentum factor: Trend direction, volume surge, price momentum, breakout
3. ML factor: Random Forest direction prediction (P(up))
4. Graduated stop loss (from v2)
5. Fear & Greed (VIX) contrarian entry
6. Multi-Fibonacci resonance (0.382 / 0.5 / 0.618)

Position size = base_size * composite_score (3-factor weighted)
"""

import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from strategies.base import BaseStrategy, Signal


@dataclass
class LivermorePosition:
    ticker: str
    layers: list = field(default_factory=list)
    highest_price: float = 0.0
    bars_held: int = 0
    stops_hit: list = field(default_factory=list)

    @property
    def total_shares(self):
        return sum(l["shares"] for l in self.layers)

    @property
    def avg_entry_price(self):
        total_cost = sum(l["shares"] * l["price"] for l in self.layers)
        return total_cost / self.total_shares if self.total_shares > 0 else 0.0

    @property
    def total_commission(self):
        return sum(l["commission"] for l in self.layers)

    def market_value(self, price):
        return self.total_shares * price

    def pnl_pct(self, price):
        avg = self.avg_entry_price
        return (price - avg) / avg if avg > 0 else 0.0

    def add_layer(self, shares, price, date, commission):
        self.layers.append({"shares": shares, "price": price, "date": date, "commission": commission})


# Graduated stop loss tiers
STOP_TIERS = [
    ("stop_1", -0.10, 0.10),   # -10% from peak -> sell 10%  (wider than v2's -8%)
    ("stop_2", -0.12, 0.25),   # -12% from peak -> sell 25% more
    ("stop_3", -0.15, 0.50),   # -15% from peak -> sell 50% more
    ("stop_4", -0.18, 1.00),   # -18% from peak -> sell all
]


class LivermoreV3Strategy(BaseStrategy):
    """
    Livermore v3: Value + Momentum + ML Composite Scoring.

    Value Score (0-1):
    - RSI: oversold = high score
    - Bollinger Band position: lower band = high score
    - SMA deviation: below SMA200 = higher score (contrarian)

    Momentum Score (0-1):
    - Trend direction (price vs SMA50/200)
    - Volume vs average
    - Price momentum (20-day return)
    - Breakout above recent high

    ML Score (0-1):
    - Random Forest P(up) from trained model
    - 0.5 = neutral, >0.5 = bullish, <0.5 = bearish
    - Falls back to 0.5 if model unavailable
    """

    def __init__(
        self,
        lookback: int = 60,
        fib_level: float = 0.5,
        max_pyramids: int = 3,
        trend_sma: int = 200,
        vix_fear: float = 30.0,
        vix_extreme: float = 40.0,
        breakout_lookback: int = 20,
        # 3-factor weights (sum to 1.0)
        value_weight: float = 0.3,
        momentum_weight: float = 0.4,
        ml_weight: float = 0.3,
        # Thresholds
        min_composite_score: float = 0.50,
        pe_cheap: float = 25.0,
        pe_expensive: float = 35.0,
        # ML settings
        model_path: str = "models/QQQ_random_forest.pkl",
        walk_forward: bool = False,
        retrain_interval: int = 60,
    ):
        super().__init__("LivermoreV3")
        self.lookback = lookback
        self.fib_level = fib_level
        self.max_pyramids = max_pyramids
        self.trend_sma = trend_sma
        self.vix_fear = vix_fear
        self.vix_extreme = vix_extreme
        self.breakout_lookback = breakout_lookback
        self.value_weight = value_weight
        self.momentum_weight = momentum_weight
        self.ml_weight = ml_weight
        self.min_composite_score = min_composite_score
        self.pe_cheap = pe_cheap
        self.pe_expensive = pe_expensive
        # ML
        self.model_path = model_path
        self.walk_forward = walk_forward
        self.retrain_interval = retrain_interval
        self._predictor = None
        self._ml_loaded = False
        self._prediction_cache = {}  # idx -> ml_score
        self._last_retrain_bar = 0
        self._ml_feature_matrix = None
        self._ml_target_vector = None

    def _value_score(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate value score (0 = expensive, 1 = cheap)."""
        scores = []

        # 1. PE score
        if "implied_pe" in df.columns:
            pe = df["implied_pe"].iloc[idx]
            if not pd.isna(pe):
                if pe <= self.pe_cheap:
                    pe_score = 1.0
                elif pe >= self.pe_expensive:
                    pe_score = 0.0
                else:
                    pe_score = 1.0 - (pe - self.pe_cheap) / (self.pe_expensive - self.pe_cheap)
                scores.append(pe_score)

        # 2. RSI score (lower RSI = more oversold = higher value)
        if "RSI" in df.columns:
            rsi = df["RSI"].iloc[idx]
            if not pd.isna(rsi):
                if rsi <= 30:
                    rsi_score = 1.0
                elif rsi >= 70:
                    rsi_score = 0.0
                else:
                    rsi_score = 1.0 - (rsi - 30) / 40
                scores.append(rsi_score)

        # 3. Bollinger Band position
        if all(c in df.columns for c in ["Close", "BB_Lower", "BB_Upper"]):
            bb_lower = df["BB_Lower"].iloc[idx]
            bb_upper = df["BB_Upper"].iloc[idx]
            close = df["Close"].iloc[idx]
            if not pd.isna(bb_lower) and bb_upper > bb_lower:
                bb_pos = (close - bb_lower) / (bb_upper - bb_lower)
                bb_score = 1.0 - max(0.0, min(1.0, bb_pos))
                scores.append(bb_score)

        # 4. SMA deviation (below long-term SMA = contrarian buy)
        if idx >= self.trend_sma:
            close = df["Close"].iloc[idx]
            sma_val = df["Close"].iloc[idx - self.trend_sma:idx + 1].mean()
            deviation = (close - sma_val) / sma_val
            if deviation <= -0.15:
                sma_score = 1.0
            elif deviation >= 0.15:
                sma_score = 0.0
            else:
                sma_score = 0.5 - deviation / 0.30
            scores.append(sma_score)

        return np.mean(scores) if scores else 0.5

    def _momentum_score(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate momentum score (0 = weak, 1 = strong)."""
        scores = []

        # 1. Price trend (above SMA50 = bullish)
        if idx >= 50:
            close = df["Close"].iloc[idx]
            sma50 = df["Close"].iloc[idx - 49:idx + 1].mean()
            sma200 = df["Close"].iloc[max(0, idx - 199):idx + 1].mean() if idx >= 200 else sma50
            if close > sma50 > sma200:
                scores.append(1.0)
            elif close > sma50:
                scores.append(0.7)
            elif close > sma200:
                scores.append(0.4)
            else:
                scores.append(0.1)

        # 2. 20-day momentum
        if idx >= 20:
            ret_20d = (df["Close"].iloc[idx] - df["Close"].iloc[idx - 20]) / df["Close"].iloc[idx - 20]
            if ret_20d >= 0.10:
                scores.append(1.0)
            elif ret_20d >= 0.05:
                scores.append(0.8)
            elif ret_20d >= 0:
                scores.append(0.6)
            elif ret_20d >= -0.05:
                scores.append(0.3)
            else:
                scores.append(0.1)

        # 3. Volume surge (volume > 1.5x average = conviction)
        if "Volume" in df.columns and idx >= 20:
            vol = df["Volume"].iloc[idx]
            avg_vol = df["Volume"].iloc[idx - 19:idx + 1].mean()
            if avg_vol > 0:
                vol_ratio = vol / avg_vol
                if vol_ratio >= 2.0:
                    scores.append(1.0)
                elif vol_ratio >= 1.5:
                    scores.append(0.8)
                elif vol_ratio >= 1.0:
                    scores.append(0.5)
                else:
                    scores.append(0.3)

        # 4. Breakout
        if idx >= self.breakout_lookback:
            close = df["Close"].iloc[idx]
            recent_high = df["High"].iloc[idx - self.breakout_lookback:idx].max()
            if close > recent_high:
                scores.append(1.0)
            elif close > recent_high * 0.98:
                scores.append(0.7)
            else:
                scores.append(0.3)

        return np.mean(scores) if scores else 0.5

    def _fib_entry_signal(self, df: pd.DataFrame, idx: int) -> bool:
        """Check if price is at a Fibonacci retracement level."""
        if idx < self.lookback + 1:
            return False

        start = max(0, idx - self.lookback)
        swing_high = df["High"].iloc[start:idx + 1].max()
        swing_low = df["Low"].iloc[start:idx + 1].min()
        diff = swing_high - swing_low

        if diff <= 0:
            return False

        close = df["Close"].iloc[idx]
        prev_close = df["Close"].iloc[idx - 1]

        # Check multiple Fibonacci levels
        for level in [0.382, 0.5, 0.618]:
            fib_price = swing_low + level * diff
            # Price near this fib level and bouncing
            if prev_close <= fib_price * 1.02 and close > prev_close:
                return True
        return False

    def _load_ml_model(self):
        """Load ML model if available. Call once on first use."""
        if self._ml_loaded:
            return
        self._ml_loaded = True
        try:
            from ml.predictor import TrendPredictor
            self._predictor = TrendPredictor(model_type="random_forest")
            # Try loading with the filename convention
            path = Path(self.model_path)
            if path.exists():
                import pickle
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self._predictor.model = data["model"]
                self._predictor.scaler = data["scaler"]
                self._predictor.feature_names = data.get("feature_names", [])
                self._predictor._trained = True
                print(f"  [V3] ML model loaded: {path}")
            else:
                print(f"  [V3] ML model not found at {path}, ML score disabled")
                self._predictor = None
        except Exception as e:
            print(f"  [V3] ML model load failed: {e}, ML score disabled")
            self._predictor = None

    def _prepare_ml_features(self, df: pd.DataFrame):
        """Pre-compute aligned ML features once per backtest run."""
        self._prediction_cache.clear()
        self._last_retrain_bar = 0
        self._ml_feature_matrix = None
        self._ml_target_vector = None

        self._load_ml_model()

        if self._predictor is None or not self._predictor._trained:
            return

        try:
            featured_df = self._predictor.build_feature_frame(df)
            X, y = self._predictor.prepare_feature_matrix(
                featured_df,
                self._predictor.feature_names or None,
            )

            bar_positions = pd.Series(np.arange(len(df)), index=df.index)
            aligned_positions = bar_positions.reindex(featured_df.index)
            valid_rows = aligned_positions.notna()

            if not valid_rows.any():
                return

            X = X.loc[valid_rows].copy()
            y = y.loc[valid_rows].copy()
            X.index = aligned_positions.loc[valid_rows].astype(int)
            y.index = X.index
            self._ml_feature_matrix = X
            self._ml_target_vector = y
        except Exception as e:
            print(f"  [V3] ML feature pre-compute failed: {e}, ML score disabled")
            self._ml_feature_matrix = None
            self._ml_target_vector = None

    def _retrain_ml_model(self, df: pd.DataFrame, idx: int):
        """Retrain the model on a walk-forward schedule using precomputed features."""
        if not self.walk_forward or self._predictor is None:
            return

        if idx - self._last_retrain_bar < self.retrain_interval:
            return

        train_end_idx = idx - self._predictor.forecast_horizon
        if (
            train_end_idx < 0
            or self._ml_feature_matrix is None
            or self._ml_target_vector is None
        ):
            return

        train_rows = self._ml_feature_matrix.index <= train_end_idx
        if int(train_rows.sum()) < 50:
            return

        try:
            X_train = self._ml_feature_matrix.loc[train_rows]
            y_train = self._ml_target_vector.loc[train_rows]
            self._predictor.train_from_matrix(X_train, y_train)
        except Exception:
            return

        self._last_retrain_bar = idx

    def _ml_score(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate ML score (0 = bearish, 0.5 = neutral, 1 = bullish).

        Uses cached prediction if available, otherwise runs inference.
        In walk-forward mode, retrains every retrain_interval bars.
        """
        if idx in self._prediction_cache:
            return self._prediction_cache[idx]

        self._load_ml_model()

        if (
            self._predictor is None
            or not self._predictor._trained
            or self._ml_feature_matrix is None
            or idx not in self._ml_feature_matrix.index
        ):
            return 0.5  # neutral fallback

        try:
            self._retrain_ml_model(df, idx)

            X_latest = self._ml_feature_matrix.loc[[idx]]
            X_scaled = self._predictor.scaler.transform(X_latest)
            score = float(self._predictor.model.predict_proba(X_scaled)[0][1])
        except Exception:
            score = 0.5

        self._prediction_cache[idx] = score
        return score

    def get_stop_loss(self, entry_price):
        return 0.0

    def get_take_profit(self, entry_price):
        return 0.0

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        if idx < max(self.lookback, self.trend_sma) + 1:
            return Signal.HOLD

        # VIX extreme fear → always buy
        if "VIX" in df.columns:
            vix = df["VIX"].iloc[idx]
            if not pd.isna(vix) and vix >= self.vix_extreme:
                return Signal.BUY

        # Calculate composite score (3-factor)
        value = self._value_score(df, idx)
        momentum = self._momentum_score(df, idx)
        ml = self._ml_score(df, idx)
        composite = (self.value_weight * value +
                     self.momentum_weight * momentum +
                     self.ml_weight * ml)

        # Fibonacci entry check
        fib_signal = self._fib_entry_signal(df, idx)

        # VIX fear boost
        fear_boost = 0.0
        if "VIX" in df.columns:
            vix = df["VIX"].iloc[idx]
            if not pd.isna(vix) and vix >= self.vix_fear:
                fear_boost = 0.15

        # Combined signal
        if composite + fear_boost >= self.min_composite_score and fib_signal:
            return Signal.BUY

        # Also allow pure high-score entry (strong value + momentum even without fib)
        if composite + fear_boost >= self.min_composite_score + 0.15:
            return Signal.BUY

        return Signal.HOLD


class LivermoreV3Engine:
    """Engine with pyramiding, graduated stops, and composite scoring."""

    def __init__(
        self,
        strategy: LivermoreV3Strategy,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position_pct: float = 0.30,
        pyramid_profit_trigger: float = 0.05,
        pyramid_min_score: float = 0.60,  # Min composite score to pyramid
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        self.pyramid_profit_trigger = pyramid_profit_trigger
        self.pyramid_min_score = pyramid_min_score

    def run(self, df: pd.DataFrame, ticker: str = "QQQ") -> dict:
        self.strategy._prepare_ml_features(df)

        cash = self.initial_capital
        position = None
        trades = []
        equity = []
        pyramid_count = 0

        for i in range(len(df)):
            close = df["Close"].iloc[i]
            date = df.index[i]
            high = df["High"].iloc[i]
            low = df["Low"].iloc[i]

            # === Update position & graduated stops ===
            if position and position.total_shares > 0:
                position.bars_held += 1
                position.highest_price = max(position.highest_price, high)

                drawdown = (low - position.highest_price) / position.highest_price

                for tier_name, trigger_pct, sell_frac in STOP_TIERS:
                    if tier_name in position.stops_hit:
                        continue
                    if drawdown <= trigger_pct:
                        position.stops_hit.append(tier_name)
                        sell_shares = max(1, int(position.total_shares * sell_frac)) if sell_frac < 1.0 else position.total_shares
                        sell_shares = min(sell_shares, position.total_shares)

                        if sell_shares > 0:
                            sell_price = close * (1 - self.slippage)
                            comm = sell_shares * sell_price * self.commission
                            cash += sell_shares * sell_price - comm

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
                                "reason": tier_name, "drawdown": f"{drawdown:.1%}",
                            })

                            if position.total_shares == 0:
                                position = None
                                pyramid_count = 0
                        break

            # === Signal & scoring ===
            signal = self.strategy.generate_signal(df, i)
            value_score = self.strategy._value_score(df, i)
            momentum_score = self.strategy._momentum_score(df, i)
            ml_score = self.strategy._ml_score(df, i)
            composite = (self.strategy.value_weight * value_score +
                        self.strategy.momentum_weight * momentum_score +
                        self.strategy.ml_weight * ml_score)

            # Position size scales with composite score
            pos_scale = 0.5 + composite * 0.5  # 0.5x to 1.0x

            if signal == Signal.BUY:
                if position is None:
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
                            is_fear = "VIX" in df.columns and not pd.isna(df["VIX"].iloc[i]) and df["VIX"].iloc[i] >= self.strategy.vix_fear
                            reason = "fear_buy" if is_fear else "composite_entry"
                            trades.append({
                                "date": date, "action": "BUY", "shares": shares,
                                "price": buy_price, "commission": comm,
                                "reason": reason,
                                "value": f"{value_score:.2f}", "momentum": f"{momentum_score:.2f}",
                                "ml": f"{ml_score:.2f}",
                                "composite": f"{composite:.2f}",
                            })

                elif (position.total_shares > 0
                      and pyramid_count < self.strategy.max_pyramids
                      and position.pnl_pct(close) > self.pyramid_profit_trigger
                      and composite >= self.pyramid_min_score):
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
                            position.stops_hit = []
                            position.highest_price = close
                            trades.append({
                                "date": date, "action": "PYRAMID", "shares": add_shares,
                                "price": buy_price, "commission": comm,
                                "reason": f"pyramid_{pyramid_count}",
                                "avg_entry": position.avg_entry_price,
                                "total_shares": position.total_shares,
                                "value": f"{value_score:.2f}", "momentum": f"{momentum_score:.2f}",
                                "ml": f"{ml_score:.2f}",
                                "composite": f"{composite:.2f}",
                            })

            # Record equity
            total_value = cash
            if position and position.total_shares > 0:
                total_value += position.market_value(close)
            equity.append(total_value)

        # Close at end
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
            "equity_curve": equity_series,
            "metrics": self._calc_metrics(equity_series, trades),
        }

    def _calc_metrics(self, equity, trades):
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

        buys = [t for t in trades if t["action"] == "BUY"]
        sells = [t for t in trades if t["action"] == "SELL"]
        pyramids = [t for t in trades if t["action"] == "PYRAMID"]
        fear_buys = [t for t in trades if t.get("reason") == "fear_buy"]
        total_comm = sum(t["commission"] for t in trades)

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
            "buys": len(buys),
            "sells": len(sells),
            "pyramid_adds": len(pyramids),
            "fear_buys": len(fear_buys),
            "total_commission": float(total_comm),
        }

    def summary(self, result, ticker="QQQ"):
        m = result["metrics"]
        eq = result["equity_curve"]
        return f"""
{'='*55}
Livermore V3 (Value + Momentum) on {ticker}
{'='*55}
Period: {eq.index[0].strftime('%Y-%m-%d')} -> {eq.index[-1].strftime('%Y-%m-%d')}
Initial: ${m['initial_equity']:,.0f} -> Final: ${m['final_equity']:,.0f}
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
