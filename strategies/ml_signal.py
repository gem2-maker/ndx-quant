"""ML-based trading strategy — bridges TrendPredictor to BacktestEngine."""

from __future__ import annotations

import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal
from ml.predictor import TrendPredictor
from indicators.technical import add_all_indicators


class MLSignalStrategy(BaseStrategy):
    """
    Trading strategy driven by ML predictions.

    Trains a TrendPredictor on historical data, then uses its predictions
    as trading signals during backtesting. Supports confidence-based filtering
    and look-ahead prevention via proper time-series splitting.

    Parameters:
        model_type: ML model to use ("xgboost", "random_forest", "gradient_boosting")
        lookback: Rolling window for feature engineering
        forecast_horizon: Bars ahead to predict
        confidence_threshold: Minimum confidence to act on ("low", "medium", "high")
        retrain_interval: Retrain model every N bars (0 = never retrain)
        n_estimators: Number of trees for ensemble
        max_depth: Max tree depth
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        lookback: int = 20,
        forecast_horizon: int = 5,
        confidence_threshold: str = "low",
        retrain_interval: int = 0,
        n_estimators: int = 200,
        max_depth: int = 6,
        train_period_ratio: float = 0.3,
        min_probability_edge: float = 0.08,
        trend_filter: bool = True,
        momentum_filter: bool = True,
        volatility_limit: float = 0.04,
    ):
        super().__init__(name=f"ML_{model_type}")
        self.model_type = model_type
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.confidence_threshold = confidence_threshold
        self.retrain_interval = retrain_interval
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.train_period_ratio = train_period_ratio
        self.min_probability_edge = min_probability_edge
        self.trend_filter = trend_filter
        self.momentum_filter = momentum_filter
        self.volatility_limit = volatility_limit

        self._predictor: TrendPredictor | None = None
        self._trained = False
        self._last_retrain_idx = 0
        self._prediction_cache: dict[int, tuple[int, float, str]] = {}
        self._confidence_order = {"low": 0, "medium": 1, "high": 2}

    def _ensure_trained(self, df: pd.DataFrame, current_idx: int) -> None:
        """Train or retrain the model if needed."""
        min_train_bars = max(100, self.lookback * 10)
        train_end_idx = current_idx - self.forecast_horizon

        if train_end_idx < min_train_bars:
            return  # Not enough data yet

        need_train = (
            not self._trained
            or (
                self.retrain_interval > 0
                and (current_idx - self._last_retrain_idx) >= self.retrain_interval
            )
        )

        if not need_train:
            return

        # Train on data up to (current_idx - forecast_horizon) to prevent look-ahead
        train_df = df.iloc[:train_end_idx].copy()

        self._predictor = TrendPredictor(
            model_type=self.model_type,
            lookback=self.lookback,
            forecast_horizon=self.forecast_horizon,
            test_size=0.15,  # Small test split for walk-forward
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
        )

        try:
            self._predictor.train(train_df)
            self._trained = True
            self._last_retrain_idx = current_idx
        except (ValueError, RuntimeError) as e:
            # Training failed (not enough data, etc.) — stay untrained
            self._trained = False

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        """Generate trading signal for bar at index `idx`."""
        # Check cache first
        if idx in self._prediction_cache:
            pred, prob, conf = self._prediction_cache[idx]
            if self._should_trade(conf, prob, df, idx, pred):
                if pred == 1:
                    return Signal.BUY
                return Signal.SELL if self._allow_short(df, idx) else Signal.HOLD
            return self._fallback_signal(df, idx)

        # Train model if needed
        self._ensure_trained(df, idx)

        if not self._trained:
            return self._fallback_signal(df, idx)

        # Get prediction for current bar
        try:
            # We need to build features at this specific index
            window_start = max(0, idx - self.lookback * 3)  # Extra buffer for indicators
            window_df = df.iloc[window_start:idx + 1].copy()

            if len(window_df) < self.lookback + 10:
                return Signal.HOLD

            # Build features and predict
            featured_df = self._predictor.feature_engineer.build_features(window_df)
            X, _ = self._predictor.feature_engineer.get_X_y(featured_df)

            if X.empty:
                return Signal.HOLD

            # Take the last row (current bar)
            X_latest = X.tail(1)
            X_scaled = self._predictor.scaler.transform(X_latest)

            prediction = int(self._predictor.model.predict(X_scaled)[0])
            prob_up = float(self._predictor.model.predict_proba(X_scaled)[0][1])
            confidence = self._predictor._confidence(prob_up)

            # Cache it
            self._prediction_cache[idx] = (prediction, prob_up, confidence)

            if self._should_trade(confidence, prob_up, df, idx, prediction):
                if prediction == 1:
                    return Signal.BUY
                return Signal.SELL if self._allow_short(df, idx) else Signal.HOLD
            return self._fallback_signal(df, idx)

        except Exception:
            return Signal.HOLD

    def _should_trade(self, confidence: str, prob_up: float, df: pd.DataFrame, idx: int, prediction: int) -> bool:
        """Check if model confidence + market regime filters allow a trade."""
        confidence_ok = self._confidence_order.get(confidence, 0) >= self._confidence_order.get(self.confidence_threshold, 0)
        edge_ok = abs(prob_up - 0.5) >= self.min_probability_edge
        volatility_ok = self._volatility_ok(df, idx)
        trend_ok = self._trend_ok(df, idx, prediction)
        momentum_ok = self._momentum_ok(df, idx, prediction)
        return confidence_ok and edge_ok and volatility_ok and trend_ok and momentum_ok

    def position_size_multiplier(self, df: pd.DataFrame, idx: int) -> float:
        """Dynamic sizing from ML edge + trend strength + volatility regime."""
        if idx < 200:
            return 0.0

        close = df["Close"]
        price = float(close.iloc[idx])
        sma50 = float(close.iloc[max(0, idx - 49):idx + 1].mean())
        sma200 = float(close.iloc[max(0, idx - 199):idx + 1].mean())

        # Trend strength: distance from SMA200 normalized by recent volatility.
        returns = close.pct_change().iloc[max(0, idx - 30):idx]
        vol = float(returns.std()) if not returns.empty else 0.0
        if vol <= 1e-9:
            trend_strength = 0.0
        else:
            trend_strength = min(abs((price / sma200) - 1.0) / (2.0 * vol), 1.0)

        # Volatility penalty: scale down in noisier markets.
        vol_penalty = 1.0
        if self.volatility_limit > 0 and vol > 0:
            vol_penalty = min(self.volatility_limit / vol, 1.0)

        cached = self._prediction_cache.get(idx)
        if cached is None:
            # Fallback trend sizing when ML has no actionable edge yet.
            bullish = price >= sma50 >= sma200
            base = 7.0 if bullish else 0.0
            return max(0.0, min(base * vol_penalty, 10.0))

        prediction, prob_up, _ = cached
        edge = min(abs(prob_up - 0.5) / 0.5, 1.0)  # 0~1
        direction_ok = (prediction == 1 and price >= sma50 >= sma200) or (prediction == 0 and price <= sma50 <= sma200)
        if not direction_ok:
            return 0.0

        score = 0.55 * edge + 0.45 * trend_strength
        # Convert confidence score to leverage of base cap (10% base * multiplier).
        # Typical range 3~10 => 30%~100% notional exposure.
        sized = 3.0 + 7.0 * score
        return max(1.0, min(sized * vol_penalty, 10.0))

    def _volatility_ok(self, df: pd.DataFrame, idx: int) -> bool:
        """Skip trades when short-term volatility is unusually high."""
        if idx < self.lookback + 5:
            return False
        returns = df["Close"].pct_change().iloc[max(0, idx - self.lookback):idx]
        if returns.empty:
            return False
        return float(returns.std()) <= self.volatility_limit

    def _trend_ok(self, df: pd.DataFrame, idx: int, prediction: int) -> bool:
        """Align BUY/SELL decision with medium-term trend regime."""
        if not self.trend_filter:
            return True
        if idx < 200:
            return False

        close = df["Close"]
        sma50 = float(close.iloc[max(0, idx - 49):idx + 1].mean())
        sma200 = float(close.iloc[max(0, idx - 199):idx + 1].mean())
        price = float(close.iloc[idx])

        # BUY only in bullish regime, SELL only in bearish regime.
        if prediction == 1:
            return price >= sma50 and sma50 >= sma200
        return price <= sma50 and sma50 <= sma200

    def _momentum_ok(self, df: pd.DataFrame, idx: int, prediction: int) -> bool:
        """Second opinion from RSI+MACD to reduce whipsaw trades."""
        if not self.momentum_filter:
            return True
        if idx < 35:
            return False

        row = df.iloc[idx]
        rsi = float(row.get("RSI", np.nan))
        macd = float(row.get("MACD", np.nan))
        macd_signal = float(row.get("MACD_Signal", np.nan))

        if np.isnan(rsi) or np.isnan(macd) or np.isnan(macd_signal):
            return False

        if prediction == 1:
            return (rsi >= 50.0) and (macd >= macd_signal)
        return (rsi <= 45.0) and (macd <= macd_signal)

    def _allow_short(self, df: pd.DataFrame, idx: int) -> bool:
        """Only allow short exits in clearly bearish regimes."""
        if idx < 200:
            return False
        close = df["Close"]
        price = float(close.iloc[idx])
        sma200 = float(close.iloc[max(0, idx - 199):idx + 1].mean())
        rsi = float(df.iloc[idx].get("RSI", np.nan))
        return price < sma200 * 0.985 and (not np.isnan(rsi) and rsi < 45.0)

    def _fallback_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        """Trend-following fallback when ML edge is weak (strategy fusion)."""
        if idx < 200:
            return Signal.HOLD
        close = df["Close"]
        price = float(close.iloc[idx])
        sma50 = float(close.iloc[max(0, idx - 49):idx + 1].mean())
        sma200 = float(close.iloc[max(0, idx - 199):idx + 1].mean())
        rsi = float(df.iloc[idx].get("RSI", np.nan))

        bullish = price >= sma50 >= sma200 and (np.isnan(rsi) or rsi >= 50)
        bearish = price <= sma50 <= sma200 and (not np.isnan(rsi) and rsi <= 45)

        if bullish:
            return Signal.BUY
        if bearish:
            return Signal.SELL if self._allow_short(df, idx) else Signal.HOLD
        return Signal.HOLD

    def get_stop_loss(self, entry_price: float) -> float:
        """Wider protective stop for trend-following behavior."""
        return entry_price * 0.90

    def get_take_profit(self, entry_price: float) -> float:
        """Let winners run; only cap very extended moves."""
        return entry_price * 1.35

    @property
    def training_info(self) -> dict:
        """Return info about the trained model."""
        if not self._trained or not self._predictor:
            return {"trained": False}
        return {
            "trained": True,
            "model_type": self.model_type,
            "features": len(self._predictor.feature_names),
            "last_retrain_at_bar": self._last_retrain_idx,
            "cached_predictions": len(self._prediction_cache),
            "min_probability_edge": self.min_probability_edge,
            "trend_filter": self.trend_filter,
            "momentum_filter": self.momentum_filter,
            "volatility_limit": self.volatility_limit,
        }
