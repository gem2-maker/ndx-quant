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
            if self._should_trade(conf):
                return Signal.BUY if pred == 1 else Signal.SELL
            return Signal.HOLD

        # Train model if needed
        self._ensure_trained(df, idx)

        if not self._trained:
            return Signal.HOLD

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

            if self._should_trade(confidence):
                return Signal.BUY if prediction == 1 else Signal.SELL
            return Signal.HOLD

        except Exception:
            return Signal.HOLD

    def _should_trade(self, confidence: str) -> bool:
        """Check if confidence meets threshold."""
        return self._confidence_order.get(confidence, 0) >= self._confidence_order.get(self.confidence_threshold, 0)

    def get_stop_loss(self, entry_price: float) -> float:
        """ML strategy uses tighter stop loss (3%)."""
        return entry_price * 0.97

    def get_take_profit(self, entry_price: float) -> float:
        """ML strategy uses tighter take profit (10%)."""
        return entry_price * 1.10

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
        }
