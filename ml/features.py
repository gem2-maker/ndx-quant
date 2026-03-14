"""Feature engineering for ML prediction.

Transforms OHLCV data + technical indicators into ML-ready feature matrices.
Supports rolling windows, cross features, and lagged features.
"""

import pandas as pd
import numpy as np

from indicators.technical import (
    sma, ema, rsi, macd, bollinger_bands, atr,
    add_all_indicators,
)


class FeatureEngineer:
    """Build feature matrices from price data for ML models."""

    def __init__(
        self,
        lookback: int = 20,
        forecast_horizon: int = 5,
        target_type: str = "direction",  # "direction" | "return" | "volatility"
    ):
        """
        Args:
            lookback: Number of past bars for rolling features.
            forecast_horizon: Number of bars ahead to predict.
            target_type: What to predict — "direction" (up/down), "return" (%), "volatility".
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.target_type = target_type

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build full feature set from OHLCV DataFrame.

        Returns DataFrame with feature columns + target column.
        NaN rows (from rolling windows) are dropped.
        """
        df = df.copy()

        # 1. Add all standard technical indicators
        df = add_all_indicators(df)

        # 2. Price-based features
        df = self._price_features(df)

        # 3. Rolling statistical features
        df = self._rolling_features(df)

        # 4. Cross features (interactions between indicators)
        df = self._cross_features(df)

        # 5. Lagged features
        df = self._lag_features(df)

        # 6. Target variable
        df = self._add_target(df)

        # Drop NaN rows
        df = df.dropna()

        return df

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw price-derived features."""
        # Returns over multiple horizons
        for n in [1, 2, 3, 5, 10, 20]:
            df[f"return_{n}d"] = df["Close"].pct_change(n)

        # Log returns (more normal distribution)
        df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))

        # High-Low range (normalized)
        df["hl_range_pct"] = (df["High"] - df["Low"]) / df["Close"]

        # Open-Close body (normalized)
        df["oc_body_pct"] = (df["Close"] - df["Open"]) / df["Open"]

        # Gap (today open vs yesterday close)
        df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

        # Distance from moving averages (normalized)
        for ma in ["SMA_20", "SMA_50", "EMA_12", "EMA_26"]:
            if ma in df.columns:
                df[f"dist_{ma}"] = (df["Close"] - df[ma]) / df[ma]

        # Bollinger Band position (0 = lower band, 1 = upper band)
        if all(c in df.columns for c in ["BB_Upper", "BB_Lower"]):
            bb_width = df["BB_Upper"] - df["BB_Lower"]
            df["bb_position"] = np.where(
                bb_width > 0,
                (df["Close"] - df["BB_Lower"]) / bb_width,
                0.5,
            )
            df["bb_width_pct"] = bb_width / df["Close"]

        return df

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling window statistics."""
        windows = [5, 10, 20]

        for w in windows:
            # Volatility (rolling std of returns)
            df[f"volatility_{w}d"] = df["Daily_Return"].rolling(w).std()

            # Realized volatility (annualized)
            df[f"realized_vol_{w}d"] = df["Daily_Return"].rolling(w).std() * np.sqrt(252)

            # Rolling mean return
            df[f"mean_return_{w}d"] = df["Daily_Return"].rolling(w).mean()

            # Rolling Sharpe (annualized)
            mean_r = df["Daily_Return"].rolling(w).mean()
            std_r = df["Daily_Return"].rolling(w).std()
            df[f"sharpe_{w}d"] = np.where(std_r > 0, mean_r / std_r * np.sqrt(252), 0)

            # Rolling skewness & kurtosis of returns
            df[f"skew_{w}d"] = df["Daily_Return"].rolling(w).skew()
            df[f"kurt_{w}d"] = df["Daily_Return"].rolling(w).kurt()

            # Volume features
            if "Volume" in df.columns:
                df[f"volume_sma_{w}"] = df["Volume"].rolling(w).mean()
                df[f"volume_ratio_{w}"] = df["Volume"] / df[f"volume_sma_{w}"]

            # Price range features
            df[f"high_{w}d"] = df["High"].rolling(w).max()
            df[f"low_{w}d"] = df["Low"].rolling(w).min()
            df[f"close_vs_high_{w}d"] = df["Close"] / df[f"high_{w}d"] - 1
            df[f"close_vs_low_{w}d"] = df["Close"] / df[f"low_{w}d"] - 1

            # Consecutive up/down days
            sign = np.sign(df["Daily_Return"])
            df[f"consec_up_{w}d"] = sign.rolling(w).apply(
                lambda x: self._max_consecutive(x, 1), raw=True
            )
            df[f"consec_down_{w}d"] = sign.rolling(w).apply(
                lambda x: self._max_consecutive(x, -1), raw=True
            )

        return df

    def _cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interaction features between indicators."""
        # RSI × MACD momentum
        if "RSI" in df.columns and "MACD_Hist" in df.columns:
            df["rsi_macd_signal"] = df["RSI"] * df["MACD_Hist"]

        # Trend strength: SMA crossover distance × volume
        if "SMA_20" in df.columns and "SMA_50" in df.columns:
            df["sma_cross_dist"] = (df["SMA_20"] - df["SMA_50"]) / df["SMA_50"]
            if "Volume" in df.columns:
                df["trend_volume"] = df["sma_cross_dist"] * np.log1p(df["Volume"])

        # Volatility regime: high vol + oversold = potential bounce
        if "RSI" in df.columns:
            df["vol_rsi_regime"] = np.where(
                (df.get("volatility_20d", 0) > df.get("volatility_20d", pd.Series(0)).rolling(60).mean()) &
                (df["RSI"] < 30),
                1, 0,
            )

        # MACD trend alignment
        if all(c in df.columns for c in ["MACD", "MACD_Signal"]):
            df["macd_aligned"] = (
                (df["MACD"] > df["MACD_Signal"]).astype(int) -
                (df["MACD"] < df["MACD_Signal"]).astype(int)
            )

        # RSI bands
        if "RSI" in df.columns:
            df["rsi_overbought"] = (df["RSI"] > 70).astype(int)
            df["rsi_oversold"] = (df["RSI"] < 30).astype(int)
            df["rsi_neutral"] = ((df["RSI"] >= 40) & (df["RSI"] <= 60)).astype(int)

        return df

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lagged versions of key features."""
        lag_cols = ["RSI", "MACD_Hist", "Daily_Return", "bb_position"]
        lags = [1, 2, 3, 5]

        for col in lag_cols:
            if col in df.columns:
                for lag in lags:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable for supervised learning."""
        future_return = df["Close"].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)

        if self.target_type == "direction":
            df["target"] = (future_return > 0).astype(int)  # 1 = up, 0 = down
        elif self.target_type == "return":
            df["target"] = future_return
        elif self.target_type == "volatility":
            df["target"] = (
                df["Daily_Return"]
                .rolling(self.forecast_horizon)
                .std()
                .shift(-self.forecast_horizon)
            )
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return list of feature column names (exclude OHLCV, indicators, target)."""
        exclude = {
            "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "RSI", "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Mid", "BB_Lower",
            "ATR", "VWAP", "Daily_Return", "Cumulative_Return",
            "target",
        }
        return [c for c in df.columns if c not in exclude]

    def get_X_y(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split into feature matrix X and target vector y."""
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].copy()
        y = df["target"].copy()
        return X, y

    @staticmethod
    def _max_consecutive(arr: np.ndarray, value: float) -> float:
        """Count max consecutive occurrences of value in array."""
        max_count = 0
        count = 0
        for v in arr:
            if v == value:
                count += 1
                max_count = max(max_count, count)
            else:
                count = 0
        return max_count
