"""Momentum strategy — buy strong performers, sell weak ones."""

import pandas as pd

from config import SMA_SHORT, SMA_LONG
from strategies.base import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """
    Dual Moving Average Momentum:
    - BUY when short SMA crosses above long SMA (golden cross)
    - SELL when short SMA crosses below long SMA (death cross)
    - Confirmation: price must be above/below both MAs
    """

    def __init__(
        self,
        short_period: int = SMA_SHORT,
        long_period: int = SMA_LONG,
    ):
        super().__init__("Momentum")
        self.short_period = short_period
        self.long_period = long_period

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        if idx < self.long_period + 1:
            return Signal.HOLD

        sma_short = df["SMA_20"].iloc[idx]
        sma_long = df["SMA_50"].iloc[idx]
        prev_sma_short = df["SMA_20"].iloc[idx - 1]
        prev_sma_long = df["SMA_50"].iloc[idx - 1]
        close = df["Close"].iloc[idx]

        if pd.isna(sma_short) or pd.isna(sma_long):
            return Signal.HOLD

        # Golden cross: short crosses above long + price confirmation
        if prev_sma_short <= prev_sma_long and sma_short > sma_long and close > sma_long:
            return Signal.BUY

        # Death cross: short crosses below long
        if prev_sma_short >= prev_sma_long and sma_short < sma_long and close < sma_long:
            return Signal.SELL

        return Signal.HOLD


class PriceMomentumStrategy(BaseStrategy):
    """
    Simple price momentum:
    - BUY when N-day return is strongly positive and RSI is rising but not overbought
    - SELL when RSI is overbought or momentum turns negative
    """

    def __init__(self, lookback: int = 20, rsi_threshold: float = 65):
        super().__init__("PriceMomentum")
        self.lookback = lookback
        self.rsi_threshold = rsi_threshold

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        if idx < self.lookback + 1:
            return Signal.HOLD

        close = df["Close"].iloc[idx]
        prev_close = df["Close"].iloc[idx - self.lookback]
        momentum = (close - prev_close) / prev_close

        rsi_val = df["RSI"].iloc[idx]
        if pd.isna(rsi_val):
            return Signal.HOLD

        # Strong positive momentum + RSI confirms but not overbought
        if momentum > 0.05 and 40 < rsi_val < self.rsi_threshold:
            return Signal.BUY

        # Overbought or momentum fading
        if rsi_val > 75 or momentum < -0.03:
            return Signal.SELL

        return Signal.HOLD
