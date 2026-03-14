"""Mean Reversion strategy — buy oversold, sell overbought."""

import pandas as pd

from config import BB_PERIOD, BB_STD, RSI_OVERSOLD, RSI_OVERBOUGHT
from strategies.base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """
    Bollinger Band + RSI Mean Reversion:
    - BUY when price touches lower Bollinger Band AND RSI is oversold
    - SELL when price touches upper Bollinger Band AND RSI is overbought
    """

    def __init__(
        self,
        bb_period: int = BB_PERIOD,
        bb_std: float = BB_STD,
        rsi_oversold: float = RSI_OVERSOLD,
        rsi_overbought: float = RSI_OVERBOUGHT,
    ):
        super().__init__("MeanReversion")
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        if idx < self.bb_period:
            return Signal.HOLD

        close = df["Close"].iloc[idx]
        bb_upper = df["BB_Upper"].iloc[idx]
        bb_lower = df["BB_Lower"].iloc[idx]
        rsi_val = df["RSI"].iloc[idx]

        if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(rsi_val):
            return Signal.HOLD

        # Price at lower band + RSI oversold → buy the dip
        if close <= bb_lower and rsi_val < self.rsi_oversold:
            return Signal.BUY

        # Price at upper band + RSI overbought → take profit
        if close >= bb_upper and rsi_val > self.rsi_overbought:
            return Signal.SELL

        return Signal.HOLD
