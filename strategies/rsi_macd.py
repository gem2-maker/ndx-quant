"""RSI + MACD combo strategy."""

import pandas as pd

from config import RSI_OVERSOLD, RSI_OVERBOUGHT
from strategies.base import BaseStrategy, Signal


class RsiMacdStrategy(BaseStrategy):
    """
    RSI + MACD Combo:
    - BUY when RSI crosses above oversold AND MACD histogram turns positive
    - SELL when RSI crosses above overbought OR MACD histogram turns negative
    """

    def __init__(
        self,
        rsi_oversold: float = RSI_OVERSOLD,
        rsi_overbought: float = RSI_OVERBOUGHT,
    ):
        super().__init__("RSI+MACD")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        if idx < 2:
            return Signal.HOLD

        rsi_val = df["RSI"].iloc[idx]
        prev_rsi = df["RSI"].iloc[idx - 1]
        macd_hist = df["MACD_Hist"].iloc[idx]
        prev_macd_hist = df["MACD_Hist"].iloc[idx - 1]

        if any(pd.isna(v) for v in [rsi_val, prev_rsi, macd_hist, prev_macd_hist]):
            return Signal.HOLD

        # RSI coming out of oversold + MACD turning positive
        if prev_rsi < self.rsi_oversold and rsi_val >= self.rsi_oversold:
            if prev_macd_hist <= 0 and macd_hist > 0:
                return Signal.BUY
            # Also buy if RSI crosses up even without MACD confirmation (aggressive)
            if rsi_val - prev_rsi > 5:
                return Signal.BUY

        # RSI overbought or MACD turning negative
        if rsi_val > self.rsi_overbought:
            return Signal.SELL
        if prev_macd_hist >= 0 and macd_hist < 0 and rsi_val > 50:
            return Signal.SELL

        return Signal.HOLD
