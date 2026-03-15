import math
import unittest

import pandas as pd

from backtest.engine import BacktestEngine
from strategies.base import BaseStrategy, Signal


class ScriptedStrategy(BaseStrategy):
    def __init__(self, signals: list[Signal], stop_loss_pct: float = 0.05, take_profit_pct: float = 10.0):
        super().__init__(name="scripted")
        self.signals = signals
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate_signal(self, df: pd.DataFrame, idx: int) -> Signal:
        return self.signals[idx]

    def get_stop_loss(self, entry_price: float) -> float:
        return entry_price * (1 - self.stop_loss_pct)

    def get_take_profit(self, entry_price: float) -> float:
        return entry_price * (1 + self.take_profit_pct)


class BacktestAnalyticsTests(unittest.TestCase):
    def test_completed_trade_metrics_for_signal_exit(self):
        df = pd.DataFrame(
            {
                "Close": [100.0, 110.0, 105.0],
                "SMA_20": [100.0, 100.0, 100.0],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        strategy = ScriptedStrategy([Signal.BUY, Signal.HOLD, Signal.SELL])
        engine = BacktestEngine(strategy, initial_capital=1_000.0, commission=0.0, slippage=0.0, max_position_pct=1.0)

        result = engine.run(df, ticker="QQQ")

        self.assertEqual(len(result.completed_trades), 1)
        trade = result.completed_trades[0]
        self.assertEqual(trade.exit_reason, "signal_profit")
        self.assertEqual(trade.shares, 10)
        self.assertEqual(trade.hold_period_bars, 2)
        self.assertEqual(trade.hold_period_days, 2)
        self.assertAlmostEqual(trade.gross_pnl, 50.0)
        self.assertAlmostEqual(trade.net_pnl, 50.0)
        self.assertAlmostEqual(trade.return_pct, 0.05)
        self.assertAlmostEqual(trade.max_favorable_excursion_pct, 0.10)
        self.assertAlmostEqual(trade.max_adverse_excursion_pct, 0.0)

        metrics = result.metrics
        self.assertEqual(metrics["completed_trades"], 1)
        self.assertAlmostEqual(metrics["win_rate"], 1.0)
        self.assertTrue(math.isinf(metrics["profit_factor"]))
        self.assertAlmostEqual(metrics["average_hold_days"], 2.0)
        self.assertAlmostEqual(metrics["exposure_ratio"], 2 / 3)

    def test_completed_trade_metrics_for_stop_loss_exit(self):
        df = pd.DataFrame(
            {
                "Close": [100.0, 94.0, 96.0],
                "SMA_20": [100.0, 100.0, 100.0],
            },
            index=pd.date_range("2024-02-01", periods=3, freq="D"),
        )
        strategy = ScriptedStrategy([Signal.BUY, Signal.HOLD, Signal.HOLD], stop_loss_pct=0.05, take_profit_pct=10.0)
        engine = BacktestEngine(strategy, initial_capital=1_000.0, commission=0.0, slippage=0.0, max_position_pct=1.0)

        result = engine.run(df, ticker="QQQ")

        self.assertEqual(len(result.completed_trades), 1)
        trade = result.completed_trades[0]
        self.assertEqual(trade.exit_reason, "stop_loss")
        self.assertEqual(trade.hold_period_bars, 1)
        self.assertEqual(trade.hold_period_days, 1)
        self.assertAlmostEqual(trade.net_pnl, -60.0)
        self.assertAlmostEqual(trade.max_favorable_excursion_pct, 0.0)
        self.assertAlmostEqual(trade.max_adverse_excursion_pct, -0.06)

        metrics = result.metrics
        self.assertEqual(metrics["completed_trades"], 1)
        self.assertAlmostEqual(metrics["win_rate"], 0.0)
        self.assertAlmostEqual(metrics["profit_factor"], 0.0)
        self.assertAlmostEqual(metrics["gross_loss"], -60.0)
        self.assertEqual(metrics["max_consecutive_losses"], 1)


if __name__ == "__main__":
    unittest.main()
