"""Backtesting engine — event-driven backtester."""

import pandas as pd

from config import INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE, MAX_POSITION_PCT
from strategies.base import BaseStrategy, Signal, Position, Trade, BacktestResult, CompletedTrade
from indicators.technical import add_all_indicators


class BacktestEngine:
    """
    Simple event-driven backtester.

    Walks through price data bar by bar, applies strategy signals,
    and tracks portfolio performance.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = INITIAL_CAPITAL,
        commission: float = COMMISSION_RATE,
        slippage: float = SLIPPAGE,
        max_position_pct: float = MAX_POSITION_PCT,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_pct = max_position_pct

    @staticmethod
    def _complete_trade(
        position: Position,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_commission: float,
        exit_reason: str,
    ) -> CompletedTrade:
        gross_pnl = (exit_price - position.entry_price) * position.shares
        net_pnl = gross_pnl - position.entry_commission - exit_commission
        capital_at_risk = (position.entry_price * position.shares) + position.entry_commission
        return_pct = (net_pnl / capital_at_risk) if capital_at_risk > 0 else 0.0

        return CompletedTrade(
            ticker=position.ticker,
            entry_date=position.entry_date,
            exit_date=exit_date,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            entry_commission=position.entry_commission,
            exit_commission=exit_commission,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=return_pct,
            hold_period_bars=max(position.bars_held, 1),
            hold_period_days=max((exit_date - position.entry_date).days, 0),
            max_favorable_excursion_pct=(
                (position.highest_price - position.entry_price) / position.entry_price
                if position.entry_price else 0.0
            ),
            max_adverse_excursion_pct=(
                (position.lowest_price - position.entry_price) / position.entry_price
                if position.entry_price else 0.0
            ),
            exit_reason=exit_reason,
        )

    def run(self, df: pd.DataFrame, ticker: str = "TICKER") -> BacktestResult:
        """Run backtest on a single ticker's DataFrame."""
        # Ensure indicators are present
        if "SMA_20" not in df.columns:
            df = add_all_indicators(df)

        cash = self.initial_capital
        position: Position | None = None
        trades: list[Trade] = []
        completed_trades: list[CompletedTrade] = []
        equity = []

        for i in range(len(df)):
            close = df["Close"].iloc[i]
            date = df.index[i]

            # Update position current price
            if position:
                position.current_price = close
                position.bars_held += 1
                position.highest_price = max(position.highest_price, close)
                position.lowest_price = min(position.lowest_price, close)

                # Check stop loss
                if position.stop_loss > 0 and close <= position.stop_loss:
                    shares = position.shares
                    sell_price = close * (1 - self.slippage)
                    comm = shares * sell_price * self.commission
                    cash += shares * sell_price - comm
                    trades.append(Trade(
                        ticker=ticker, date=date, action="SELL",
                        shares=shares, price=sell_price,
                        commission=comm, reason="stop_loss",
                    ))
                    completed_trades.append(self._complete_trade(position, date, sell_price, comm, "stop_loss"))
                    position = None

                # Check take profit
                elif position.take_profit > 0 and close >= position.take_profit:
                    shares = position.shares
                    sell_price = close * (1 - self.slippage)
                    comm = shares * sell_price * self.commission
                    cash += shares * sell_price - comm
                    pnl = "profit" if sell_price > position.entry_price else "loss"
                    trades.append(Trade(
                        ticker=ticker, date=date, action="SELL",
                        shares=shares, price=sell_price,
                        commission=comm, reason=f"take_profit_{pnl}",
                    ))
                    completed_trades.append(self._complete_trade(position, date, sell_price, comm, f"take_profit_{pnl}"))
                    position = None

            # Get signal
            signal = self.strategy.generate_signal(df, i)

            # Execute trades
            if signal == Signal.BUY and position is None:
                # Calculate position size (supports strategy-level dynamic sizing)
                size_mult = self.strategy.position_size_multiplier(df, i)
                # Allow strategy to scale base cap (e.g., 0.0~10.0 where 10x on 10% cap = 100% notional).
                size_mult = max(0.0, min(float(size_mult), 10.0))
                max_value = cash * self.max_position_pct * size_mult
                buy_price = close * (1 + self.slippage)
                shares = int(max_value / buy_price)
                if shares > 0:
                    comm = shares * buy_price * self.commission
                    cost = shares * buy_price + comm
                    if cost <= cash:
                        cash -= cost
                        position = Position(
                            ticker=ticker, shares=shares,
                            entry_price=buy_price, entry_date=date,
                            current_price=close,
                            entry_commission=comm,
                            stop_loss=self.strategy.get_stop_loss(buy_price),
                            take_profit=self.strategy.get_take_profit(buy_price),
                            bars_held=0,
                            highest_price=close,
                            lowest_price=close,
                        )
                        trades.append(Trade(
                            ticker=ticker, date=date, action="BUY",
                            shares=shares, price=buy_price,
                            commission=comm, reason="signal",
                        ))

            elif signal == Signal.SELL and position is not None:
                shares = position.shares
                sell_price = close * (1 - self.slippage)
                comm = shares * sell_price * self.commission
                cash += shares * sell_price - comm
                pnl = "profit" if sell_price > position.entry_price else "loss"
                trades.append(Trade(
                    ticker=ticker, date=date, action="SELL",
                    shares=shares, price=sell_price,
                    commission=comm, reason=f"signal_{pnl}",
                ))
                completed_trades.append(self._complete_trade(position, date, sell_price, comm, f"signal_{pnl}"))
                position = None

            # Record equity
            total_value = cash
            if position:
                total_value += position.shares * close
            equity.append(total_value)

        # Close any open position at end
        if position:
            close = df["Close"].iloc[-1]
            position.current_price = close
            position.highest_price = max(position.highest_price, close)
            position.lowest_price = min(position.lowest_price, close)
            sell_price = close * (1 - self.slippage)
            comm = position.shares * sell_price * self.commission
            cash += position.shares * sell_price - comm
            trades.append(Trade(
                ticker=ticker, date=df.index[-1], action="SELL",
                shares=position.shares, price=sell_price,
                commission=comm, reason="end_of_data",
            ))
            completed_trades.append(self._complete_trade(position, df.index[-1], sell_price, comm, "end_of_data"))

        result = BacktestResult(
            trades=trades,
            completed_trades=completed_trades,
            equity_curve=pd.Series(equity, index=df.index),
            positions={},
        )
        return result

    def summary(self, result: BacktestResult, ticker: str = "TICKER") -> str:
        """Generate a human-readable summary."""
        metrics = result.metrics
        eq = result.equity_curve

        return f"""
{'='*50}
Backtest: {self.strategy.name} on {ticker}
{'='*50}
Period: {eq.index[0].strftime('%Y-%m-%d')} → {eq.index[-1].strftime('%Y-%m-%d')}
Initial Capital: ${self.initial_capital:,.2f}
Final Equity: ${metrics['final_equity']:,.2f}
Total Return: {metrics['total_return']:+.2%}
Annualized Return: {metrics['annualized_return']:+.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2%}
Completed Trades: {metrics['completed_trades']} | Win Rate: {metrics['win_rate']:.2%}
Profit Factor: {metrics['profit_factor']:.2f} | Expectancy: ${metrics['expectancy']:,.2f}
Avg Hold: {metrics['average_hold_days']:.1f} days | Exposure: {metrics['exposure_ratio']:.2%}
Total Commission: ${metrics['total_commission']:,.2f}
{'='*50}
"""
