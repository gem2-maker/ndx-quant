"""Backtesting engine — event-driven backtester."""

import pandas as pd
from datetime import datetime

from config import INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE, MAX_POSITION_PCT, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from strategies.base import BaseStrategy, Signal, Position, Trade, BacktestResult
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

    def run(self, df: pd.DataFrame, ticker: str = "TICKER") -> BacktestResult:
        """Run backtest on a single ticker's DataFrame."""
        # Ensure indicators are present
        if "SMA_20" not in df.columns:
            df = add_all_indicators(df)

        cash = self.initial_capital
        position: Position | None = None
        trades: list[Trade] = []
        equity = []

        for i in range(len(df)):
            close = df["Close"].iloc[i]
            date = df.index[i]

            # Update position current price
            if position:
                position.current_price = close

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
                    position = None

            # Get signal
            signal = self.strategy.generate_signal(df, i)

            # Execute trades
            if signal == Signal.BUY and position is None:
                # Calculate position size
                max_value = cash * self.max_position_pct
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
                            stop_loss=self.strategy.get_stop_loss(buy_price),
                            take_profit=self.strategy.get_take_profit(buy_price),
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
                position = None

            # Record equity
            total_value = cash
            if position:
                total_value += position.shares * close
            equity.append(total_value)

        # Close any open position at end
        if position:
            close = df["Close"].iloc[-1]
            sell_price = close * (1 - self.slippage)
            comm = position.shares * sell_price * self.commission
            cash += position.shares * sell_price - comm
            trades.append(Trade(
                ticker=ticker, date=df.index[-1], action="SELL",
                shares=position.shares, price=sell_price,
                commission=comm, reason="end_of_data",
            ))

        result = BacktestResult(
            trades=trades,
            equity_curve=pd.Series(equity, index=df.index),
            positions={},
        )
        return result

    def summary(self, result: BacktestResult, ticker: str = "TICKER") -> str:
        """Generate a human-readable summary."""
        total_return = result.total_return
        eq = result.equity_curve

        # Calculate metrics
        daily_returns = eq.pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5) if daily_returns.std() > 0 else 0
        max_dd = ((eq / eq.cummax()) - 1).min()

        buys = [t for t in result.trades if t.action == "BUY"]
        sells = [t for t in result.trades if t.action == "SELL"]
        total_commission = sum(t.commission for t in result.trades)

        return f"""
{'='*50}
Backtest: {self.strategy.name} on {ticker}
{'='*50}
Period: {eq.index[0].strftime('%Y-%m-%d')} → {eq.index[-1].strftime('%Y-%m-%d')}
Initial Capital: ${self.initial_capital:,.2f}
Final Equity: ${eq.iloc[-1]:,.2f}
Total Return: {total_return:+.2%}
Sharpe Ratio: {sharpe:.2f}
Max Drawdown: {max_dd:.2%}
Total Trades: {len(buys)} buys, {len(sells)} sells
Total Commission: ${total_commission:,.2f}
{'='*50}
"""
