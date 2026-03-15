"""Backtest visualization — equity curves, drawdowns, heatmaps, trade analysis.

Generates publication-quality charts from BacktestResult objects.
All plots support saving to PNG/PDF/SVG.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from typing import Optional

from strategies.base import BacktestResult, Trade


# Default style
STYLE = {
    "bg_color": "#0d1117",
    "text_color": "#c9d1d9",
    "grid_color": "#21262d",
    "bull_color": "#3fb950",      # Green
    "bear_color": "#f85149",      # Red
    "accent_color": "#58a6ff",    # Blue
    "gold_color": "#d29922",      # Gold
    "buy_marker": "^",
    "sell_marker": "v",
}


def _apply_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Apply dark theme to axis."""
    ax.set_facecolor(STYLE["bg_color"])
    ax.figure.set_facecolor(STYLE["bg_color"])
    ax.tick_params(colors=STYLE["text_color"], labelsize=8)
    ax.spines["bottom"].set_color(STYLE["grid_color"])
    ax.spines["left"].set_color(STYLE["grid_color"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=STYLE["grid_color"], linewidth=0.5, alpha=0.5)
    if title:
        ax.set_title(title, color=STYLE["text_color"], fontsize=11, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=STYLE["text_color"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=STYLE["text_color"], fontsize=9)


class BacktestVisualizer:
    """Generate charts from backtest results."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def full_report(
        self,
        result: BacktestResult,
        ticker: str,
        df: Optional[pd.DataFrame] = None,
        save: bool = True,
    ) -> str:
        """Generate a full multi-panel report.

        Creates a 4-panel chart:
        1. Equity curve with buy/sell markers
        2. Drawdown chart
        3. Monthly returns heatmap
        4. Trade analysis (win/loss distribution)

        Args:
            result: BacktestResult from engine.run()
            ticker: Ticker symbol
            df: Optional price DataFrame (for overlay)
            save: If True, save to file

        Returns:
            Path to saved file, or empty string if not saved
        """
        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.35, wspace=0.25)

        # 1. Equity curve (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1, result, ticker, df)

        # 2. Drawdown (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_drawdown(ax2, result)

        # 3. Monthly heatmap (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_monthly_heatmap(ax3, result)

        # 4. Trade analysis (bottom, full width)
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_trade_analysis(ax4, result)

        fig.suptitle(
            f"Backtest Report: {ticker}",
            color=STYLE["text_color"],
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        if save:
            filepath = self.output_dir / f"{ticker}_backtest_report.png"
            fig.savefig(filepath, dpi=150, bbox_inches="tight",
                        facecolor=STYLE["bg_color"], edgecolor="none")
            plt.close(fig)
            return str(filepath)

        plt.close(fig)
        return ""

    def equity_curve_only(
        self,
        result: BacktestResult,
        ticker: str,
        df: Optional[pd.DataFrame] = None,
        save: bool = True,
    ) -> str:
        """Standalone equity curve chart."""
        fig, ax = plt.subplots(figsize=(14, 6))
        self._plot_equity_curve(ax, result, ticker, df)

        if save:
            filepath = self.output_dir / f"{ticker}_equity.png"
            fig.savefig(filepath, dpi=150, bbox_inches="tight",
                        facecolor=STYLE["bg_color"], edgecolor="none")
            plt.close(fig)
            return str(filepath)

        plt.close(fig)
        return ""

    def drawdown_only(self, result: BacktestResult, ticker: str, save: bool = True) -> str:
        """Standalone drawdown chart."""
        fig, ax = plt.subplots(figsize=(14, 4))
        self._plot_drawdown(ax, result)
        ax.set_title(f"Drawdown: {ticker}", color=STYLE["text_color"],
                     fontsize=11, fontweight="bold", pad=10)

        if save:
            filepath = self.output_dir / f"{ticker}_drawdown.png"
            fig.savefig(filepath, dpi=150, bbox_inches="tight",
                        facecolor=STYLE["bg_color"], edgecolor="none")
            plt.close(fig)
            return str(filepath)

        plt.close(fig)
        return ""

    def compare_strategies(
        self,
        results: dict[str, BacktestResult],
        ticker: str,
        save: bool = True,
    ) -> str:
        """Compare multiple strategy equity curves.

        Args:
            results: dict of {strategy_name: BacktestResult}
            ticker: Ticker symbol
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        _apply_style(ax, title=f"Strategy Comparison: {ticker}",
                     xlabel="Date", ylabel="Equity ($)")

        colors = [STYLE["accent_color"], STYLE["bull_color"],
                  STYLE["gold_color"], STYLE["bear_color"], "#bc8cff"]

        for i, (name, result) in enumerate(results.items()):
            eq = result.equity_curve
            norm_eq = eq / eq.iloc[0]  # Normalize to 1.0
            color = colors[i % len(colors)]
            ax.plot(norm_eq.index, norm_eq.values, label=name,
                    color=color, linewidth=1.5, alpha=0.9)

        ax.axhline(y=1.0, color=STYLE["text_color"], linewidth=0.5,
                    linestyle="--", alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, facecolor=STYLE["bg_color"],
                  edgecolor=STYLE["grid_color"], labelcolor=STYLE["text_color"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()

        if save:
            filepath = self.output_dir / f"{ticker}_compare.png"
            fig.savefig(filepath, dpi=150, bbox_inches="tight",
                        facecolor=STYLE["bg_color"], edgecolor="none")
            plt.close(fig)
            return str(filepath)

        plt.close(fig)
        return ""

    # === Internal plot methods ===

    def _plot_equity_curve(self, ax, result: BacktestResult, ticker: str, df=None):
        """Plot equity curve with buy/sell markers."""
        eq = result.equity_curve
        _apply_style(ax, title=f"Equity Curve: {ticker}",
                     xlabel="", ylabel="Equity ($)")

        # Color segments by gain/loss vs previous
        colors = np.where(eq.values >= eq.iloc[0], STYLE["bull_color"], STYLE["bear_color"])

        # Plot line with gradient
        ax.fill_between(eq.index, eq.values, eq.iloc[0],
                        where=(eq.values >= eq.iloc[0]),
                        color=STYLE["bull_color"], alpha=0.1)
        ax.fill_between(eq.index, eq.values, eq.iloc[0],
                        where=(eq.values < eq.iloc[0]),
                        color=STYLE["bear_color"], alpha=0.1)
        ax.plot(eq.index, eq.values, color=STYLE["accent_color"],
                linewidth=1.5, label="Equity")

        # Starting capital line
        ax.axhline(y=eq.iloc[0], color=STYLE["text_color"], linewidth=0.5,
                    linestyle="--", alpha=0.3, label=f"Initial: ${eq.iloc[0]:,.0f}")

        # Plot buy/sell markers
        buys = [t for t in result.trades if t.action == "BUY"]
        sells = [t for t in result.trades if t.action == "SELL"]

        if buys:
            buy_dates = [t.date for t in buys]
            # Find equity values at buy dates
            buy_equity = [eq.loc[d] if d in eq.index else eq.iloc[-1] for d in buy_dates]
            ax.scatter(buy_dates, buy_equity, marker=STYLE["buy_marker"],
                       color=STYLE["bull_color"], s=60, zorder=5, label=f"Buys ({len(buys)})")

        if sells:
            sell_dates = [t.date for t in sells]
            sell_equity = [eq.loc[d] if d in eq.index else eq.iloc[-1] for d in sell_dates]
            ax.scatter(sell_dates, sell_equity, marker=STYLE["sell_marker"],
                       color=STYLE["bear_color"], s=60, zorder=5, label=f"Sells ({len(sells)})")

        # Stats box
        total_return = result.total_return
        final_eq = eq.iloc[-1]
        textstr = f"Return: {total_return:+.1%}\nFinal: ${final_eq:,.0f}"
        props = dict(boxstyle="round,pad=0.5", facecolor=STYLE["bg_color"],
                     edgecolor=STYLE["grid_color"], alpha=0.9)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", color=STYLE["text_color"], bbox=props)

        ax.legend(loc="lower left", fontsize=8, facecolor=STYLE["bg_color"],
                  edgecolor=STYLE["grid_color"], labelcolor=STYLE["text_color"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig = ax.figure
        fig.autofmt_xdate()

    def _plot_drawdown(self, ax, result: BacktestResult):
        """Plot drawdown chart."""
        eq = result.equity_curve
        dd = (eq / eq.cummax() - 1) * 100  # Percentage drawdown

        _apply_style(ax, title="Drawdown", xlabel="", ylabel="Drawdown (%)")

        ax.fill_between(dd.index, dd.values, 0,
                        color=STYLE["bear_color"], alpha=0.3)
        ax.plot(dd.index, dd.values, color=STYLE["bear_color"],
                linewidth=1, alpha=0.8)

        # Max drawdown marker
        max_dd = dd.min()
        max_dd_date = dd.idxmin()
        ax.axhline(y=max_dd, color=STYLE["gold_color"], linewidth=0.5,
                    linestyle="--", alpha=0.5)
        ax.text(max_dd_date, max_dd, f"  Max DD: {max_dd:.1f}%",
                color=STYLE["gold_color"], fontsize=8, va="top")

        ax.set_ylim(min(max_dd * 1.2, -1), 1)

    def _plot_monthly_heatmap(self, ax, result: BacktestResult):
        """Plot monthly returns heatmap."""
        eq = result.equity_curve
        monthly = eq.resample("ME").last().pct_change() * 100

        if len(monthly) < 2:
            ax.text(0.5, 0.5, "Insufficient data\nfor monthly heatmap",
                    transform=ax.transAxes, ha="center", va="center",
                    color=STYLE["text_color"], fontsize=10)
            _apply_style(ax, title="Monthly Returns (%)")
            return

        # Create pivot: rows=year, cols=month
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = monthly_df.pivot_table(values="return", index="year",
                                        columns="month", aggfunc="first")

        month_names = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]

        # Custom colormap: red for negative, green for positive
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ["#f85149", "#0d1117", "#3fb950"]
        cmap = LinearSegmentedColormap.from_list("pnl", colors_list, N=256)

        vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 1)
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                        vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, color=STYLE["text_color"], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, color=STYLE["text_color"], fontsize=8)

        _apply_style(ax, title="Monthly Returns (%)")

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                            color="white", fontsize=7, fontweight="bold")

    def _plot_trade_analysis(self, ax, result: BacktestResult):
        """Plot trade PnL distribution and statistics."""
        _apply_style(ax, title="Trade Analysis", xlabel="Trade #", ylabel="PnL ($)")

        # Extract trade PnL (pair buys with sells)
        pnls = []
        buy_queue = []
        for t in result.trades:
            if t.action == "BUY":
                buy_queue.append(t)
            elif t.action == "SELL" and buy_queue:
                buy = buy_queue.pop(0)
                pnl = (t.price - buy.price) * t.shares - t.commission - buy.commission
                pnls.append(pnl)

        if not pnls:
            ax.text(0.5, 0.5, "No completed trades", transform=ax.transAxes,
                    ha="center", va="center", color=STYLE["text_color"], fontsize=10)
            return

        colors = [STYLE["bull_color"] if p >= 0 else STYLE["bear_color"] for p in pnls]
        bars = ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.8)

        ax.axhline(y=0, color=STYLE["text_color"], linewidth=0.5, alpha=0.3)

        # Stats
        wins = [p for p in pnls if p >= 0]
        losses = [p for p in pnls if p < 0]
        win_rate = len(wins) / len(pnls) * 100
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        total_pnl = sum(pnls)

        textstr = (f"Trades: {len(pnls)} | Win: {win_rate:.0f}%\n"
                   f"Avg Win: ${avg_win:,.0f} | Avg Loss: ${avg_loss:,.0f}\n"
                   f"Total PnL: ${total_pnl:,.0f} | "
                   f"Profit Factor: {abs(sum(wins)/sum(losses)) if losses else float('inf'):.2f}")
        props = dict(boxstyle="round,pad=0.5", facecolor=STYLE["bg_color"],
                     edgecolor=STYLE["grid_color"], alpha=0.9)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", color=STYLE["text_color"], bbox=props)
