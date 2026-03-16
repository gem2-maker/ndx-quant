"""Compare TrendFollowing V1 vs V2 on QQQ."""
import sys, os
sys.path.insert(0, r"D:\openclaw\workspace\ndx-quant")

from data.fetcher import DataFetcher
from strategies.trend_following import TrendFollowingStrategy, TrendFollowingEngine
from strategies.trend_following import TrendFollowingStrategy, TrendFollowingEngine
from strategies.trend_following_v2 import TrendFollowingV2, TrendFollowingV2Engine
from strategies.trend_following_v3 import TrendFollowingV3, TrendFollowingV3Engine
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKER = "QQQ"
YEARS = 5  # 5 years of data for meaningful comparison

print(f"Fetching {YEARS}y of {TICKER} data...")
fetcher = DataFetcher()
df = fetcher.fetch(
    ticker=TICKER,
    period="max",
    interval="1d",
)
# Trim to last N years
if len(df) > 252 * YEARS:
    df = df.iloc[-252 * YEARS:]
print(f"Data: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} bars)")

# === V1: Original TrendFollowing ===
print("\n" + "="*60)
print("Running V1 (SMA50/200 + ADX + Volume + Pullback)...")
v1_strategy = TrendFollowingStrategy(
    sma_fast=50, sma_slow=200,
    adx_threshold=25.0, adx_exit_threshold=20.0,
    volume_factor=1.0, pullback_pct=0.03,
    trailing_stop_pct=0.12,
)
v1_engine = TrendFollowingEngine(
    strategy=v1_strategy,
    initial_capital=100_000,
    commission=0.001, slippage=0.0005,
    position_pct=0.95,
)
v1_result = v1_engine.run(df, ticker=TICKER)
print(v1_engine.summary(v1_result, TICKER))

# === V2: Original Multi-Horizon Momentum (reference) ===
print("="*60)
print("Running V2 (Multi-Horizon Momentum V1)...")
v2_strategy = TrendFollowingV2(
    sma_fast=50, sma_slow=200,
    momentum_lookbacks=(63, 126),  # Drop 252, too slow
    momentum_skip_days=10,
    sma_weight=0.6, momentum_weight=0.2, adx_weight=0.2,
    entry_threshold=0.08, exit_threshold=-0.05,
    adx_threshold=20.0,
    vol_threshold=0.015,
    rebalance_mode="monthly",
    trailing_stop_pct=0.08,
    signal_lag=1,
)
v2_engine = TrendFollowingV2Engine(
    strategy=v2_strategy,
    initial_capital=100_000,
    commission=0.001, slippage=0.0005,
    position_pct=0.95,
)
v2_result = v2_engine.run(df, ticker=TICKER)
print(v2_engine.summary(v2_result, TICKER))

# === V3: Improved (SMA+ADX+Momentum+Monthly Entry) ===
print("="*60)
print("Running V3 (SMA50/200 + ADX + 63d Momentum + Monthly Entry)...")
v3_strategy = TrendFollowingV3(
    sma_fast=50, sma_slow=200,
    adx_entry=21.0, adx_exit=17.0, adx_period=14,
    momentum_period=63,
    trailing_stop_pct=0.10,
    min_bars_between_trades=10,
    rebalance_monthly=True,
)
v3_engine = TrendFollowingV3Engine(
    strategy=v3_strategy,
    initial_capital=100_000,
    commission=0.001, slippage=0.0005,
    position_pct=0.95,
)
v3_result = v3_engine.run(df, ticker=TICKER)
print(v3_engine.summary(v3_result, TICKER))

# === Comparison Chart ===
print("\nGenerating comparison chart...")
fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})

# Plot 1: Equity curves + buy & hold
ax1 = axes[0]
v1_eq = v1_result["equity_curve"]
v2_eq = v2_result["equity_curve"]
bhp = (df["Close"] / df["Close"].iloc[0]) * 100_000

ax1.plot(v1_eq.index, v1_eq.values, label="V1 (SMA+ADX+Vol)", linewidth=1.5, color="blue")
ax1.plot(v2_eq.index, v2_eq.values, label="V2 (Multi-Mom+SMA+ADX)", linewidth=1.5, color="red")
ax1.plot(bhp.index, bhp.values, label="Buy & Hold QQQ", linewidth=1, color="gray", alpha=0.7, linestyle="--")
ax1.set_title(f"Trend Following V1 vs V2 on {TICKER} ({YEARS}y)", fontsize=14)
ax1.set_ylabel("Portfolio Value ($)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

# Plot 2: Combined signal (V2)
ax2 = axes[1]
if v2_result.get("signal_series") is not None:
    sig = v2_result["signal_series"]
    ax2.plot(sig.index, sig.values, color="purple", linewidth=0.8, alpha=0.8)
    ax2.axhline(y=0.15, color="green", linestyle="--", alpha=0.5, label="Entry threshold")
    ax2.axhline(y=-0.10, color="red", linestyle="--", alpha=0.5, label="Exit threshold")
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax2.fill_between(sig.index, 0, sig.values, where=sig.values > 0, alpha=0.2, color="green")
    ax2.fill_between(sig.index, 0, sig.values, where=sig.values < 0, alpha=0.2, color="red")
    ax2.set_ylabel("V2 Combined Signal")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

# Plot 3: QQQ price
ax3 = axes[2]
ax3.plot(df.index, df["Close"], color="black", linewidth=1)
ax3.set_ylabel(f"{TICKER} Price ($)")
ax3.set_xlabel("Date")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = r"D:\openclaw\workspace\ndx-quant\trend_v1_v2_comparison.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"\nChart saved: {chart_path}")

# Print summary comparison
print("\n" + "="*60)
print("SIDE-BY-SIDE COMPARISON")
print("="*60)
print(f"{'Metric':<25} {'V1':>12} {'V2':>12} {'V3':>12} {'Buy&Hold':>12}")
print("-"*60)
v1m = v1_result["metrics"]
v2m = v2_result["metrics"]
v3m = v3_result["metrics"]
bhp_return = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
print(f"{'Total Return':<25} {v1m['total_return']:>+11.2%} {v2m['total_return']:>+11.2%} {v3m['total_return']:>+11.2%} {bhp_return:>+11.2%}")
print(f"{'Annualized Return':<25} {v1m['annualized_return']:>+11.2%} {v2m['annualized_return']:>+11.2%} {v3m['annualized_return']:>+11.2%} {((1+bhp_return)**(252/len(df))-1):>+11.2%}")
print(f"{'Sharpe Ratio':<25} {v1m['sharpe_ratio']:>11.2f} {v2m['sharpe_ratio']:>11.2f} {v3m['sharpe_ratio']:>11.2f} {'N/A':>12}")
print(f"{'Max Drawdown':<25} {v1m['max_drawdown']:>11.2%} {v2m['max_drawdown']:>11.2%} {v3m['max_drawdown']:>11.2%} {'N/A':>12}")
print(f"{'Calmar Ratio':<25} {v1m['calmar_ratio']:>11.2f} {v2m['calmar_ratio']:>11.2f} {v3m['calmar_ratio']:>11.2f} {'N/A':>12}")
print(f"{'Round Trips':<25} {v1m['round_trips']:>11d} {v2m['round_trips']:>11d} {v3m['round_trips']:>11d} {'0':>12}")
if 'win_rate' in v2m:
    print(f"{'Win Rate':<25} {'N/A':>12} {v2m.get('win_rate',0):>11.1%} {v3m.get('win_rate',0):>11.1%} {'N/A':>12}")
    print(f"{'Profit Factor':<25} {'N/A':>12} {v2m.get('profit_factor',0):>11.2f} {v3m.get('profit_factor',0):>11.2f} {'N/A':>12}")
    print(f"{'Exposure Ratio':<25} {'N/A':>12} {v2m.get('exposure_ratio',0):>11.1%} {v3m.get('exposure_ratio',0):>11.1%} {'N/A':>12}")
print(f"{'Commission':<25} ${v1m['total_commission']:>10,.0f} ${v2m['total_commission']:>10,.0f} ${v3m['total_commission']:>10,.0f} {'$0':>12}")
