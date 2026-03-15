"""Test Livermore strategy vs other strategies on 5-year QQQ data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from data.fetcher import DataFetcher
from indicators.technical import add_all_indicators
from strategies.livermore import LivermoreStrategy, LivermoreEngine
from strategies.momentum import MomentumStrategy
from backtest.engine import BacktestEngine
from strategies.mean_reversion import MeanReversionStrategy


def buy_and_hold_return(df: pd.DataFrame) -> float:
    return (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1


def main():
    # Fix Windows console encoding
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("Fetching 5-year QQQ data...")
    fetcher = DataFetcher()
    df = fetcher.get_data("QQQ", period="5y", interval="1d")

    if df.empty or len(df) < 100:
        print("❌ Failed to fetch data")
        return

    df = add_all_indicators(df)
    print(f"✅ Got {len(df)} bars: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Price: ${df['Close'].iloc[0]:.2f} → ${df['Close'].iloc[-1]:.2f}")

    bh_return = buy_and_hold_return(df)
    print(f"   Buy & Hold: {bh_return:+.2%}")
    print()

    # === Livermore Strategy ===
    print("🔥 Testing Livermore Trend Strategy...")
    livermore = LivermoreStrategy(
        lookback=60,
        fib_level=0.618,
        stop_loss_pct=0.065,
        max_pyramids=3,
        trend_sma=200,
    )
    engine = LivermoreEngine(
        strategy=livermore,
        initial_capital=100_000,
        commission=0.001,
        slippage=0.0005,
        max_position_pct=0.25,
        pyramid_profit_trigger=0.05,
    )
    result = engine.run(df, ticker="QQQ")
    print(engine.summary(result, "QQQ"))

    # Show trade details
    print("\n📋 Trade Details:")
    print("─" * 80)
    for t in result["trades"]:
        action = t["action"]
        if action == "BUY":
            print(f"  📈 {t['date'].strftime('%Y-%m-%d')} BUY {t['shares']} @ ${t['price']:.2f} ({t['reason']})")
        elif action == "PYRAMID":
            print(f"  🏗️  {t['date'].strftime('%Y-%m-%d')} PYRAMID +{t['shares']} @ ${t['price']:.2f} ({t['reason']}) | avg=${t['avg_entry']:.2f} total={t['total_shares']}")
        elif action == "SELL":
            layers_info = f" [{t.get('layers', 1)} layers]" if t.get('layers', 1) > 1 else ""
            print(f"  📉 {t['date'].strftime('%Y-%m-%d')} SELL {t['shares']} @ ${t['price']:.2f} ({t['reason']}){layers_info}")

    # === Comparison ===
    print("\n\n" + "=" * 55)
    print("📊 Strategy Comparison (5-year)")
    print("=" * 55)

    # Momentum (simple SMA crossover)
    print("\n🔄 Running Momentum (SMA 20/50)...")
    momentum = MomentumStrategy()
    std_engine = BacktestEngine(momentum)
    mom_result = std_engine.run(df, "QQQ")
    mom_metrics = mom_result.metrics
    print(f"   Return: {mom_metrics['total_return']:+.2%} | Sharpe: {mom_metrics['sharpe_ratio']:.2f} | MaxDD: {mom_metrics['max_drawdown']:.2%}")

    # Livermore
    lm = result["metrics"]
    print(f"\n🔥 Livermore:")
    print(f"   Return: {lm['total_return']:+.2%} | Sharpe: {lm['sharpe_ratio']:.2f} | MaxDD: {lm['max_drawdown']:.2%}")
    print(f"   Win Rate: {lm['win_rate']:.2%} | Profit Factor: {lm['profit_factor']:.2f}")
    print(f"   Pyramids: {lm['pyramid_adds']} | Avg Layers: {lm['avg_layers']:.1f}")

    # Buy & Hold
    print(f"\n📈 Buy & Hold:")
    print(f"   Return: {bh_return:+.2%}")

    print("\n" + "=" * 55)
    print("Done! ✅")


if __name__ == "__main__":
    main()
