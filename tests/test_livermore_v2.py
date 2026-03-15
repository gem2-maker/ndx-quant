"""Test Livermore v2 with graduated stop loss + Fear & Greed."""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from data.fetcher import DataFetcher
from indicators.technical import add_all_indicators
from strategies.livermore_v2 import LivermoreV2Strategy, LivermoreV2Engine
from strategies.livermore import LivermoreStrategy, LivermoreEngine
from strategies.momentum import MomentumStrategy
from backtest.engine import BacktestEngine


def main():
    fetcher = DataFetcher()

    print('Fetching QQQ (5y)...')
    df = fetcher.fetch('QQQ', period='5y', interval='1d')
    df = add_all_indicators(df)

    print('Fetching VIX (^VIX)...')
    vix_df = fetcher.fetch('^VIX', period='5y', interval='1d')

    # Merge VIX into QQQ dataframe — normalize timezones first
    if not vix_df.empty:
        vix_close = vix_df['Close'].rename('VIX')
        # Strip timezone info so indices match
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        vix_close.index = vix_close.index.tz_localize(None) if vix_close.index.tz else vix_close.index
        df = df.join(vix_close, how='left')
        df['VIX'] = df['VIX'].ffill()
        print(f'VIX merged: {df["VIX"].notna().sum()} bars with VIX data')
        print(f'VIX range: {df["VIX"].min():.1f} - {df["VIX"].max():.1f}')
        fear_days = (df['VIX'] > 30).sum()
        extreme_fear_days = (df['VIX'] > 40).sum()
        print(f'Fear days (VIX>30): {fear_days} ({fear_days/len(df):.1%})')
        print(f'Extreme fear days (VIX>40): {extreme_fear_days}')
    else:
        print('WARNING: No VIX data, running without Fear&Greed')

    print(f'\nQQQ: {len(df)} bars, {df.index[0].strftime("%Y-%m-%d")} -> {df.index[-1].strftime("%Y-%m-%d")}')
    bh = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    print(f'Buy & Hold: {bh:+.2%}\n')

    # === Livermore v2 ===
    print('='*55)
    print('LIVERMORE V2: Graduated Stop Loss + Fear & Greed')
    print('='*55)

    strategy = LivermoreV2Strategy(
        lookback=60,
        fib_level=0.5,
        max_pyramids=3,
        trend_sma=200,
        vix_fear_threshold=30.0,
        vix_extreme_fear=40.0,
        breakout_lookback=20,
    )
    engine = LivermoreV2Engine(
        strategy=strategy,
        initial_capital=100_000,
        commission=0.001,
        slippage=0.0005,
        max_position_pct=0.25,
        pyramid_profit_trigger=0.05,
        fear_position_scale=0.4,
    )
    v2_result = engine.run(df, 'QQQ')
    print(engine.summary(v2_result, 'QQQ'))

    # Trade log
    print('\nTRADE LOG:')
    print('-'*70)
    for t in v2_result['trades']:
        d = t['date'].strftime('%Y-%m-%d')
        reason = t.get('reason', '')
        if t['action'] == 'BUY':
            print(f'  BUY      {d}  {t["shares"]:>5} @ ${t["price"]:.2f}  ({reason})')
        elif t['action'] == 'PYRAMID':
            print(f'  PYRAMID  {d}  +{t["shares"]:>4} @ ${t["price"]:.2f}  avg=${t["avg_entry"]:.2f}  total={t["total_shares"]}')
        elif t['action'] == 'SELL':
            dd = t.get('drawdown', '')
            print(f'  SELL     {d}  {t["shares"]:>5} @ ${t["price"]:.2f}  ({reason} {dd})')

    # === Compare all strategies ===
    print('\n\n' + '='*60)
    print('FULL COMPARISON')
    print('='*60)
    print(f'{"Strategy":<30} {"Return":>10} {"Sharpe":>8} {"MaxDD":>10}')
    print('-'*60)

    # Buy & Hold
    print(f'{"Buy & Hold":<30} {bh:>+10.2%} {"N/A":>8} {"N/A":>10}')

    # Momentum
    mom = MomentumStrategy()
    mom_engine = BacktestEngine(mom)
    mom_result = mom_engine.run(df, 'QQQ')
    mm = mom_result.metrics
    print(f'{"Momentum (SMA20/50)":<30} {mm["total_return"]:>+10.2%} {mm["sharpe_ratio"]:>8.2f} {mm["max_drawdown"]:>10.2%}')

    # Livermore v1
    lm1 = LivermoreStrategy(lookback=60, fib_level=0.618, stop_loss_pct=0.065, max_pyramids=3, trend_sma=200)
    lm1_engine = LivermoreEngine(strategy=lm1, initial_capital=100000, commission=0.001, slippage=0.0005, max_position_pct=0.25, pyramid_profit_trigger=0.05)
    v1_result = lm1_engine.run(df, 'QQQ')
    v1m = v1_result['metrics']
    print(f'{"Livermore v1 (6.5% stop)":<30} {v1m["total_return"]:>+10.2%} {v1m["sharpe_ratio"]:>8.2f} {v1m["max_drawdown"]:>10.2%}')

    # Livermore v2
    v2m = v2_result['metrics']
    print(f'{"Livermore V2 (graduated)":<30} {v2m["total_return"]:>+10.2%} {v2m["sharpe_ratio"]:>8.2f} {v2m["max_drawdown"]:>10.2%}')

    print('='*60)
    print('Done!')


if __name__ == '__main__':
    main()
