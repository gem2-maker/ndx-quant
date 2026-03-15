"""Test Livermore v3 — Value + Momentum composite."""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from data.fetcher import DataFetcher
from indicators.technical import add_all_indicators
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine
from strategies.livermore_v2 import LivermoreV2Strategy, LivermoreV2Engine
from strategies.momentum import MomentumStrategy
from backtest.engine import BacktestEngine


def main():
    fetcher = DataFetcher()

    print('Fetching QQQ (5y)...')
    df = fetcher.fetch('QQQ', period='5y', interval='1d')
    df = add_all_indicators(df)

    print('Fetching VIX...')
    vix_df = fetcher.fetch('^VIX', period='5y', interval='1d')
    if not vix_df.empty:
        vix_close = vix_df['Close'].rename('VIX')
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        vix_close.index = vix_close.index.tz_localize(None) if vix_close.index.tz else vix_close.index
        df = df.join(vix_close, how='left')
        df['VIX'] = df['VIX'].ffill()
        print(f'VIX: {(df["VIX"]>30).sum()} fear days')

    # Load implied PE
    pe_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', 'qqq_implied_pe_v2.csv')
    if os.path.exists(pe_file):
        pe_df = pd.read_csv(pe_file, index_col=0, parse_dates=True)
        pe_df.index = pd.to_datetime(pe_df.index)
        try:
            pe_df.index = pe_df.index.tz_localize(None)
        except:
            pass
        pe_series = pe_df['implied_pe'].rename('implied_pe')
        df = df.join(pe_series, how='left')
        df['implied_pe'] = df['implied_pe'].ffill()
        print(f'PE loaded: {df["implied_pe"].notna().sum()} bars, range {df["implied_pe"].min():.1f} - {df["implied_pe"].max():.1f}')

    df = df.dropna(subset=['Close'])
    bh = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1

    print(f'\nQQQ: {len(df)} bars, Buy&Hold: {bh:+.2%}\n')

    # === V3 ===
    print('Running V3 (Value + Momentum)...')
    v3_strat = LivermoreV3Strategy(
        lookback=60, fib_level=0.5, max_pyramids=3, trend_sma=200,
        vix_fear=30.0, vix_extreme=40.0, breakout_lookback=20,
        value_weight=0.4, momentum_weight=0.6,
        min_composite_score=0.55, pe_cheap=25.0, pe_expensive=35.0,
    )
    v3_engine = LivermoreV3Engine(
        strategy=v3_strat, initial_capital=100000,
        commission=0.001, slippage=0.0005,
        max_position_pct=0.30, pyramid_profit_trigger=0.05,
        pyramid_min_score=0.60,
    )
    v3_result = v3_engine.run(df, 'QQQ')
    print(v3_engine.summary(v3_result, 'QQQ'))

    # Trade log
    print('\nTRADE LOG:')
    print('-' * 75)
    for t in v3_result['trades']:
        d = t['date'].strftime('%Y-%m-%d')
        r = t.get('reason', '')
        if t['action'] == 'BUY':
            v = t.get('value', '')
            m = t.get('momentum', '')
            c = t.get('composite', '')
            print(f'  BUY     {d}  {t["shares"]:>5} @ ${t["price"]:.2f}  ({r} V:{v} M:{m} C:{c})')
        elif t['action'] == 'PYRAMID':
            print(f'  PYRAMID {d}  +{t["shares"]:>4} @ ${t["price"]:.2f}  avg=${t["avg_entry"]:.2f} tot={t["total_shares"]} C:{t.get("composite","")}')
        elif t['action'] == 'SELL':
            dd = t.get('drawdown', '')
            print(f'  SELL    {d}  {t["shares"]:>5} @ ${t["price"]:.2f}  ({r} {dd})')

    # === Compare all ===
    print('\n\n' + '='*65)
    print('FULL COMPARISON (5-year QQQ)')
    print('='*65)
    print(f'{"Strategy":<35} {"Return":>10} {"Sharpe":>8} {"MaxDD":>10}')
    print('-'*65)
    print(f'{"Buy & Hold":<35} {bh:>+10.2%} {"N/A":>8} {"N/A":>10}')

    mom = MomentumStrategy()
    mom_r = BacktestEngine(mom).run(df, 'QQQ')
    mm = mom_r.metrics
    print(f'{"Momentum SMA20/50":<35} {mm["total_return"]:>+10.2%} {mm["sharpe_ratio"]:>8.2f} {mm["max_drawdown"]:>10.2%}')

    # V2 for comparison
    v2_strat = LivermoreV2Strategy(lookback=60, fib_level=0.5, max_pyramids=3, trend_sma=200, vix_fear_threshold=30.0, vix_extreme_fear=40.0, breakout_lookback=20)
    v2_eng = LivermoreV2Engine(strategy=v2_strat, initial_capital=100000, commission=0.001, slippage=0.0005, max_position_pct=0.25, pyramid_profit_trigger=0.05, fear_position_scale=0.4)
    v2_r = v2_eng.run(df, 'QQQ')
    v2m = v2_r['metrics']
    print(f'{"V2 (Graduated Stop)":<35} {v2m["total_return"]:>+10.2%} {v2m["sharpe_ratio"]:>8.2f} {v2m["max_drawdown"]:>10.2%}')

    v3m = v3_result['metrics']
    print(f'{"V3 (Value+Momentum)":<35} {v3m["total_return"]:>+10.2%} {v3m["sharpe_ratio"]:>8.2f} {v3m["max_drawdown"]:>10.2%}')
    print('='*65)
    print('Done!')


if __name__ == '__main__':
    main()
