"""Quick test for Livermore strategy."""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from data.fetcher import DataFetcher
from indicators.technical import add_all_indicators
from strategies.livermore import LivermoreStrategy, LivermoreEngine
from strategies.momentum import MomentumStrategy
from backtest.engine import BacktestEngine

fetcher = DataFetcher()
df = fetcher.fetch('QQQ', period='5y', interval='1d')
df = add_all_indicators(df)

print(f'Got {len(df)} bars: {df.index[0]} to {df.index[-1]}')
bh = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
print(f'BuyHold: {bh:+.2%}')
print()

# Livermore
lm = LivermoreStrategy(lookback=60, fib_level=0.618, stop_loss_pct=0.065, max_pyramids=3, trend_sma=200)
engine = LivermoreEngine(strategy=lm, initial_capital=100000, commission=0.001, slippage=0.0005, max_position_pct=0.25, pyramid_profit_trigger=0.05)
result = engine.run(df, 'QQQ')
print(engine.summary(result, 'QQQ'))

# Trade log
print('\nTRADE LOG:')
print('-' * 70)
for t in result['trades']:
    d = t['date'].strftime('%Y-%m-%d')
    if t['action'] == 'BUY':
        print(f'  BUY      {d}  {t["shares"]:>5} @ ${t["price"]:.2f}  ({t["reason"]})')
    elif t['action'] == 'PYRAMID':
        print(f'  PYRAMID  {d}  +{t["shares"]:>4} @ ${t["price"]:.2f}  avg=${t["avg_entry"]:.2f}  total={t["total_shares"]}')
    elif t['action'] == 'SELL':
        print(f'  SELL     {d}  {t["shares"]:>5} @ ${t["price"]:.2f}  ({t["reason"]})')

# Momentum comparison
print('\n\nMOMENTUM COMPARISON:')
mom = MomentumStrategy()
std_engine = BacktestEngine(mom)
mom_result = std_engine.run(df, 'QQQ')
mm = mom_result.metrics
lm_m = result['metrics']
print(f'  Momentum:  {mm["total_return"]:+.2%}  Sharpe={mm["sharpe_ratio"]:.2f}  MaxDD={mm["max_drawdown"]:.2%}')
print(f'  Livermore: {lm_m["total_return"]:+.2%}  Sharpe={lm_m["sharpe_ratio"]:.2f}  MaxDD={lm_m["max_drawdown"]:.2%}')
print(f'  BuyHold:   {bh:+.2%}')
