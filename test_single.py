import sys, io
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data.fetcher import DataFetcher
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine
import warnings
warnings.filterwarnings("ignore")

print("Fetching data...")
fetcher = DataFetcher()
df = fetcher.fetch("QQQ", period="2y", interval="1d")
print(f"Data: {len(df)} bars")

strategy = LivermoreV3Strategy(min_composite_score=0.35)
engine = LivermoreV3Engine(strategy, pyramid_profit_trigger=0.03, max_position_pct=0.60, pyramid_min_score=0.50)

print("Running backtest...")
result = engine.run(df, ticker="QQQ")

m = result["metrics"]
trades = result["trades"]
buys = [t for t in trades if t["action"] in ("BUY", "PYRAMID")]
print(f"Total return: {m.get('total_return',0):.2%}")
print(f"Sharpe: {m.get('sharpe_ratio',0):.2f}, MaxDD: {m.get('max_drawdown',0):.2%}")
print(f"Buys: {len(buys)}, Sells: {len([t for t in trades if t['action']=='SELL'])}")
print(f"Final equity: ${result['equity_curve'].iloc[-1]:,.0f}")

print("\nTrades:")
for t in trades:
    print(f"  {t['date'].date()} {t['action']} {t.get('shares','')} @ {t.get('price','')} ({t.get('reason','')})")
