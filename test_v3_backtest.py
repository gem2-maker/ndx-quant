"""V3 backtest with ML, warnings suppressed."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from data.fetcher import DataFetcher
from indicators.technical import add_all_indicators
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine

fetcher = DataFetcher()
df = fetcher.fetch("QQQ", period="2y")
df = add_all_indicators(df)
print(f"Loaded {len(df)} bars: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

strategy = LivermoreV3Strategy()
engine = LivermoreV3Engine(strategy, initial_capital=100000)
result = engine.run(df, ticker="QQQ")
print(engine.summary(result, "QQQ"))

# Show sample trades with ML
trades = [t for t in result["trades"] if t["action"] in ("BUY", "PYRAMID")]
print(f"\nAll entry trades ({len(trades)}):")
for t in trades:
    print(f"  {t['date'].strftime('%Y-%m-%d')} {t['action']}: "
          f"v={t.get('value','?')} m={t.get('momentum','?')} "
          f"ml={t.get('ml','?')} c={t.get('composite','?')} reason={t.get('reason','')}")
