"""Quick test of Livermore V3 with ML scoring."""
import sys
sys.path.insert(0, ".")

from data.fetcher import DataFetcher
from indicators.technical import add_all_indicators
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine

fetcher = DataFetcher()
df = fetcher.fetch("QQQ", period="2y")
df = add_all_indicators(df)
print(f"Loaded {len(df)} bars")

strategy = LivermoreV3Strategy()
engine = LivermoreV3Engine(strategy, initial_capital=100000)
result = engine.run(df, ticker="QQQ")
print(engine.summary(result, "QQQ"))

# Show a few trades with ML scores
trades = [t for t in result["trades"] if t["action"] in ("BUY", "PYRAMID")]
print("\nSample trades with ML scores:")
for t in trades[:5]:
    print(f"  {t['date'].strftime('%Y-%m-%d')} {t['action']}: "
          f"value={t.get('value','?')} momentum={t.get('momentum','?')} "
          f"ml={t.get('ml','?')} composite={t.get('composite','?')}")
