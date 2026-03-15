"""V3 backtest with ML - with error traceback."""
import sys, traceback, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from data.fetcher import DataFetcher
from indicators.technical import add_all_indicators
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine

fetcher = DataFetcher()
df = fetcher.fetch("QQQ", period="2y")
df = add_all_indicators(df)
print(f"Loaded {len(df)} bars")

strategy = LivermoreV3Strategy()
engine = LivermoreV3Engine(strategy, initial_capital=100000)

try:
    result = engine.run(df, ticker="QQQ")
    print(engine.summary(result, "QQQ"))
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
