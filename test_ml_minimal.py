"""Minimal ML scoring test."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from data.fetcher import DataFetcher
from indicators.technical import add_all_indicators
from strategies.livermore_v3 import LivermoreV3Strategy

fetcher = DataFetcher()
df = fetcher.fetch("QQQ", period="6mo")
df = add_all_indicators(df)
print(f"Loaded {len(df)} bars")

strategy = LivermoreV3Strategy()
# Test value/momentum scores
v = strategy._value_score(df, len(df)-1)
m = strategy._momentum_score(df, len(df)-1)
print(f"Value={v:.3f}  Momentum={m:.3f}")

# Test ML score
ml = strategy._ml_score(df, len(df)-1)
print(f"ML={ml:.3f}")

# Test signal
sig = strategy.generate_signal(df, len(df)-1)
print(f"Signal={sig}")

# Test composite
c = 0.3*v + 0.4*m + 0.3*ml
print(f"Composite={c:.3f}")
print("OK - all scores work")
