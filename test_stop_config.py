"""Test relaxed stop loss configurations to beat QQQ buy&hold."""
import sys, io
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data.fetcher import DataFetcher
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine, STOP_TIERS
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

fetcher = DataFetcher()
df = fetcher.fetch("QQQ", period="2y", interval="1d")
qqq_return = df.Close.iloc[-1] / df.Close.iloc[0] - 1
print(f"QQQ buy&hold: {qqq_return:.2%} ({df.index[0].date()} to {df.index[-1].date()})")
print(f"Original STOP_TIERS: {STOP_TIERS}\n")

# Test configurations: (name, strat_params, engine_params)
configs = [
    # Baseline
    ("baseline", {"min_composite_score": 0.35}, {"pyramid_profit_trigger": 0.03, "max_position_pct": 0.60, "pyramid_min_score": 0.50}),
    # Wider stops (doubled triggers)
    ("wider_stops", {"min_composite_score": 0.35}, {"pyramid_profit_trigger": 0.03, "max_position_pct": 0.60, "pyramid_min_score": 0.50}),
    # No stops - pure trend follow
    ("no_stops", {"min_composite_score": 0.35}, {"pyramid_profit_trigger": 0.03, "max_position_pct": 0.60, "pyramid_min_score": 0.50}),
    # Full position, no stops
    ("full_no_stops", {"min_composite_score": 0.30}, {"pyramid_profit_trigger": 0.02, "max_position_pct": 0.90, "pyramid_min_score": 0.40}),
    # Low entry, wide stops
    ("aggressive", {"min_composite_score": 0.25, "max_pyramids": 6}, {"pyramid_profit_trigger": 0.015, "max_position_pct": 0.85, "pyramid_min_score": 0.35}),
]

# Monkey-patch STOP_TIERS for testing
import strategies.livermore_v3 as v3

# Define custom stop tiers
WIDE_STOPS = [
    ("stop_1", -0.06, 0.05),
    ("stop_2", -0.10, 0.15),
    ("stop_3", -0.16, 0.25),
    ("stop_4", -0.25, 1.00),
]

NO_STOPS = [
    ("stop_1", -99.9, 0.0),
    ("stop_2", -99.9, 0.0),
    ("stop_3", -99.9, 0.0),
    ("stop_4", -99.9, 0.0),
]

stop_configs = {
    "baseline": v3.STOP_TIERS,
    "wider_stops": WIDE_STOPS,
    "no_stops": NO_STOPS,
    "full_no_stops": NO_STOPS,
    "aggressive": WIDE_STOPS,
}

results = []
for name, strat_kw, engine_kw in configs:
    # Patch stop tiers
    v3.STOP_TIERS = stop_configs[name]
    
    strat = LivermoreV3Strategy(**strat_kw)
    engine = LivermoreV3Engine(strat, **engine_kw)
    result = engine.run(df, ticker="QQQ")
    m = result["metrics"]
    trades = result["trades"]
    
    tr = m.get("total_return", 0)
    sharpe = m.get("sharpe_ratio", 0)
    mdd = m.get("max_drawdown", 0)
    ann = m.get("annual_return", 0)
    final = result["equity_curve"].iloc[-1]
    
    buys = [t for t in trades if t["action"] in ("BUY", "PYRAMID")]
    sells = [t for t in trades if t["action"] == "SELL"]
    
    beat = "BEAT" if tr > qqq_return else "LOSE"
    print(f"[{beat}] {name:18s}: {tr:+.2%} ann={ann:+.2%} sharpe={sharpe:.2f} mdd={mdd:.2%} buys={len(buys)} final=${final:,.0f}")

# Restore
v3.STOP_TIERS = STOP_TIERS
print(f"\nQQQ buy&hold: {qqq_return:.2%}")
print(f"Target: beat {qqq_return:.2%}")
