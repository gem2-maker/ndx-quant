"""Test V3 with dynamic stop loss based on macro score."""
import sys
sys.path.insert(0, ".")

from data.fetcher import DataFetcher
from data.macro import fetch_macro_data, merge_macro, add_macro_features
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine, STOP_TIERS
import warnings
warnings.filterwarnings("ignore")

fetcher = DataFetcher()
df = fetcher.fetch("QQQ", period="2y", interval="1d")
qqq_ret = df.Close.iloc[-1] / df.Close.iloc[0] - 1

macro = fetch_macro_data(period="2y")
df_macro = merge_macro(df, macro)
df_macro = add_macro_features(df_macro)

print(f"QQQ buy&hold: {qqq_ret:.2%}")
print(f"Stop tiers: {STOP_TIERS}")
print(f"Dynamic stops: macro=1.0 -> mult=2.0x, macro=0.5 -> mult=1.0x, macro=0.0 -> mult=0.7x\n")

strat = LivermoreV3Strategy(min_composite_score=0.35, macro_weight=0.2, dynamic_weighting=True)
engine = LivermoreV3Engine(strat, pyramid_profit_trigger=0.03, max_position_pct=0.60, pyramid_min_score=0.50)
result = engine.run(df_macro, ticker="QQQ")

m = result["metrics"]
tr = m.get("total_return", 0)
sh = m.get("sharpe_ratio", 0)
dd = m.get("max_drawdown", 0)
ann = m.get("annual_return", 0)
final = result["equity_curve"].iloc[-1]

print(f"Return: {tr:.2%} (ann {ann:.2%})")
print(f"Sharpe: {sh:.2f}, MaxDD: {dd:.2%}")
print(f"Final equity: ${final:,.0f}")
print(f"\nQQQ benchmark: {qqq_ret:.2%}")
print(f"Strategy: {tr:.2%} {'BEATS' if tr > qqq_ret else 'LOSES'}")

print("\nTrades:")
for t in result["trades"]:
    date = t["date"].strftime("%Y-%m-%d")
    action = t["action"]
    reason = t.get("reason", "")
    dd_str = t.get("drawdown", "")
    comp = t.get("composite", "")
    mac = t.get("macro", "")
    shares = t.get("shares", "")
    price = t.get("price", "")
    if action == "SELL":
        print(f"  {date} {action:6s} {reason:8s} dd={dd_str}")
    else:
        print(f"  {date} {action:6s} comp={comp} mac={mac} shares={shares}")
