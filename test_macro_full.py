"""Test V3 with real macro data merged into equity DataFrame."""
import sys, io
sys.path.insert(0, ".")
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from data.fetcher import DataFetcher
from data.macro import fetch_macro_data, merge_macro, add_macro_features
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine
import warnings
warnings.filterwarnings("ignore")

# 1. Fetch equity data
fetcher = DataFetcher()
df = fetcher.fetch("QQQ", period="2y", interval="1d")
qqq_ret = df.Close.iloc[-1] / df.Close.iloc[0] - 1
print(f"QQQ: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")
print(f"QQQ buy&hold: {qqq_ret:.2%}\n")

# 2. Fetch and merge macro data
print("Fetching macro data...")
macro = fetch_macro_data(period="2y")
df_macro = merge_macro(df, macro)
df_macro = add_macro_features(df_macro)
macro_cols = [c for c in df_macro.columns if c not in df.columns]
print(f"Added macro columns: {macro_cols}")
print(f"Latest macro: TNX={df_macro['TNX'].iloc[-1]:.2f} DXY={df_macro['DXY'].iloc[-1]:.1f} GLD={df_macro['GLD'].iloc[-1]:.1f}")

# Show macro at key dates (2025-03 crash, 2025-04 recovery)
print("\nMacro at key dates:")
for date_str in ["2025-01-03", "2025-03-06", "2025-04-04", "2025-04-23"]:
    try:
        row = df_macro.loc[date_str]
        print(f"  {date_str}: TNX={row['TNX']:.2f} DXY={row['DXY']:.1f} GLD={row['GLD']:.1f} VIX={row.get('VIX','N/A')}")
    except:
        pass

# 3. Run backtest with macro enabled
print("\n--- Running V3 + Macro + Dynamic Weights ---")
strat = LivermoreV3Strategy(min_composite_score=0.35, macro_weight=0.2, dynamic_weighting=True)
engine = LivermoreV3Engine(strat, pyramid_profit_trigger=0.03, max_position_pct=0.60, pyramid_min_score=0.50)
result = engine.run(df_macro, ticker="QQQ")

m = result["metrics"]
tr = m.get("total_return", 0)
sh = m.get("sharpe_ratio", 0)
dd = m.get("max_drawdown", 0)
ann = m.get("annual_return", 0)
final = result["equity_curve"].iloc[-1]

print(f"\nReturn: {tr:.2%} (ann {ann:.2%})")
print(f"Sharpe: {sh:.2f}, MaxDD: {dd:.2%}")
print(f"Final equity: ${final:,.0f}")

# 4. Show trades with macro context
trades = result["trades"]
buys = [t for t in trades if t["action"] in ("BUY", "PYRAMID")]
sells = [t for t in trades if t["action"] == "SELL"]
print(f"\nTrades: {len(buys)} buys/pyramids, {len(sells)} sells")
print(f"\nQQQ benchmark: {qqq_ret:.2%}")
print(f"Strategy: {tr:.2%} {'BEATS' if tr > qqq_ret else 'LOSES'}")

print("\nDetailed trades:")
for t in trades:
    date = t["date"].strftime("%Y-%m-%d")
    action = t["action"]
    comp = t.get("composite", "?")
    v = t.get("value", "?")
    mom = t.get("momentum", "?")
    ml = t.get("ml", "?")
    mac = t.get("macro", "?")
    w = t.get("weights", {})
    reason = t.get("reason", "")
    w_str = ""
    if w:
        w_str = " w=" + "/".join(f"{k}:{v:.2f}" for k, v in w.items())
    print(f"  {date} {action:6s} comp={comp} v={v} m={mom} ml={ml} mac={mac}{w_str} ({reason})")
