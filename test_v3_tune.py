"""Compare multiple V3 parameter sets to find higher returns."""
import sys, io
sys.path.insert(0, r"D:\openclaw\workspace\ndx-quant")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data.fetcher import DataFetcher
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

fetcher = DataFetcher(cache_dir="cache", use_sqlite=True)
df = fetcher.fetch("QQQ", period="2y", interval="1d")
print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

# Pre-train ML model
from ml.predictor import TrendPredictor
import pickle
from pathlib import Path
predictor = TrendPredictor(model_type="random_forest", n_estimators=100, max_depth=5)
result = predictor.train(df)
model_path = Path("models/QQQ_random_forest.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump({"model": predictor.model, "scaler": predictor.scaler, "feature_names": predictor.feature_names}, f)
print(f"ML model trained (train={result.train_accuracy:.2%} test={result.test_accuracy:.2%})\n")

# Format: (name, strat_params, engine_params)
configs = [
    ("baseline", {"min_composite_score": 0.50}, {"pyramid_profit_trigger": 0.05, "max_position_pct": 0.30, "pyramid_min_score": 0.60}),
    ("low_entry", {"min_composite_score": 0.35}, {"pyramid_profit_trigger": 0.05, "max_position_pct": 0.30, "pyramid_min_score": 0.60}),
    ("big_pos", {"min_composite_score": 0.40}, {"pyramid_profit_trigger": 0.03, "max_position_pct": 0.60, "pyramid_min_score": 0.50}),
    ("ultra", {"min_composite_score": 0.30}, {"pyramid_profit_trigger": 0.02, "max_position_pct": 0.80, "pyramid_min_score": 0.40}),
    ("mid", {"min_composite_score": 0.38, "max_pyramids": 5}, {"pyramid_profit_trigger": 0.02, "max_position_pct": 0.50, "pyramid_min_score": 0.45}),
    ("ultra2", {"min_composite_score": 0.25, "max_pyramids": 6}, {"pyramid_profit_trigger": 0.015, "max_position_pct": 0.90, "pyramid_min_score": 0.35}),
]

results = []
for name, strat_kw, engine_kw in configs:
    strat = LivermoreV3Strategy(**strat_kw)
    engine = LivermoreV3Engine(strat, **engine_kw)
    result = engine.run(df, ticker="QQQ")
    m = result["metrics"]
    trades = result["trades"]
    buys = [t for t in trades if t["action"] in ("BUY", "PYRAMID")]
    sells = [t for t in trades if t["action"] == "SELL"]
    
    tr = m.get("total_return", 0)
    sharpe = m.get("sharpe_ratio", 0)
    mdd = m.get("max_drawdown", 0)
    
    results.append({
        "config": name,
        "total_return": tr,
        "annual_return": m.get("annual_return", 0),
        "sharpe": sharpe,
        "max_dd": mdd,
        "volatility": m.get("volatility", 0),
        "num_buys": len(buys),
        "final": result["equity_curve"].iloc[-1],
    })
    
    status = ">>" if tr > 0.15 else "  "
    print(f"{status} {name:12s}: ret={tr:+.2%} ann={m.get('annual_return',0):+.2%} sharpe={sharpe:.2f} mdd={mdd:.2%} buys={len(buys)} final=${result['equity_curve'].iloc[-1]:,.0f}")

print("\n" + "="*70)
best = max(results, key=lambda x: x["total_return"])
print(f"Best: {best['config']} → {best['total_return']:+.2%}, Sharpe {best['sharpe']:.2f}, MaxDD {best['max_dd']:.2%}")
