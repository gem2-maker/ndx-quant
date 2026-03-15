from data.fetcher import DataFetcher
from strategies.livermore_v3 import LivermoreV3Strategy, LivermoreV3Engine
import warnings; warnings.filterwarnings('ignore')
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
fetcher = DataFetcher()
df = fetcher.fetch('QQQ', period='2y', interval='1d')
strat = LivermoreV3Strategy(min_composite_score=0.35, macro_weight=0.2, dynamic_weighting=True)
engine = LivermoreV3Engine(strat, pyramid_profit_trigger=0.03, max_position_pct=0.60, pyramid_min_score=0.50)
result = engine.run(df, ticker='QQQ')
m = result['metrics']
tr = m.get('total_return', 0)
sh = m.get('sharpe_ratio', 0)
dd = m.get('max_drawdown', 0)
print(f'Return: {tr:.2%}, Sharpe: {sh:.2f}, MaxDD: {dd:.2%}')
trades = [t for t in result['trades'] if t['action'] in ('BUY','PYRAMID')]
print(f'Entries: {len(trades)}')
for t in result['trades']:
    print(f'  {t["date"].strftime("%Y-%m-%d")} {t["action"]:6s} comp={t.get("composite","?")} v={t.get("value","?")} m={t.get("momentum","?")} ml={t.get("ml","?")} mac={t.get("macro","?")} w={t.get("weights",{})}')
print('QQQ benchmark: 36.69%')
