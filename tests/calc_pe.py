"""Estimate historical QQQ PE ratio from price + earnings growth model."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import yfinance as yf

qqq = yf.Ticker('QQQ')
hist = qqq.history(period='5y')
print(f'Data: {len(hist)} bars')

current_pe = 32.5
current_price = 593.72
implied_ttm_eps = current_price / current_pe
print(f'Current PE: {current_pe}')
print(f'Implied TTM EPS: ${implied_ttm_eps:.2f}')

# Nasdaq-100 earnings growth ~13% annually
growth_rate = 0.13

hist = hist.copy()
hist['years_ago'] = (hist.index[-1] - hist.index).days / 365.25
hist['implied_eps'] = implied_ttm_eps / ((1 + growth_rate) ** hist['years_ago'])
hist['implied_pe'] = hist['Close'] / hist['implied_eps']

print(f'\nImplied PE range: {hist["implied_pe"].min():.1f} - {hist["implied_pe"].max():.1f}')

# Key dates
key_dates = [
    '2021-11-01', '2022-01-03', '2022-06-01', '2022-10-01',
    '2023-01-03', '2023-06-01', '2024-01-02', '2024-06-03',
    '2025-01-02', '2025-06-02', '2026-01-02'
]
print('\nImplied PE at key dates:')
import pandas as pd
for d in key_dates:
    try:
        ts = pd.Timestamp(d, tz='America/New_York')
        idx = hist.index[hist.index >= ts]
        if len(idx) > 0:
            idx = idx[0]
            row = hist.loc[idx]
            print(f'  {idx.strftime("%Y-%m-%d")}: Price=${row["Close"]:.2f}  PE={row["implied_pe"]:.1f}')
    except Exception as e:
        print(f'  {d}: error - {e}')

# Percentiles
print(f'\nPE percentiles:')
print(f'  25th: {hist["implied_pe"].quantile(0.25):.1f}')
print(f'  50th: {hist["implied_pe"].quantile(0.50):.1f}')
print(f'  75th: {hist["implied_pe"].quantile(0.75):.1f}')

# Distribution
cheap = (hist['implied_pe'] < 25).sum()
normal = ((hist['implied_pe'] >= 25) & (hist['implied_pe'] <= 35)).sum()
expensive = (hist['implied_pe'] > 35).sum()
print(f'\nDistribution:')
print(f'  PE < 25 (cheap): {cheap} ({cheap/len(hist):.1%})')
print(f'  PE 25-35 (normal): {normal} ({normal/len(hist):.1%})')
print(f'  PE > 35 (expensive): {expensive} ({expensive/len(hist):.1%})')

# Save for strategy use
hist[['Close', 'implied_pe']].to_csv('cache/qqq_implied_pe.csv')
print('\nSaved to cache/qqq_implied_pe.csv')
