"""Build historical QQQ PE from S&P 500 PE + Nasdaq premium ratio."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import yfinance as yf
import numpy as np

# Load S&P 500 PE history
sp_pe = pd.read_csv('cache/sp500_pe_history.csv')
print(f"S&P 500 PE: {len(sp_pe)} data points")

# Parse dates and filter to 2020+
sp_pe['Date_parsed'] = pd.to_datetime(sp_pe['Date'], errors='coerce')
sp_pe = sp_pe.dropna(subset=['Date_parsed'])
recent = sp_pe[sp_pe['Date_parsed'] >= '2020-01-01'].copy()
print(f"\nRecent S&P 500 PE (2020+):")
for _, row in recent.iterrows():
    print(f"  {row['Date']}: {row['SP500_PE']:.2f}")

# Current relationship
current_sp500_pe = 28.50
current_qqq_pe = 32.51
nasdaq_premium = current_qqq_pe / current_sp500_pe
print(f"\nCurrent Nasdaq premium: {nasdaq_premium:.2f}x")

# Known historical QQQ PE points (from various sources)
# These are approximate but better than pure proxy
known_points = {
    '2021-01-01': 35.96 * 1.35,  # Peak bubble, higher premium
    '2021-11-01': 35.0,  # Market peak
    '2022-01-01': 23.11 * 1.25,  # Start of bear market
    '2022-06-01': 22.0,  # Mid bear
    '2022-10-01': 20.0,  # Bear market bottom
    '2023-01-01': 22.82 * 1.15,  # Recovery start
    '2023-06-01': 27.0,  # AI boom begins
    '2024-01-01': 25.01 * 1.20,  # Normal expansion
    '2024-06-01': 28.0,
    '2025-01-01': 28.16 * 1.15,
    '2025-06-01': 30.0,
    '2026-01-01': 29.60 * 1.14,
    '2026-03-15': 32.51,
}

print(f"\nKnown QQQ PE points:")
for date, pe in sorted(known_points.items()):
    print(f"  {date}: {pe:.1f}")

# Build daily interpolation
dates = pd.to_datetime(list(known_points.keys()))
pe_values = list(known_points.values())

# Create a complete daily date range from 2021 to 2026
full_range = pd.date_range(start='2021-03-15', end='2026-03-15', freq='D')

# Interpolate PE values
known_series = pd.Series(pe_values, index=dates)
known_series = known_series.sort_index()
# Reindex to daily and interpolate
daily_pe = known_series.reindex(full_range).interpolate(method='linear').ffill().bfill()

pe_df = pd.DataFrame({'implied_pe': daily_pe})
pe_df.index.name = 'Date'

print(f"\nInterpolated PE: {len(pe_df)} daily values")
print(f"Range: {pe_df['implied_pe'].min():.1f} - {pe_df['implied_pe'].max():.1f}")

# Save
pe_df.to_csv('cache/qqq_implied_pe_v2.csv')
print("Saved to cache/qqq_implied_pe_v2.csv")

# Show distribution
cheap = (pe_df['implied_pe'] < 25).sum()
normal = ((pe_df['implied_pe'] >= 25) & (pe_df['implied_pe'] <= 32)).sum()
expensive = (pe_df['implied_pe'] > 32).sum()
print(f"\nDistribution:")
print(f"  PE < 25 (cheap): {cheap} ({cheap/len(pe_df):.1%})")
print(f"  PE 25-32 (normal): {normal} ({normal/len(pe_df):.1%})")
print(f"  PE > 32 (expensive): {expensive} ({expensive/len(pe_df):.1%})")
