"""Build QQQ PE from S&P 500 PE + current Nasdaq-100 premium."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

# Load S&P 500 PE (157 years of data from multpl.com)
sp_pe = pd.read_csv('cache/sp500_pe_history.csv')
sp_pe['Date_parsed'] = pd.to_datetime(sp_pe['Date'], errors='coerce')
sp_pe = sp_pe.dropna(subset=['Date_parsed']).sort_values('Date_parsed')

# Current relationship
current_sp_pe = 28.50
current_qqq_pe = 32.51
premium = current_qqq_pe / current_sp_pe  # 1.14x

print(f"Current S&P 500 PE: {current_sp_pe}")
print(f"Current QQQ PE: {current_qqq_pe}")
print(f"Nasdaq-100 premium: {premium:.2f}x")
print(f"\nS&P 500 PE data: {len(sp_pe)} points, {sp_pe['Date_parsed'].iloc[0].year}-{sp_pe['Date_parsed'].iloc[-1].year}")

# Historical Nasdaq premium varies:
# - 2000 dot-com: Nasdaq PE was 5-8x S&P (extreme)
# - 2010-2019: ~1.3-1.5x
# - 2020-2021 COVID boom: ~1.5-1.8x  
# - 2022 bear: ~1.1-1.3x (compressed)
# - 2023-2024 normal: ~1.2-1.4x
# Use a dynamic premium based on S&P PE level (higher S&P PE = lower premium)

def estimate_nasdaq_premium(sp_pe_val):
    """Estimate Nasdaq-100 PE premium over S&P 500.
    Higher S&P PE → lower premium (Nasdaq already stretched)
    Lower S&P PE → moderate premium (Nasdaq cheaper but still premium)
    """
    if sp_pe_val > 35:
        return 1.05  # Bubble territory, Nasdaq close to S&P
    elif sp_pe_val > 28:
        return 1.15  # High but normal
    elif sp_pe_val > 22:
        return 1.25  # Normal
    elif sp_pe_val > 18:
        return 1.35  # Cheap - Nasdaq more attractive
    else:
        return 1.45  # Very cheap

# Filter to 2021-2026 for our backtest window
recent = sp_pe[sp_pe['Date_parsed'] >= '2020-01-01'].copy()
print(f"\nRecent S&P 500 PE → QQQ PE estimate:")
for _, row in recent.iterrows():
    sp = row['SP500_PE']
    qqq_pe = sp * estimate_nasdaq_premium(sp)
    print(f"  {row['Date']}: S&P={sp:.1f} → QQQ PE={qqq_pe:.1f} (premium={estimate_nasdaq_premium(sp):.2f}x)")

# Build daily interpolation for backtest (2021-03 to 2026-03)
# Create known points
known_dates = recent['Date_parsed'].tolist()
known_pes = [row['SP500_PE'] * estimate_nasdaq_premium(row['SP500_PE']) 
             for _, row in recent.iterrows()]

# Add current known point
from datetime import datetime
known_dates.append(pd.Timestamp('2026-03-15'))
known_pes.append(32.51)

# Create daily range
full_range = pd.date_range(start='2021-03-15', end='2026-03-15', freq='D')

# Interpolate
known_series = pd.Series(known_pes, index=known_dates).sort_index()
known_series = known_series[~known_series.index.duplicated(keep='last')]
daily_pe = known_series.reindex(full_range).interpolate(method='linear').ffill().bfill()

pe_df = pd.DataFrame({'implied_pe': daily_pe})
pe_df.index.name = 'Date'

print(f"\nFinal PE series: {len(pe_df)} bars, range {pe_df['implied_pe'].min():.1f}-{pe_df['implied_pe'].max():.1f}")
print(f"Distribution: cheap(<25)={(pe_df['implied_pe']<25).sum()}, normal={(pe_df['implied_pe'].between(25,32)).sum()}, expensive(>32)={(pe_df['implied_pe']>32).sum()}")

pe_df.to_csv('cache/qqq_implied_pe_v3.csv')
print("Saved to cache/qqq_implied_pe_v3.csv")
