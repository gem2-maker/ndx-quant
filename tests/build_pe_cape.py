"""Build Nasdaq-100 PE from Shiller CAPE data + Nasdaq premium."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import xlrd
import pandas as pd
import numpy as np

wb = xlrd.open_workbook('cache/shiller_ie_data.xls')
ws = wb.sheet_by_index(1)

# Parse Shiller data (skip header rows 0-7)
data = []
for r in range(8, ws.nrows):
    date_val = ws.cell_value(r, 0)
    price = ws.cell_value(r, 1)
    earnings = ws.cell_value(r, 3)
    cape = ws.cell_value(r, 12)
    
    if not date_val or date_val == '':
        continue
    
    try:
        year_frac = float(date_val)
        year = int(year_frac)
        month = int(round((year_frac - year) * 12 + 1))
        if month > 12:
            month = 12
        date_str = f"{year}-{month:02d}-01"
    except:
        continue
    
    # CAPE might be 'NA'
    cape_val = None
    if cape != 'NA' and cape != '':
        try:
            cape_val = float(cape)
        except:
            pass
    
    # Regular PE = Price / Earnings
    pe_val = None
    if price and earnings:
        try:
            pe_val = float(price) / float(earnings)
        except:
            pass
    
    data.append({
        'date': date_str,
        'sp_price': price,
        'sp_earnings': earnings,
        'sp_pe': pe_val,
        'cape': cape_val,
    })

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
print(f"Shiller data: {len(df)} months, {df['date'].iloc[0].strftime('%Y-%m')} to {df['date'].iloc[-1].strftime('%Y-%m')}")

# Filter to 2021-2026
recent = df[df['date'] >= '2020-01-01'].copy()
print(f"\nRecent S&P 500 PE and CAPE (2020+):")
for _, row in recent.iterrows():
    pe = f"{row['sp_pe']:.1f}" if row['sp_pe'] else 'N/A'
    cape = f"{row['cape']:.1f}" if row['cape'] else 'N/A'
    print(f"  {row['date'].strftime('%Y-%m')}: PE={pe}, CAPE={cape}")

# Nasdaq-100 PE ≈ CAPE * dynamic_premium
# CAPE is more stable than PE, better for estimating
def nasdaq_premium_from_cape(cape):
    """Higher CAPE = lower premium"""
    if cape is None:
        return 1.2
    if cape > 35:
        return 1.05
    elif cape > 30:
        return 1.10
    elif cape > 25:
        return 1.20
    elif cape > 20:
        return 1.30
    else:
        return 1.40

print(f"\nEstimated QQQ PE from CAPE:")
for _, row in recent.iterrows():
    cape = row['cape']
    if cape:
        premium = nasdaq_premium_from_cape(cape)
        qqq_pe = cape * premium
        print(f"  {row['date'].strftime('%Y-%m')}: CAPE={cape:.1f} x {premium:.2f} = QQQ PE={qqq_pe:.1f}")

# Build daily series
known_dates = df[df['cape'].notna()]['date'].tolist()
known_pes = [row['cape'] * nasdaq_premium_from_cape(row['cape']) 
             for _, row in df[df['cape'].notna()].iterrows()]

# Add current known point
known_dates.append(pd.Timestamp('2026-03-15'))
known_pes.append(32.51)

# Filter to backtest range
start = pd.Timestamp('2021-03-15')
end = pd.Timestamp('2026-03-15')
filtered_dates = []
filtered_pes = []
for d, pe in zip(known_dates, known_pes):
    if d >= start - pd.Timedelta(days=60):  # a bit before for interpolation
        filtered_dates.append(d)
        filtered_pes.append(pe)

full_range = pd.date_range(start=start, end=end, freq='D')
known_series = pd.Series(filtered_pes, index=filtered_dates).sort_index()
known_series = known_series[~known_series.index.duplicated(keep='last')]
daily_pe = known_series.reindex(full_range).interpolate(method='linear').ffill().bfill()

pe_df = pd.DataFrame({'implied_pe': daily_pe})
pe_df.index.name = 'Date'

print(f"\nFinal: {len(pe_df)} bars, range {pe_df['implied_pe'].min():.1f}-{pe_df['implied_pe'].max():.1f}")
cheap = (pe_df['implied_pe'] < 25).sum()
normal = pe_df['implied_pe'].between(25, 32).sum()
expensive = (pe_df['implied_pe'] > 32).sum()
print(f"  Cheap(<25): {cheap}, Normal(25-32): {normal}, Expensive(>32): {expensive}")

pe_df.to_csv('cache/qqq_implied_pe_v4.csv')
print("Saved to cache/qqq_implied_pe_v4.csv")
