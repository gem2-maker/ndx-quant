"""Use annual financials to get 3-4 years of historical EPS."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import yfinance as yf
import pandas as pd
import json

# Top holdings with weights
HOLDINGS = {
    'AAPL': 0.085, 'MSFT': 0.078, 'NVDA': 0.075, 'AMZN': 0.055,
    'META': 0.045, 'GOOGL': 0.04, 'AVGO': 0.035, 'TSLA': 0.03,
    'COST': 0.025, 'NFLX': 0.02, 'AMD': 0.02, 'GOOG': 0.035,
    'ADBE': 0.02, 'PEP': 0.015, 'LIN': 0.015, 'CSCO': 0.015,
    'TMUS': 0.015, 'QCOM': 0.015, 'INTU': 0.012, 'ISRG': 0.012,
}
total_weight = sum(HOLDINGS.values())

all_annual = {}
for ticker, weight in HOLDINGS.items():
    try:
        t = yf.Ticker(ticker)
        fs = t.financials  # Annual
        if fs is None or fs.empty:
            continue
        
        ni_row = None
        for row_name in ['Net Income From Continuing And Discontinued Operation',
                         'Net Income From Continuing Operation Net Minority Interest',
                         'Net Income']:
            if row_name in fs.index:
                ni_row = fs.loc[row_name]
                break
        if ni_row is None:
            continue
        
        info = t.info
        shares = info.get('sharesOutstanding', 0)
        if not shares:
            continue
        
        annual_data = {}
        for date, ni in ni_row.items():
            if not pd.isna(ni):
                annual_data[pd.Timestamp(date).year] = ni / shares
        
        all_annual[ticker] = {'weight': weight, 'eps_by_year': annual_data}
        print(f"{ticker}: {dict((y, f'${e:.2f}') for y, e in annual_data.items())}")
    except Exception as e:
        print(f"{ticker}: error - {e}")

# Calculate weighted EPS per year
yearly_eps = {}
for ticker, data in all_annual.items():
    weight = data['weight']
    for year, eps in data['eps_by_year'].items():
        if year not in yearly_eps:
            yearly_eps[year] = {'total': 0, 'weight': 0}
        yearly_eps[year]['total'] += eps * weight
        yearly_eps[year]['weight'] += weight

print("\n=== QQQ Aggregate Annual EPS (estimated from top holdings) ===")
for year in sorted(yearly_eps.keys()):
    d = yearly_eps[year]
    norm_eps = d['total'] / d['weight'] * total_weight
    print(f"  {year}: EPS=${norm_eps:.2f} (weight coverage: {d['weight']:.1%})")

# Now get QQQ price at year end
qqq = yf.Ticker('QQQ')
hist = qqq.history(period='10y')
hist.index = hist.index.tz_localize(None)

print("\n=== QQQ Year-end Prices ===")
for year in sorted(yearly_eps.keys()):
    try:
        year_data = hist[hist.index.year == year]
        if not year_data.empty:
            year_end_price = year_data['Close'].iloc[-1]
            if year in yearly_eps and yearly_eps[year]['weight'] > 0:
                norm_eps = yearly_eps[year]['total'] / yearly_eps[year]['weight'] * total_weight
                pe = year_end_price / norm_eps if norm_eps > 0 else 0
                print(f"  {year}: Price=${year_end_price:.2f}, EPS=${norm_eps:.2f}, PE={pe:.1f}")
    except Exception as e:
        print(f"  {year}: error - {e}")

# Also do quarterly with linear interpolation
print("\n=== Approach: Interpolate quarterly EPS ===")
# Get quarterly for recent, annual for older
quarterly_data = []
for ticker, weight in HOLDINGS.items():
    try:
        t = yf.Ticker(ticker)
        qfs = t.quarterly_financials
        if qfs is None or qfs.empty:
            continue
        ni_row = None
        for row_name in ['Net Income From Continuing And Discontinued Operation',
                         'Net Income From Continuing Operation Net Minority Interest',
                         'Net Income']:
            if row_name in qfs.index:
                ni_row = qfs.loc[row_name]
                break
        if ni_row is None:
            continue
        info = t.info
        shares = info.get('sharesOutstanding', 0)
        if not shares:
            continue
        for date, ni in ni_row.items():
            if not pd.isna(ni):
                quarterly_data.append({
                    'ticker': ticker, 'weight': weight,
                    'quarter': pd.Timestamp(date), 'eps': ni / shares
                })
    except:
        pass

if quarterly_data:
    qdf = pd.DataFrame(quarterly_data)
    qdf['quarter_str'] = qdf['quarter'].dt.to_period('Q').astype(str)
    
    # Weighted average EPS per quarter
    quarterly_eps = qdf.groupby('quarter_str').apply(
        lambda x: (x['eps'] * x['weight']).sum() / x['weight'].sum() * total_weight
    ).sort_index()
    
    print("Available quarterly EPS:")
    for q, eps in quarterly_eps.items():
        print(f"  {q}: ${eps:.2f}")
    
    # Save for strategy
    quarterly_eps.to_csv('cache/qqq_historical_quarterly_eps.csv')
    print("\nSaved to cache/qqq_historical_quarterly_eps.csv")
