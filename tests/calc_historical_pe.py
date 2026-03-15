"""Calculate historical QQQ PE from top holdings' financial data."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import yfinance as yf
import pandas as pd
import json

# QQQ top ~20 holdings with approximate weights
HOLDINGS = {
    'AAPL': 0.085, 'MSFT': 0.078, 'NVDA': 0.075, 'AMZN': 0.055,
    'META': 0.045, 'GOOGL': 0.04, 'AVGO': 0.035, 'TSLA': 0.03,
    'COST': 0.025, 'NFLX': 0.02, 'AMD': 0.02, 'GOOG': 0.035,
    'ADBE': 0.02, 'PEP': 0.015, 'LIN': 0.015, 'CSCO': 0.015,
    'TMUS': 0.015, 'QCOM': 0.015, 'INTU': 0.012, 'ISRG': 0.012,
}

total_weight = sum(HOLDINGS.values())
print(f"Total weight covered: {total_weight:.1%}")

# Collect quarterly net income and shares for each holding
all_data = {}

for ticker, weight in HOLDINGS.items():
    try:
        t = yf.Ticker(ticker)
        
        # Get quarterly financials for net income
        qfs = t.quarterly_financials
        if qfs is None or qfs.empty:
            print(f"  {ticker}: no financials")
            continue
        
        # Get net income
        ni_row = None
        for row_name in ['Net Income From Continuing And Discontinued Operation', 
                         'Net Income From Continuing Operation Net Minority Interest',
                         'Net Income']:
            if row_name in qfs.index:
                ni_row = qfs.loc[row_name]
                break
        
        if ni_row is None:
            print(f"  {ticker}: no net income row")
            continue
        
        # Get shares outstanding from info
        info = t.info
        shares = info.get('sharesOutstanding', 0)
        if not shares:
            print(f"  {ticker}: no shares outstanding")
            continue
        
        # Get current market cap for validation
        mc = info.get('marketCap', 0)
        
        all_data[ticker] = {
            'weight': weight,
            'shares': shares,
            'market_cap': mc,
            'net_income': ni_row.to_dict(),
        }
        print(f"  {ticker}: OK ({len(ni_row)} quarters, shares={shares/1e6:.0f}M)")
        
    except Exception as e:
        print(f"  {ticker}: error - {e}")

# Calculate weighted quarterly earnings
print(f"\nCollected data for {len(all_data)} tickers")

# Build quarterly aggregate
quarterly_earnings = {}
for ticker, data in all_data.items():
    weight = data['weight']
    shares = data['shares']
    for date, ni in data['net_income'].items():
        if pd.isna(ni):
            continue
        date_str = pd.Timestamp(date).strftime('%Y-%m')
        if date_str not in quarterly_earnings:
            quarterly_earnings[date_str] = {'total_earnings': 0, 'total_weight': 0}
        # EPS contribution = net income / shares * weight
        eps = ni / shares
        quarterly_earnings[date_str]['total_earnings'] += eps * weight
        quarterly_earnings[date_str]['total_weight'] += weight

# Normalize by total weight covered
for date_str in quarterly_earnings:
    qw = quarterly_earnings[date_str]
    if qw['total_weight'] > 0:
        qw['weighted_eps'] = qw['total_earnings'] / qw['total_weight'] * total_weight
    else:
        qw['weighted_eps'] = 0

print("\nQuarterly weighted EPS (top holdings aggregate):")
for date_str in sorted(quarterly_earnings.keys()):
    qw = quarterly_earnings[date_str]
    print(f"  {date_str}: EPS=${qw.get('weighted_eps', 0):.2f}, weight={qw['total_weight']:.1%}")

# Save for use
with open('cache/qqq_quarterly_eps.json', 'w') as f:
    json.dump(quarterly_earnings, f, indent=2)
print("\nSaved to cache/qqq_quarterly_eps.json")
