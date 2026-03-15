"""Get longer historical EPS using annual financials for top QQQ holdings."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import yfinance as yf
import pandas as pd
import json

TOP_HOLDINGS = [
    ('AAPL', 0.085), ('MSFT', 0.078), ('NVDA', 0.075), ('AMZN', 0.055),
    ('META', 0.045), ('GOOGL', 0.04), ('AVGO', 0.035), ('TSLA', 0.03),
    ('COST', 0.025), ('NFLX', 0.02), ('AMD', 0.02), ('GOOG', 0.035),
    ('ADBE', 0.02), ('PEP', 0.015), ('LIN', 0.015), ('CSCO', 0.015),
    ('TMUS', 0.015), ('QCOM', 0.015), ('INTU', 0.012), ('ISRG', 0.012),
]
total_weight = sum(w for _, w in TOP_HOLDINGS)

results = {}
for ticker, weight in TOP_HOLDINGS:
    try:
        t = yf.Ticker(ticker)
        
        # Annual financials (usually 4 years)
        fin = t.financials
        if fin is None or fin.empty:
            print(f"{ticker}: no annual financials")
            continue
        
        # Find net income row
        ni_row = None
        for row_name in ['Net Income From Continuing And Discontinued Operation',
                         'Net Income From Continuing Operation Net Minority Interest',
                         'Net Income']:
            if row_name in fin.index:
                ni_row = fin.loc[row_name]
                break
        if ni_row is None:
            print(f"{ticker}: no net income in {list(fin.index[:5])}")
            continue
        
        # Quarterly financials (5-7 quarters)
        qfin = t.quarterly_financials
        qni_row = None
        if qfin is not None and not qfin.empty:
            for row_name in ['Net Income From Continuing And Discontinued Operation',
                             'Net Income From Continuing Operation Net Minority Interest',
                             'Net Income']:
                if row_name in qfin.index:
                    qni_row = qfin.loc[row_name]
                    break
        
        # Shares outstanding
        info = t.info
        shares = info.get('sharesOutstanding', 0)
        if not shares:
            print(f"{ticker}: no shares")
            continue
        
        # Build quarterly EPS series
        quarterly_eps = {}
        
        # From quarterly financials
        if qni_row is not None:
            for date, ni in qni_row.items():
                if not pd.isna(ni):
                    q = pd.Timestamp(date).to_period('Q')
                    quarterly_eps[str(q)] = ni / shares
        
        # From annual - compute annual EPS
        annual_eps = {}
        for date, ni in ni_row.items():
            if not pd.isna(ni):
                year = pd.Timestamp(date).year
                annual_eps[year] = ni / shares
        
        results[ticker] = {
            'weight': weight,
            'shares': shares,
            'quarterly_eps': quarterly_eps,
            'annual_eps': annual_eps,
        }
        
        print(f"{ticker}: annual={sorted(annual_eps.keys())}, quarters={sorted(quarterly_eps.keys())}")
        
    except Exception as e:
        print(f"{ticker}: error - {e}")

# Build aggregate PE series
# Combine annual + quarterly to get maximum coverage
print(f"\n{'='*60}")
print("Aggregating weighted EPS...")

# Create quarterly timeline from all data
all_quarters = set()
all_years = set()
for data in results.values():
    all_quarters.update(data['quarterly_eps'].keys())
    all_years.update(data['annual_eps'].keys())

all_quarters = sorted(all_quarters)
all_years = sorted(all_years)

print(f"Available years: {all_years}")
print(f"Available quarters: {all_quarters}")

# Calculate TTM (trailing twelve months) EPS for each quarter
# and weighted annual EPS for each year
yearly_weighted_eps = {}
for year in all_years:
    total_eps = 0
    total_w = 0
    for ticker, data in results.items():
        if year in data['annual_eps']:
            total_eps += data['annual_eps'][year] * data['weight']
            total_w += data['weight']
    if total_w > 0:
        yearly_weighted_eps[year] = total_eps / total_w * total_weight

print(f"\nWeighted Annual EPS:")
for year, eps in sorted(yearly_weighted_eps.items()):
    print(f"  {year}: ${eps:.4f}")

# Get QQQ prices at year end
qqq = yf.Ticker('QQQ')
hist = qqq.history(period='10y')
hist.index = hist.index.tz_localize(None)

print(f"\nQQQ Year-end PE (calculated):")
for year in sorted(yearly_weighted_eps.keys()):
    year_data = hist[hist.index.year == year]
    if not year_data.empty:
        price = year_data['Close'].iloc[-1]
        eps = yearly_weighted_eps[year]
        pe = price / eps if eps > 0 else 0
        print(f"  {year}: Price=${price:.2f}, EPS=${eps:.2f}, PE={pe:.1f}")

# Save
with open('cache/qqq_historical_pe.json', 'w') as f:
    json.dump({
        'yearly_weighted_eps': yearly_weighted_eps,
        'total_weight_coverage': total_weight,
    }, f, indent=2)
print("\nSaved to cache/qqq_historical_pe.json")
