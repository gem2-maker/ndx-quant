"""Search for free historical PE data sources for QQQ/Nasdaq-100."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import json
import time

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# 1. Financial Modeling Prep - has historical PE, free tier
print("=== Financial Modeling Prep ===")
try:
    # Free API key available for demo
    url = "https://financialmodelingprep.com/api/v3/historical-price-to-earning-ratio/QQQ"
    params = {"limit": 50}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list):
            print(f"Got {len(data)} data points")
            for item in data[:5]:
                print(f"  {item}")
        else:
            print(f"Response: {str(data)[:300]}")
    elif r.status_code == 401:
        print("Requires API key (free tier available)")
except Exception as e:
    print(f"Error: {e}")

time.sleep(1)

# 2. Polygon.io - free tier
print("\n=== Polygon.io ===")
try:
    url = "https://api.polygon.io/v2/aggs/ticker/QQQ/range/1/month/2021-01-01/2026-03-15"
    params = {"adjusted": "true", "apikey": "DEMO_KEY"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Results: {data.get('resultsCount', 0)}")
    else:
        print(f"Need key: {r.text[:200]}")
except Exception as e:
    print(f"Error: {e}")

time.sleep(1)

# 3. IEX Cloud - has PE ratios
print("\n=== IEX Cloud ===")
try:
    url = "https://cloud.iexapis.com/stable/stock/QQQ/quote"
    params = {"token": "DEMO_KEY"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")

time.sleep(1)

# 4. Tiingo - fundamental data
print("\n=== Tiingo ===")
try:
    url = "https://api.tiingo.com/tiingo/fundamentals/QQQ/daily"
    headers_tiingo = {**headers, 'Authorization': 'Token DEMO_KEY'}
    r = requests.get(url, headers=headers_tiingo, timeout=15)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Data: {str(data)[:500]}")
    else:
        print(f"Need key: {r.text[:200]}")
except Exception as e:
    print(f"Error: {e}")

# 5. Try Nasdaq Data Link (formerly Quandl)
print("\n=== Nasdaq Data Link ===")
try:
    # SHILLER PE data
    url = "https://data.nasdaq.com/api/v3/datasets/MULTPL/SHILLER_PE_RATIO_MONTH.json"
    params = {"api_key": "DEMO_KEY", "limit": 10}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    print(f"Shiller PE status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        dataset = data.get('dataset', {})
        points = dataset.get('data', [])
        print(f"Got {len(points)} points")
        for p in points[:5]:
            print(f"  {p[0]}: {p[1]}")
    else:
        print(f"Text: {r.text[:200]}")
except Exception as e:
    print(f"Error: {e}")

# 6. SEC EDGAR - company filings
print("\n=== SEC EDGAR ===")
try:
    # Get Apple's 10-Q filings for earnings data
    url = "https://efts.sec.gov/LATEST/search-index"
    params = {"q": "Nasdaq-100 PE ratio", "dateRange": "custom", "startdt": "2021-01-01"}
    r = requests.get(url, headers={**headers, 'User-Agent': 'OpenClaw datovm@gmail.com'}, timeout=15)
    print(f"SEC EDGAR status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")

# 7. Try free FMP API (no key needed for basic)
print("\n=== FMP free endpoint ===")
try:
    url = "https://financialmodelingprep.com/api/v3/ratios/QQQ?limit=10"
    r = requests.get(url, headers=headers, timeout=15)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list) and data:
            for item in data[:3]:
                date = item.get('date', 'N/A')
                pe = item.get('priceEarningsRatio', 'N/A')
                pb = item.get('priceToBookRatio', 'N/A')
                print(f"  {date}: PE={pe}, P/B={pb}")
        else:
            print(f"Response: {str(data)[:300]}")
except Exception as e:
    print(f"Error: {e}")

# 8. Try Alpha Vantage 
print("\n=== Alpha Vantage ===")
try:
    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": "QQQ", "apikey": "demo"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        pe = data.get('PERatio', 'N/A')
        peg = data.get('PEGRatio', 'N/A')
        print(f"PE: {pe}, PEG: {peg}")
except Exception as e:
    print(f"Error: {e}")
