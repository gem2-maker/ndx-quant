"""Try to scrape historical PE from multiple sources."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Try Yahoo Finance API for QQQ historical PE
# This is an undocumented API endpoint
url = "https://query1.finance.yahoo.com/v10/finance/quoteSummary/QQQ"
params = {
    "modules": "defaultKeyStatistics,summaryDetail",
}

try:
    r = requests.get(url, params=params, headers=headers, timeout=10)
    print(f"Yahoo API status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(json.dumps(data, indent=2, default=str)[:2000])
except Exception as e:
    print(f"Yahoo API error: {e}")

# Try another Yahoo endpoint for historical data
print("\n=== Try Yahoo v8 chart ===")
try:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/QQQ"
    params = {"range": "5y", "interval": "1mo"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        # Check for PE data in response
        result = data.get('chart', {}).get('result', [{}])[0]
        meta = result.get('meta', {})
        print(f"Meta keys: {list(meta.keys())}")
        print(f"Regular market price: {meta.get('regularMarketPrice', 'N/A')}")
except Exception as e:
    print(f"Error: {e}")

# Try FRED for S&P 500 PE (not Nasdaq-100 but related)
print("\n=== FRED S&P 500 PE Ratio ===")
try:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "PE_RATIO",
        "api_key": "DEMO_KEY",  # FRED has a demo key
        "file_type": "json",
        "sort_order": "desc",
        "limit": 20,
    }
    r = requests.get(url, params=params, headers=headers, timeout=10)
    print(f"FRED status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        for obs in data.get('observations', [])[:5]:
            print(f"  {obs['date']}: {obs['value']}")
    else:
        print(f"FRED response: {r.text[:500]}")
except Exception as e:
    print(f"FRED error: {e}")
