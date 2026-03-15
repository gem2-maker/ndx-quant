"""Get historical S&P 500 PE from Shiller data and estimate Nasdaq-100 PE."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import io
import csv

headers = {'User-Agent': 'Mozilla/5.0'}

# Shiller's historical S&P 500 data (CAPE ratio and PE)
# Source: http://www.econ.yale.edu/~shiller/data.htm
# Alternative: try to get it from a public source

print("=== Trying Shiller S&P 500 data ===")
try:
    # Try downloading Shiller's Excel data
    url = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
    r = requests.get(url, headers=headers, timeout=15)
    print(f"Shiller data status: {r.status_code}, size: {len(r.content)} bytes")
    if r.status_code == 200:
        with open('cache/shiller_ie_data.xls', 'wb') as f:
            f.write(r.content)
        print("Saved to cache/shiller_ie_data.xls")
except Exception as e:
    print(f"Shiller error: {e}")

# Try multpl.com for Nasdaq-100 PE specifically
print("\n=== Trying multpl.com ===")
for url in [
    "https://www.multpl.com/nasdaq-100-pe-ratio",
    "https://www.multpl.com/s-p-500-pe-ratio",
    "https://www.multpl.com/s-p-500-pe-ratio/table/by-year",
]:
    try:
        r = requests.get(url, headers=headers, timeout=15)
        print(f"\n{url}: status={r.status_code}, size={len(r.text)}")
        if r.status_code == 200 and len(r.text) > 1000:
            # Extract table data
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, 'html.parser')
            tables = soup.find_all('table')
            print(f"  Found {len(tables)} tables")
            for i, table in enumerate(tables[:2]):
                rows = table.find_all('tr')
                print(f"  Table {i}: {len(rows)} rows")
                for row in rows[:5]:
                    cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                    print(f"    {cells}")
    except Exception as e:
        print(f"{url}: error - {e}")

# Try StockAnalysis API for QQQ PE history
print("\n=== StockAnalysis ===")
try:
    url = "https://stockanalysis.com/etf/qqq/financials/"
    r = requests.get(url, headers=headers, timeout=15)
    print(f"Status: {r.status_code}, size: {len(r.text)}")
    if r.status_code == 200:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, 'html.parser')
        # Look for PE ratio data
        text = soup.get_text()
        if 'P/E Ratio' in text or 'PE Ratio' in text:
            # Find the context
            for line in text.split('\n'):
                if 'P/E' in line or 'PE' in line[:10]:
                    print(f"  Found: {line.strip()[:100]}")
except Exception as e:
    print(f"Error: {e}")
