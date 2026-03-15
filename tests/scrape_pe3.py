"""Scrape full S&P 500 PE table and download Shiller data."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import requests
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0'}

# 1. Get full S&P 500 PE table from multpl.com
print("=== S&P 500 PE from multpl.com ===")
url = "https://www.multpl.com/s-p-500-pe-ratio/table/by-year"
r = requests.get(url, headers=headers, timeout=15)

if r.status_code == 200:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(r.text, 'html.parser')
    table = soup.find('table')
    if table:
        rows = table.find_all('tr')
        data = []
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
            if len(cells) >= 2:
                data.append(cells[:2])
        
        df = pd.DataFrame(data[1:], columns=['Date', 'SP500_PE'])
        # Clean PE values (remove † and other chars)
        df['SP500_PE'] = df['SP500_PE'].str.replace('†', '').str.strip()
        df['SP500_PE'] = pd.to_numeric(df['SP500_PE'], errors='coerce')
        df = df.dropna(subset=['SP500_PE'])
        
        print(f"Got {len(df)} data points")
        print(f"Date range: {df['Date'].iloc[-1]} to {df['Date'].iloc[0]}")
        print(f"\nLast 10 entries:")
        for _, row in df.head(10).iterrows():
            print(f"  {row['Date']}: {row['SP500_PE']:.2f}")
        
        # Save
        df.to_csv('cache/sp500_pe_history.csv', index=False)
        print("\nSaved to cache/sp500_pe_history.csv")

# 2. Try to read Shiller XLS data
print("\n=== Shiller data ===")
try:
    import openpyxl
    wb = openpyxl.load_workbook('cache/shiller_ie_data.xls', data_only=True)
    print(f"Sheets: {wb.sheetnames}")
except ImportError:
    print("openpyxl not available, trying xlrd")
    try:
        import xlrd
        wb = xlrd.open_workbook('cache/shiller_ie_data.xls')
        print(f"Sheets: {wb.sheet_names()}")
        # Read first sheet
        ws = wb.sheet_by_index(0)
        print(f"Rows: {ws.nrows}, Cols: {ws.ncols}")
        # Print header
        print(f"Header row: {[ws.cell_value(0, c) for c in range(min(10, ws.ncols))]}")
        # Print last few rows with data
        print(f"\nLast 5 rows:")
        for r in range(max(0, ws.nrows-5), ws.nrows):
            row_data = [ws.cell_value(r, c) for c in range(min(5, ws.ncols))]
            print(f"  {row_data}")
    except ImportError:
        print("xlrd not available either")
    except Exception as e:
        print(f"xlrd error: {e}")
except Exception as e:
    print(f"openpyxl error: {e}")

# 3. Get Nasdaq-100 historical PE from another source
print("\n=== Try Gurufocus for NDX PE ===")
try:
    url = "https://www.gurufocus.com/term/pettm/QQQ/PE-Ratio/QQQ"
    r = requests.get(url, headers=headers, timeout=15)
    print(f"Gurufocus status: {r.status_code}")
    if r.status_code == 200:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, 'html.parser')
        text = soup.get_text()
        # Look for PE data
        import re
        pe_matches = re.findall(r'(\d{4}-\d{2}-\d{2})\D+?(\d+\.?\d*)', text[:5000])
        if pe_matches:
            print(f"Found PE data: {pe_matches[:5]}")
        else:
            # Try to find table
            table = soup.find('table')
            if table:
                rows = table.find_all('tr')
                print(f"Table with {len(rows)} rows")
                for row in rows[:5]:
                    cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                    print(f"  {cells}")
except Exception as e:
    print(f"Gurufocus error: {e}")
