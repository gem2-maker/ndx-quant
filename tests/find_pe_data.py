"""Find historical PE data for Nasdaq-100 / QQQ."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import yfinance as yf

# Approach 1: Try QQQ earnings directly
print("=== Approach 1: QQQ ETF direct ===")
qqq = yf.Ticker('QQQ')
info = qqq.info
print(f"Trailing PE: {info.get('trailingPE', 'N/A')}")
print(f"Forward PE: {info.get('forwardPE', 'N/A')}")
print(f"Price to Book: {info.get('priceToBook', 'N/A')}")

# Approach 2: Calculate from top holdings
print("\n=== Approach 2: Top holdings earnings ===")
top_holdings = [
    ('AAPL', 0.085), ('MSFT', 0.078), ('NVDA', 0.075),
    ('AMZN', 0.055), ('META', 0.045), ('GOOGL', 0.04),
    ('AVGO', 0.035), ('TSLA', 0.03), ('COST', 0.025),
    ('NFLX', 0.02)
]

total_weight = sum(w for _, w in top_holdings)
print(f"Top 10 weight: {total_weight:.1%} of QQQ")

for ticker, weight in top_holdings:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        pe = info.get('trailingPE', 'N/A')
        shares = info.get('sharesOutstanding', 0)
        mc = info.get('marketCap', 0)
        eps = info.get('trailingEps', 'N/A')
        print(f"  {ticker}: PE={pe}, EPS={eps}, MarketCap=${mc/1e9:.0f}B")
    except Exception as e:
        print(f"  {ticker}: error - {e}")

# Approach 3: Get quarterly financials for aggregate
print("\n=== Approach 3: Quarterly financials ===")
aapl = yf.Ticker('AAPL')
try:
    qfs = aapl.quarterly_financials
    print(f"AAPL quarterly financials: {qfs.shape}")
    print(f"Rows: {list(qfs.index[:10])}")
    if 'Net Income' in qfs.index:
        ni = qfs.loc['Net Income']
        print(f"Net Income (last 4Q): {ni.to_dict()}")
except Exception as e:
    print(f"Error: {e}")

try:
    qbs = aapl.quarterly_balance_sheet
    print(f"AAPL quarterly balance sheet: {qbs.shape}")
    print(f"Rows: {list(qbs.index[:10])}")
except Exception as e:
    print(f"Error: {e}")
