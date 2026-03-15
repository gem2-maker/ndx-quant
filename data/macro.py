"""Fetch US macro-economic proxy data and merge into equity DataFrame."""
import pandas as pd
import yfinance as yf
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Macro proxies available via yfinance (free, no API key)
MACRO_TICKERS = {
    "^TNX": "TNX",        # 10Y Treasury yield
    "^FVX": "FVX",        # 5Y Treasury yield  
    "DX-Y.NYB": "DXY",    # US Dollar Index
    "GLD": "GLD",          # Gold ETF (risk-off proxy)
    "TLT": "TLT",          # 20Y Treasury ETF (bond prices)
}


def fetch_macro_data(start=None, end=None, period="2y") -> pd.DataFrame:
    """Fetch macro proxy tickers and return aligned daily DataFrame.
    
    Returns DataFrame with columns: TNX, FVX, DXY, GLD, TLT
    Index is DatetimeIndex, forward-filled to daily.
    """
    frames = {}
    for ticker, col_name in MACRO_TICKERS.items():
        try:
            tk = yf.Ticker(ticker)
            if start and end:
                hist = tk.history(start=start, end=end)
            else:
                hist = tk.history(period=period)
            if not hist.empty:
                frames[col_name] = hist["Close"]
                print(f"  [macro] {ticker} -> {col_name}: {len(hist)} bars")
        except Exception as e:
            print(f"  [macro] {ticker} failed: {e}")
    
    if not frames:
        return pd.DataFrame()
    
    df = pd.DataFrame(frames)
    # Forward-fill then back-fill (macro data may have gaps on holidays)
    df = df.ffill().bfill()
    return df


def merge_macro(equity_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Merge macro data into equity DataFrame on date index.
    
    Aligns macro dates to equity dates via forward-fill.
    """
    if macro_df.empty:
        return equity_df
    
    # Align macro data to equity date range
    macro_aligned = macro_df.reindex(equity_df.index, method="ffill")
    
    # Add columns that don't already exist
    result = equity_df.copy()
    for col in macro_aligned.columns:
        if col not in result.columns:
            result[col] = macro_aligned[col]
    
    return result


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived macro features: yield curve spread, rate momentum."""
    df = df.copy()
    
    # Yield curve spread (10Y - 5Y)
    if "TNX" in df.columns and "FVX" in df.columns:
        df["YIELD_SPREAD"] = df["TNX"] - df["FVX"]
    
    # 30-day rate change (momentum)
    if "TNX" in df.columns:
        df["TNX_CHG_30D"] = df["TNX"].pct_change(30)
        df["TNX_CHG_90D"] = df["TNX"].pct_change(90)
    
    # DXY 30-day change
    if "DXY" in df.columns:
        df["DXY_CHG_30D"] = df["DXY"].pct_change(30)
    
    # GLD momentum (gold rising = risk-off)
    if "GLD" in df.columns:
        df["GLD_MOM_20D"] = df["GLD"].pct_change(20)
    
    return df


if __name__ == "__main__":
    print("Fetching macro data...")
    macro = fetch_macro_data(period="2y")
    print(f"\nMacro DataFrame: {macro.shape}")
    print(f"Date range: {macro.index[0].date()} to {macro.index[-1].date()}")
    print(f"Columns: {list(macro.columns)}")
    print(f"\nLatest values:")
    print(macro.tail(1).T.to_string())
    
    # Test merge
    from data.fetcher import DataFetcher
    print("\nFetching QQQ...")
    fetcher = DataFetcher()
    qqq = fetcher.fetch("QQQ", period="2y", interval="1d")
    merged = merge_macro(qqq, macro)
    enriched = add_macro_features(merged)
    print(f"\nMerged: {enriched.shape}, columns: {list(enriched.columns)}")
    print(f"Sample macro values at last bar:")
    for col in ["TNX", "FVX", "DXY", "GLD", "TLT", "YIELD_SPREAD", "TNX_CHG_30D"]:
        if col in enriched.columns:
            print(f"  {col}: {enriched[col].iloc[-1]:.4f}")
