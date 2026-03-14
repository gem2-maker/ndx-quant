"""Nasdaq-100 constituents (as of early 2026)."""

# Top 100 non-financial companies on Nasdaq by market cap
# Updated periodically — for exact list see https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index

NDX_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "NFLX", "ADBE", "AMD", "PEP", "CSCO", "TMUS", "INTC", "QCOM", "INTU", "AMGN",
    "TXN", "HON", "ISRG", "BKNG", "AMAT", "CMCSA", "ADP", "GILD", "MDLZ", "REGN",
    "VRTX", "SBUX", "ADI", "LRCX", "PYPL", "MU", "MELI", "SNPS", "CDNS", "KLAC",
    "MRVL", "ORLY", "CSX", "ABNB", "CHTR", "MAR", "FTNT", "NXPI", "DASH", "KDP",
    "MNST", "PCAR", "CTAS", "ROP", "PANW", "WDAY", "CPRT", "IDXX", "MCHP", "AEP",
    "ROST", "PAYX", "DXCM", "XEL", "ODFL", "FAST", "VRSK", "KHC", "CTSH", "EA",
    "EXC", "ZS", "TTD", "GEHC", "CCEP", "FANG", "LULU", "BKR", "ON", "DDOG",
    "CEG", "KDP", "GFS", "SMCI", "CRWD", "PLTR", "MDB", "TEAM", "MRNA", "DLTR",
    "BIIB", "SIRI", "WBD", "ALGN", "LCID", "ZM", "EBAY", "JD", "BIDU", "PDD",
]


def get_tickers(sector: str | None = None) -> list[str]:
    """Return NDX-100 tickers, optionally filtered by sector."""
    # Basic sector mapping for common groupings
    SECTOR_MAP = {
        "tech": ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "TXN", "ADI",
                 "MRVL", "MU", "MCHP", "ON", "AVGO", "LRCX", "KLAC", "AMAT",
                 "SNPS", "CDNS", "SMCI", "CRWD", "PANW", "FTNT", "ZS", "DDOG",
                 "MDB", "NET", "TEAM", "CRM", "ADBE", "INTU", "NOW"],
        "consumer": ["AMZN", "TSLA", "COST", "NFLX", "SBUX", "ROST", "MAR",
                     "ABNB", "BKNG", "ORLY", "LULU", "DLTR", "MNST", "KHC"],
        "healthcare": ["AMGN", "GILD", "REGN", "VRTX", "ISRG", "DXCM", "MRNA",
                       "BIIB", "ALGN", "GEHC"],
        "communication": ["META", "GOOGL", "GOOG", "CMCSA", "CHTR", "TTD",
                          "WBD", "SIRI", "ZM", "BIDU", "JD"],
    }
    if sector and sector.lower() in SECTOR_MAP:
        return SECTOR_MAP[sector.lower()]
    return NDX_TICKERS
