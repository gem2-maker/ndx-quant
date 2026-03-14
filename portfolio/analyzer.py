"""Portfolio analysis tools."""

import pandas as pd
import numpy as np

from data.fetcher import DataFetcher


class PortfolioAnalyzer:
    """Analyze portfolio risk, correlations, and performance."""

    def __init__(self):
        self.fetcher = DataFetcher()

    def correlation_matrix(
        self,
        tickers: list[str],
        period: str = "1y",
    ) -> pd.DataFrame:
        """Calculate return correlation matrix for a list of tickers."""
        returns = {}
        data = self.fetcher.fetch_multiple(tickers, period=period)
        for ticker, df in data.items():
            returns[ticker] = df["Close"].pct_change().dropna()

        if not returns:
            return pd.DataFrame()

        # Align indices
        df_returns = pd.DataFrame(returns)
        return df_returns.corr()

    def risk_metrics(
        self,
        ticker: str,
        period: str = "1y",
        risk_free_rate: float = 0.05,
    ) -> dict:
        """Calculate risk metrics for a single ticker."""
        df = self.fetcher.fetch(ticker, period=period)
        if df.empty:
            return {}

        returns = df["Close"].pct_change().dropna()
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * (252 ** 0.5)
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max) - 1
        max_dd = drawdown.min()

        # Sortino (downside deviation)
        downside = returns[returns < 0]
        downside_vol = downside.std() * (252 ** 0.5)
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

        # Value at Risk (95%)
        var_95 = returns.quantile(0.05)

        return {
            "ticker": ticker,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "var_95": var_95,
            "best_day": returns.max(),
            "worst_day": returns.min(),
            "positive_days": (returns > 0).sum() / len(returns),
        }

    def sector_breakdown(self, tickers: list[str]) -> dict[str, list[str]]:
        """Group tickers by sector (basic mapping)."""
        from data import SECTOR_MAP if hasattr(__import__('data'), 'SECTOR_MAP') else {}
        # Simple reverse lookup
        result = {"other": []}
        sector_map = {
            "tech": ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "TXN", "ADI",
                     "MRVL", "MU", "MCHP", "ON", "AVGO", "LRCX", "KLAC", "AMAT",
                     "SNPS", "CDNS", "SMCI", "CRWD", "PANW", "FTNT"],
            "consumer": ["AMZN", "TSLA", "COST", "NFLX", "SBUX", "ROST", "MAR",
                         "ABNB", "BKNG", "ORLY", "LULU"],
            "healthcare": ["AMGN", "GILD", "REGN", "VRTX", "ISRG", "DXCM"],
            "communication": ["META", "GOOGL", "GOOG", "CMCSA", "CHTR", "TTD"],
        }
        for ticker in tickers:
            found = False
            for sector, members in sector_map.items():
                if ticker in members:
                    result.setdefault(sector, []).append(ticker)
                    found = True
                    break
            if not found:
                result["other"].append(ticker)
        return {k: v for k, v in result.items() if v}

    def top_momentum(
        self,
        tickers: list[str],
        period: str = "3mo",
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Find top N stocks by momentum (return over period)."""
        data = self.fetcher.fetch_multiple(tickers, period=period)
        momentum = []
        for ticker, df in data.items():
            if len(df) > 1:
                ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
                momentum.append((ticker, ret))
        momentum.sort(key=lambda x: x[1], reverse=True)
        return momentum[:top_n]
