"""ndx-quant CLI — Nasdaq-100 Quantitative Trading Toolkit."""

import sys
import argparse
import json

# Add project root to path
sys.path.insert(0, ".")

from data.fetcher import DataFetcher
from data import get_tickers
from indicators.technical import add_all_indicators
from strategies import get_strategy, STRATEGIES
from backtest.engine import BacktestEngine
from portfolio.analyzer import PortfolioAnalyzer


def cmd_fetch(args):
    """Fetch price data."""
    fetcher = DataFetcher()

    if args.all:
        data = fetcher.fetch_ndx(period=args.period, sector=args.sector)
        print(f"\nFetched {len(data)} tickers")
        for ticker, df in data.items():
            print(f"  {ticker}: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
    else:
        ticker = args.ticker or "QQQ"
        df = fetcher.fetch(ticker, period=args.period)
        if not df.empty:
            df = add_all_indicators(df)
            print(f"\n{ticker} — {len(df)} bars")
            print(df.tail(5)[["Close", "SMA_20", "SMA_50", "RSI", "MACD"]].to_string())
        else:
            print(f"No data for {ticker}")


def cmd_backtest(args):
    """Run a backtest."""
    strategy = get_strategy(args.strategy)
    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Running {strategy.name} backtest on {ticker}...")
    df = fetcher.fetch(ticker, period=args.period)
    if df.empty:
        print(f"No data for {ticker}")
        return

    engine = BacktestEngine(strategy)
    result = engine.run(df, ticker)
    print(engine.summary(result, ticker))


def cmd_analyze(args):
    """Analyze portfolio risk."""
    analyzer = PortfolioAnalyzer()
    ticker = args.ticker or "QQQ"

    print(f"Analyzing {ticker}...\n")
    metrics = analyzer.risk_metrics(ticker, period=args.period)

    if not metrics:
        print(f"No data for {ticker}")
        return

    for key, val in metrics.items():
        if isinstance(val, float):
            if "ratio" in key or "return" in key or "drawdown" in key or "var" in key:
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")


def cmd_momentum(args):
    """Find top momentum stocks in NDX-100."""
    analyzer = PortfolioAnalyzer()
    tickers = get_tickers(args.sector)

    print(f"Scanning {len(tickers)} NDX-100 tickers for momentum...\n")
    top = analyzer.top_momentum(tickers, period=args.period, top_n=args.top)

    print(f"Top {args.top} by momentum ({args.period}):")
    print("-" * 40)
    for i, (ticker, ret) in enumerate(top, 1):
        bar = "+" * int(abs(ret) * 50)
        print(f"  {i:2d}. {ticker:5s} {ret:+.2%} {bar}")


def cmd_strategies(args):
    """List available strategies."""
    print("Available strategies:")
    for name, cls in STRATEGIES.items():
        print(f"  {name:20s} — {cls.__doc__.strip().split(chr(10))[0] if cls.__doc__ else 'No description'}")


def main():
    parser = argparse.ArgumentParser(
        prog="ndx-quant",
        description="Nasdaq-100 Quantitative Trading Toolkit",
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch price data")
    p_fetch.add_argument("ticker", nargs="?", help="Ticker symbol (default: QQQ)")
    p_fetch.add_argument("--period", default="2y", help="Data period (default: 2y)")
    p_fetch.add_argument("--all", action="store_true", help="Fetch all NDX-100 tickers")
    p_fetch.add_argument("--sector", help="Filter by sector (tech/consumer/healthcare/communication)")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("--strategy", "-s", required=True, choices=list(STRATEGIES.keys()))
    p_bt.add_argument("--ticker", "-t", default="QQQ", help="Ticker to backtest")
    p_bt.add_argument("--period", default="2y", help="Data period")

    # analyze
    p_an = sub.add_parser("analyze", help="Analyze risk metrics")
    p_an.add_argument("--ticker", "-t", default="QQQ")
    p_an.add_argument("--period", default="1y")

    # momentum
    p_mom = sub.add_parser("momentum", help="Find top momentum stocks")
    p_mom.add_argument("--period", default="3mo")
    p_mom.add_argument("--top", type=int, default=10)
    p_mom.add_argument("--sector", help="Filter by sector")

    # strategies
    sub.add_parser("strategies", help="List available strategies")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "fetch": cmd_fetch,
        "backtest": cmd_backtest,
        "analyze": cmd_analyze,
        "momentum": cmd_momentum,
        "strategies": cmd_strategies,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
