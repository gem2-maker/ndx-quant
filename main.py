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
from ml import FeatureEngineer, TrendPredictor, ModelEvaluator
from config import ML_MODEL_TYPE, ML_LOOKBACK, ML_FORECAST_HORIZON, ML_TEST_SIZE, ML_N_ESTIMATORS, ML_MAX_DEPTH


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


def cmd_cache(args):
    """Manage data cache."""
    from data.cache import DataCache
    cache = DataCache()

    if args.action == "stats":
        print(cache.format_stats())
    elif args.action == "clear":
        if args.ticker:
            count = cache.invalidate(args.ticker, args.period)
            print(f"Removed {count} entries for {args.ticker}")
        else:
            count = cache.clear_all()
            print(f"Cleared entire cache ({count} entries)")
    elif args.action == "compact":
        count = cache.compact()
        print(f"Removed {count} stale entries, database compacted")


def cmd_backtest_ml(args):
    """Run backtest with ML-driven strategy."""
    from strategies.ml_signal import MLSignalStrategy
    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Running ML backtest on {ticker}...")
    print(f"  Model: {args.model}, Lookback: {args.lookback}, Horizon: {args.horizon}")
    print(f"  Confidence: {args.confidence}, Retrain every: {args.retrain} bars\n")

    df = fetcher.fetch(ticker, period=args.period)
    if df.empty:
        print(f"No data for {ticker}")
        return

    # Ensure indicators are present
    df = add_all_indicators(df)

    strategy = MLSignalStrategy(
        model_type=args.model,
        lookback=args.lookback,
        forecast_horizon=args.horizon,
        confidence_threshold=args.confidence,
        retrain_interval=args.retrain,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    engine = BacktestEngine(strategy)
    result = engine.run(df, ticker)

    print(engine.summary(result, ticker))
    print(f"\nML Strategy Info: {strategy.training_info}")


def cmd_train(args):
    """Train ML prediction model."""
    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Training {args.model} model on {ticker} (period={args.period})...\n")
    df = fetcher.fetch(ticker, period=args.period)
    if df.empty:
        print(f"No data for {ticker}")
        return

    predictor = TrendPredictor(
        model_type=args.model,
        lookback=args.lookback,
        forecast_horizon=args.horizon,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )

    result = predictor.train(df)
    print(predictor.summary(result))

    if args.save:
        predictor.save(ticker)
        print(f"\nModel saved as {ticker}_{args.model}.pkl")


def cmd_predict(args):
    """Predict trend direction using trained ML model."""
    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Predicting {ticker} with {args.model} model...\n")
    df = fetcher.fetch(ticker, period=args.period)
    if df.empty:
        print(f"No data for {ticker}")
        return

    predictor = TrendPredictor(
        model_type=args.model,
        lookback=args.lookback,
        forecast_horizon=args.horizon,
    )

    # Train on data (or load if saved model exists)
    try:
        predictor.load(ticker)
        print("  Loaded saved model\n")
    except FileNotFoundError:
        print("  Training new model...")
        predictor.train(df)
        print()

    # Predict latest
    predictions = predictor.predict(df, top_n=args.n)

    print(f"Latest predictions for {ticker}:")
    print("-" * 55)
    for p in predictions:
        conf_bar = "+" * int(p.probability * 30)
        print(f"  {p.date.strftime('%Y-%m-%d')}  {p.signal:4s}  "
              f"P(up)={p.probability:.2%}  [{p.confidence:6s}]  {conf_bar}")

    if args.evaluate:
        print()
        evaluator = ModelEvaluator(predictor)
        metrics = evaluator.evaluate(df)
        print(evaluator.format_metrics(metrics))


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

    # train
    p_train = sub.add_parser("train", help="Train ML prediction model")
    p_train.add_argument("ticker", nargs="?", help="Ticker symbol (default: QQQ)")
    p_train.add_argument("--period", default="2y", help="Data period (default: 2y)")
    p_train.add_argument("--model", "-m", default=ML_MODEL_TYPE,
                         choices=["xgboost", "random_forest", "gradient_boosting"])
    p_train.add_argument("--lookback", type=int, default=ML_LOOKBACK)
    p_train.add_argument("--horizon", type=int, default=ML_FORECAST_HORIZON)
    p_train.add_argument("--test-size", type=float, default=ML_TEST_SIZE)
    p_train.add_argument("--n-estimators", type=int, default=ML_N_ESTIMATORS)
    p_train.add_argument("--max-depth", type=int, default=ML_MAX_DEPTH)
    p_train.add_argument("--save", action="store_true", help="Save model to disk")

    # predict
    p_pred = sub.add_parser("predict", help="Predict trend direction")
    p_pred.add_argument("ticker", nargs="?", help="Ticker symbol (default: QQQ)")
    p_pred.add_argument("--period", default="2y", help="Data period")
    p_pred.add_argument("--model", "-m", default=ML_MODEL_TYPE,
                        choices=["xgboost", "random_forest", "gradient_boosting"])
    p_pred.add_argument("--n", type=int, default=5, help="Number of recent predictions")
    p_pred.add_argument("--lookback", type=int, default=ML_LOOKBACK)
    p_pred.add_argument("--horizon", type=int, default=ML_FORECAST_HORIZON)
    p_pred.add_argument("--evaluate", action="store_true", help="Show evaluation metrics")

    # backtest-ml
    p_btml = sub.add_parser("backtest-ml", help="Backtest with ML-driven strategy")
    p_btml.add_argument("ticker", nargs="?", help="Ticker symbol (default: QQQ)")
    p_btml.add_argument("--period", default="2y", help="Data period")
    p_btml.add_argument("--model", "-m", default=ML_MODEL_TYPE,
                        choices=["xgboost", "random_forest", "gradient_boosting"])
    p_btml.add_argument("--lookback", type=int, default=ML_LOOKBACK)
    p_btml.add_argument("--horizon", type=int, default=ML_FORECAST_HORIZON)
    p_btml.add_argument("--confidence", default="low", choices=["low", "medium", "high"],
                        help="Minimum confidence threshold")
    p_btml.add_argument("--retrain", type=int, default=0,
                        help="Retrain every N bars (0=never)")
    p_btml.add_argument("--n-estimators", type=int, default=ML_N_ESTIMATORS)
    p_btml.add_argument("--max-depth", type=int, default=ML_MAX_DEPTH)

    # cache
    p_cache = sub.add_parser("cache", help="Manage data cache")
    p_cache.add_argument("action", choices=["stats", "clear", "compact"],
                         help="Cache action")
    p_cache.add_argument("--ticker", "-t", help="Target ticker (for clear)")
    p_cache.add_argument("--period", "-p", help="Target period (for clear)")

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
        "train": cmd_train,
        "predict": cmd_predict,
        "backtest-ml": cmd_backtest_ml,
        "cache": cmd_cache,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
