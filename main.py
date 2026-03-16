"""ndx-quant CLI — Nasdaq-100 Quantitative Trading Toolkit."""

import sys
import argparse
import json
import sqlite3
from io import StringIO
from typing import Any

# Add project root to path
sys.path.insert(0, ".")

from data.fetcher import DataFetcher
from data import get_tickers
from indicators.technical import add_all_indicators
from strategies import get_strategy, STRATEGIES
from backtest.engine import BacktestEngine
from portfolio.analyzer import PortfolioAnalyzer
from ml import FeatureEngineer, TrendPredictor, ModelEvaluator, VolatilityPredictor
from config import (
    ML_MODEL_TYPE, ML_LOOKBACK, ML_FORECAST_HORIZON, ML_TEST_SIZE,
    ML_N_ESTIMATORS, ML_MAX_DEPTH, GARCH_MODEL_TYPE, GARCH_P, GARCH_Q, VOL_TARGET,
    INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE,
)


def load_cached_data(ticker: str, period: str):
    """Load cached price data directly from SQLite without cache-layer writes."""
    from data.cache import DB_PATH

    if not DB_PATH.exists():
        return None

    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(
            """
            SELECT data_json
            FROM ticker_data
            WHERE ticker = ? AND period = ? AND interval = ?
            ORDER BY fetched_at DESC
            LIMIT 1
            """,
            (ticker.upper(), period, "1d"),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    df = json.loads(row[0])
    import pandas as pd

    frame = pd.read_json(StringIO(json.dumps(df)), orient="split")
    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], utc=True).dt.tz_localize(None)
        frame = frame.set_index("Date")
    return frame


def fetch_years_of_data(fetcher: DataFetcher, ticker: str, years: int):
    """Fetch roughly N years of daily data."""
    period = f"{years}y"
    try:
        df = fetcher.fetch(ticker, period=period, interval="1d")
    except Exception:
        df = None

    if df is None or df.empty:
        cache_periods = [period, "10y", "5y", "2y", "1y", "6mo", "3mo"]
        for candidate in cache_periods:
            cached_df = load_cached_data(ticker, candidate)
            if cached_df is not None and not cached_df.empty:
                df = cached_df
                break

    if df is None or df.empty:
        raise RuntimeError(f"Unable to load price data for {ticker} using period={period}")

    if years > 0 and len(df) > 252 * years:
        df = df.iloc[-252 * years:]
    return df


def run_trend_comparison(
    df,
    ticker: str,
    initial_capital: float,
    commission: float,
    slippage: float,
    position_pct: float,
) -> dict[str, Any]:
    """Run V1/V2/V3 trend strategies on the same dataset."""
    from strategies.trend_following import TrendFollowingStrategy, TrendFollowingEngine
    from strategies.trend_following_v2 import TrendFollowingV2, TrendFollowingV2Engine
    from strategies.trend_following_v3 import TrendFollowingV3, TrendFollowingV3Engine

    v1_strategy = TrendFollowingStrategy(
        sma_fast=50,
        sma_slow=200,
        adx_threshold=25.0,
        adx_exit_threshold=20.0,
        volume_factor=1.0,
        pullback_pct=0.03,
        trailing_stop_pct=0.12,
    )
    v1_engine = TrendFollowingEngine(
        strategy=v1_strategy,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        position_pct=position_pct,
    )
    v1_result = v1_engine.run(df, ticker=ticker)

    v2_strategy = TrendFollowingV2(
        sma_fast=50,
        sma_slow=200,
        momentum_lookbacks=(63, 126),
        momentum_skip_days=10,
        sma_weight=0.6,
        momentum_weight=0.2,
        adx_weight=0.2,
        entry_threshold=0.08,
        exit_threshold=-0.05,
        adx_threshold=20.0,
        vol_threshold=0.015,
        rebalance_mode="monthly",
        trailing_stop_pct=0.08,
        signal_lag=1,
    )
    v2_engine = TrendFollowingV2Engine(
        strategy=v2_strategy,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        position_pct=position_pct,
    )
    v2_result = v2_engine.run(df, ticker=ticker)

    v3_strategy = TrendFollowingV3(
        sma_fast=50,
        sma_slow=200,
        adx_entry=21.0,
        adx_exit=17.0,
        adx_period=14,
        momentum_period=63,
        trailing_stop_pct=0.10,
        min_bars_between_trades=10,
        rebalance_monthly=True,
    )
    v3_engine = TrendFollowingV3Engine(
        strategy=v3_strategy,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        position_pct=position_pct,
    )
    v3_result = v3_engine.run(df, ticker=ticker)

    return {
        "V1": v1_result["metrics"],
        "V2": v2_result["metrics"],
        "V3": v3_result["metrics"],
    }


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


def cmd_backtest_trend(args):
    """Backtest using TrendFollowing custom engine."""
    from strategies.trend_following import TrendFollowingStrategy, TrendFollowingEngine

    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Running Trend Following backtest on {ticker}...")
    df = fetcher.fetch(ticker, period=args.period)
    if df.empty:
        print(f"No data for {ticker}")
        return

    strategy = TrendFollowingStrategy(
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        adx_period=args.adx_period,
        adx_threshold=args.adx_threshold,
        adx_exit_threshold=args.adx_exit,
        volume_factor=args.volume_factor,
        pullback_pct=args.pullback,
        trailing_stop_pct=args.trailing_stop,
    )
    engine = TrendFollowingEngine(
        strategy,
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
        position_pct=args.position_pct,
    )
    result = engine.run(df, ticker)
    print(engine.summary(result, ticker))


def cmd_backtest_quick(args):
    """Backtest using QuickTrade custom engine."""
    from strategies.quick_trade import QuickTradeStrategy, QuickTradeEngine

    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Running Quick Trade backtest on {ticker}...")
    df = fetcher.fetch(ticker, period=args.period)
    if df.empty:
        print(f"No data for {ticker}")
        return

    strategy = QuickTradeStrategy(
        rsi_period=args.rsi_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        bb_squeeze_threshold=args.bb_squeeze,
        volume_lookback=args.volume_lookback,
        price_lookback=args.price_lookback,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_hold_bars=args.max_hold,
    )
    engine = QuickTradeEngine(
        strategy,
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
        position_pct=args.position_pct,
    )
    result = engine.run(df, ticker)
    print(engine.summary(result, ticker))


def cmd_backtest_trend_v3(args):
    """Backtest using TrendFollowingV3 custom engine."""
    from strategies.trend_following_v3 import TrendFollowingV3, TrendFollowingV3Engine

    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Running Trend Following V3 backtest on {ticker}...")
    try:
        df = fetch_years_of_data(fetcher, ticker, args.years)
    except RuntimeError as exc:
        print(exc)
        return
    if df.empty:
        print(f"No data for {ticker}")
        return

    strategy = TrendFollowingV3(
        adx_entry=args.adx_entry,
        adx_exit=args.adx_exit,
        trailing_stop_pct=args.trailing_stop,
        rebalance_monthly=args.monthly,
        momentum_period=args.momentum_period,
    )
    engine = TrendFollowingV3Engine(
        strategy,
        initial_capital=args.initial_capital,
        commission=COMMISSION_RATE,
        slippage=SLIPPAGE,
        position_pct=0.95,
    )
    result = engine.run(df, ticker)
    print(engine.summary(result, ticker))


def cmd_compare_trend(args):
    """Compare V1, V2, and V3 trend strategies side-by-side."""
    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Comparing trend strategies on {ticker} ({args.years}y)...")
    try:
        df = fetch_years_of_data(fetcher, ticker, args.years)
    except RuntimeError as exc:
        print(exc)
        return
    if df.empty:
        print(f"No data for {ticker}")
        return

    print(
        f"Data: {df.index[0].strftime('%Y-%m-%d')} -> "
        f"{df.index[-1].strftime('%Y-%m-%d')} ({len(df)} bars)"
    )
    results = run_trend_comparison(
        df=df,
        ticker=ticker,
        initial_capital=args.initial_capital,
        commission=COMMISSION_RATE,
        slippage=SLIPPAGE,
        position_pct=0.95,
    )
    buy_hold_return = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
    buy_hold_annualized = ((1 + buy_hold_return) ** (252 / max(len(df) - 1, 1))) - 1

    print("\n" + "=" * 72)
    print(f"{'Metric':<22} {'V1':>12} {'V2':>12} {'V3':>12} {'Buy&Hold':>12}")
    print("-" * 72)
    print(
        f"{'Total Return':<22} "
        f"{results['V1']['total_return']:>+11.2%} "
        f"{results['V2']['total_return']:>+11.2%} "
        f"{results['V3']['total_return']:>+11.2%} "
        f"{buy_hold_return:>+11.2%}"
    )
    print(
        f"{'Annualized Return':<22} "
        f"{results['V1']['annualized_return']:>+11.2%} "
        f"{results['V2']['annualized_return']:>+11.2%} "
        f"{results['V3']['annualized_return']:>+11.2%} "
        f"{buy_hold_annualized:>+11.2%}"
    )
    print(
        f"{'Sharpe Ratio':<22} "
        f"{results['V1']['sharpe_ratio']:>11.2f} "
        f"{results['V2']['sharpe_ratio']:>11.2f} "
        f"{results['V3']['sharpe_ratio']:>11.2f} "
        f"{'N/A':>12}"
    )
    print(
        f"{'Max Drawdown':<22} "
        f"{results['V1']['max_drawdown']:>11.2%} "
        f"{results['V2']['max_drawdown']:>11.2%} "
        f"{results['V3']['max_drawdown']:>11.2%} "
        f"{'N/A':>12}"
    )
    print(
        f"{'Calmar Ratio':<22} "
        f"{results['V1']['calmar_ratio']:>11.2f} "
        f"{results['V2']['calmar_ratio']:>11.2f} "
        f"{results['V3']['calmar_ratio']:>11.2f} "
        f"{'N/A':>12}"
    )
    print(
        f"{'Round Trips':<22} "
        f"{results['V1']['round_trips']:>11d} "
        f"{results['V2']['round_trips']:>11d} "
        f"{results['V3']['round_trips']:>11d} "
        f"{'0':>12}"
    )
    print(
        f"{'Win Rate':<22} "
        f"{'N/A':>12} "
        f"{results['V2'].get('win_rate', 0.0):>11.1%} "
        f"{results['V3'].get('win_rate', 0.0):>11.1%} "
        f"{'N/A':>12}"
    )
    print(
        f"{'Profit Factor':<22} "
        f"{'N/A':>12} "
        f"{results['V2'].get('profit_factor', 0.0):>11.2f} "
        f"{results['V3'].get('profit_factor', 0.0):>11.2f} "
        f"{'N/A':>12}"
    )
    print(
        f"{'Exposure Ratio':<22} "
        f"{'N/A':>12} "
        f"{results['V2'].get('exposure_ratio', 0.0):>11.1%} "
        f"{results['V3'].get('exposure_ratio', 0.0):>11.1%} "
        f"{'N/A':>12}"
    )
    print(
        f"{'Commission':<22} "
        f"${results['V1']['total_commission']:>10,.0f} "
        f"${results['V2']['total_commission']:>10,.0f} "
        f"${results['V3']['total_commission']:>10,.0f} "
        f"{'$0':>12}"
    )
    print("=" * 72)


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


def cmd_volatility(args):
    """Analyze volatility using GARCH model."""
    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Fitting {args.garch_model.upper()}({args.p},{args.q}) on {ticker}...\n")
    df = fetcher.fetch(ticker, period=args.period)
    if df.empty:
        print(f"No data for {ticker}")
        return

    predictor = VolatilityPredictor(
        model_type=args.garch_model,
        p=args.p,
        q=args.q,
        vol_target=args.vol_target,
    )
    result = predictor.fit(df, ticker)
    print(predictor.summary(result))


def cmd_visualize(args):
    """Generate backtest visualization report."""
    from visualization.plots import BacktestVisualizer
    from strategies.ml_signal import MLSignalStrategy

    fetcher = DataFetcher()
    ticker = args.ticker or "QQQ"

    print(f"Generating {args.chart_type} chart for {ticker}...\n")
    df = fetcher.fetch(ticker, period=args.period)
    if df.empty:
        print(f"No data for {ticker}")
        return

    df = add_all_indicators(df)

    if args.strategy == "ml":
        strategy = MLSignalStrategy(
            model_type=args.model, lookback=ML_LOOKBACK,
            forecast_horizon=ML_FORECAST_HORIZON,
        )
    else:
        strategy = get_strategy(args.strategy)

    engine = BacktestEngine(strategy)
    result = engine.run(df, ticker)

    viz = BacktestVisualizer(output_dir=args.output)

    if args.chart_type == "full":
        path = viz.full_report(result, ticker, df)
    elif args.chart_type == "equity":
        path = viz.equity_curve_only(result, ticker, df)
    elif args.chart_type == "drawdown":
        path = viz.drawdown_only(result, ticker)
    else:
        path = viz.full_report(result, ticker, df)

    print(f"Chart saved to: {path}")


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

    # volatility
    p_vol = sub.add_parser("volatility", help="Analyze volatility with GARCH")
    p_vol.add_argument("ticker", nargs="?", help="Ticker symbol (default: QQQ)")
    p_vol.add_argument("--period", default="2y", help="Data period")
    p_vol.add_argument("--garch-model", default=GARCH_MODEL_TYPE,
                       choices=["garch", "gjr", "egarch"])
    p_vol.add_argument("--p", type=int, default=GARCH_P, help="ARCH order")
    p_vol.add_argument("--q", type=int, default=GARCH_Q, help="GARCH order")
    p_vol.add_argument("--vol-target", type=float, default=VOL_TARGET,
                       help="Target annualized volatility for position sizing")

    # visualize
    p_viz = sub.add_parser("visualize", help="Generate backtest charts")
    p_viz.add_argument("ticker", nargs="?", help="Ticker symbol (default: QQQ)")
    p_viz.add_argument("--period", default="2y", help="Data period")
    p_viz.add_argument("--strategy", "-s", default="momentum",
                       choices=list(STRATEGIES.keys()) + ["ml"],
                       help="Strategy to visualize")
    p_viz.add_argument("--model", "-m", default=ML_MODEL_TYPE,
                       choices=["xgboost", "random_forest", "gradient_boosting"])
    p_viz.add_argument("--chart-type", default="full",
                       choices=["full", "equity", "drawdown"],
                       help="Chart type to generate")
    p_viz.add_argument("--output", default="reports", help="Output directory")

    # backtest-trend
    p_bt_trend = sub.add_parser("backtest-trend", help="Backtest Trend Following strategy")
    p_bt_trend.add_argument("ticker", nargs="?", help="Ticker symbol (default: QQQ)")
    p_bt_trend.add_argument("--period", default="2y", help="Data period")
    p_bt_trend.add_argument("--sma-fast", type=int, default=50, help="Fast SMA period (default: 50)")
    p_bt_trend.add_argument("--sma-slow", type=int, default=200, help="Slow SMA period (default: 200)")
    p_bt_trend.add_argument("--adx-period", type=int, default=14, help="ADX period")
    p_bt_trend.add_argument("--adx-threshold", type=float, default=25.0, help="ADX threshold to enter")
    p_bt_trend.add_argument("--adx-exit", type=float, default=20.0, help="ADX threshold to exit")
    p_bt_trend.add_argument("--volume-factor", type=float, default=1.0, help="Volume confirmation multiplier")
    p_bt_trend.add_argument("--pullback", type=float, default=0.03, help="Pullback tolerance (e.g., 0.03 = 3%)")
    p_bt_trend.add_argument("--trailing-stop", type=float, default=0.12, help="Trailing stop % from peak")
    p_bt_trend.add_argument("--position-pct", type=float, default=0.95, help="Position size % of capital")
    p_bt_trend.add_argument("--commission", type=float, default=COMMISSION_RATE)
    p_bt_trend.add_argument("--slippage", type=float, default=SLIPPAGE)
    p_bt_trend.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")

    # backtest-quick
    p_bt_quick = sub.add_parser("backtest-quick", help="Backtest Quick Trade strategy")
    p_bt_quick.add_argument("ticker", nargs="?", help="Ticker symbol (default: QQQ)")
    p_bt_quick.add_argument("--period", default="2y", help="Data period")
    p_bt_quick.add_argument("--rsi-period", type=int, default=14)
    p_bt_quick.add_argument("--rsi-oversold", type=float, default=30.0)
    p_bt_quick.add_argument("--rsi-overbought", type=float, default=70.0)
    p_bt_quick.add_argument("--bb-period", type=int, default=20)
    p_bt_quick.add_argument("--bb-std", type=float, default=2.0)
    p_bt_quick.add_argument("--bb-squeeze", type=float, default=0.04, help="BB width squeeze threshold")
    p_bt_quick.add_argument("--volume-lookback", type=int, default=20)
    p_bt_quick.add_argument("--price-lookback", type=int, default=10)
    p_bt_quick.add_argument("--stop-loss", type=float, default=0.03, help="Stop loss %")
    p_bt_quick.add_argument("--take-profit", type=float, default=0.05, help="Take profit %")
    p_bt_quick.add_argument("--max-hold", type=int, default=14, help="Max holding bars")
    p_bt_quick.add_argument("--position-pct", type=float, default=0.30, help="Position size % of capital")
    p_bt_quick.add_argument("--commission", type=float, default=COMMISSION_RATE)
    p_bt_quick.add_argument("--slippage", type=float, default=SLIPPAGE)
    p_bt_quick.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")

    # backtest-trend-v3
    p_bt_trend_v3 = sub.add_parser("backtest-trend-v3", help="Backtest Trend Following V3 strategy")
    p_bt_trend_v3.add_argument("--ticker", default="QQQ", help="Ticker symbol (default: QQQ)")
    p_bt_trend_v3.add_argument("--years", type=int, default=5, help="How many years of data")
    p_bt_trend_v3.add_argument("--adx-entry", type=float, default=21.0, help="ADX threshold for entry")
    p_bt_trend_v3.add_argument("--adx-exit", type=float, default=17.0, help="ADX threshold for exit")
    p_bt_trend_v3.add_argument("--trailing-stop", type=float, default=0.10, help="Trailing stop percentage")
    p_bt_trend_v3.add_argument("--monthly", dest="monthly", action="store_true", default=True,
                               help="Enable monthly rebalance entry mode")
    p_bt_trend_v3.add_argument("--no-monthly", dest="monthly", action="store_false",
                               help="Disable monthly rebalance entry mode")
    p_bt_trend_v3.add_argument("--momentum-period", type=int, default=63, help="Momentum lookback in days")
    p_bt_trend_v3.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")

    # compare-trend
    p_compare_trend = sub.add_parser("compare-trend", help="Compare V1, V2, and V3 trend strategies")
    p_compare_trend.add_argument("--ticker", default="QQQ", help="Ticker symbol (default: QQQ)")
    p_compare_trend.add_argument("--years", type=int, default=5, help="How many years of data")
    p_compare_trend.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")

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
        "volatility": cmd_volatility,
        "visualize": cmd_visualize,
        "backtest-trend": cmd_backtest_trend,
        "backtest-quick": cmd_backtest_quick,
        "backtest-trend-v3": cmd_backtest_trend_v3,
        "compare-trend": cmd_compare_trend,
        "cache": cmd_cache,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
