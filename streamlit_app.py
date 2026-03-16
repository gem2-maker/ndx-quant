"""Streamlit dashboard for ndx-quant."""

from __future__ import annotations

from dataclasses import asdict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from backtest.engine import BacktestEngine
from config import (
    GARCH_MODEL_TYPE,
    GARCH_P,
    GARCH_Q,
    INITIAL_CAPITAL,
    ML_FORECAST_HORIZON,
    ML_LOOKBACK,
    ML_MAX_DEPTH,
    ML_MODEL_TYPE,
    ML_N_ESTIMATORS,
    ML_TEST_SIZE,
    VOL_TARGET,
)
from data import NDX_TICKERS, get_tickers
from data.fetcher import DataFetcher
from indicators.technical import add_all_indicators
from ml.predictor import HAS_XGB, TrendPredictor
from portfolio.analyzer import PortfolioAnalyzer
from strategies import STRATEGIES, get_strategy
from strategies.base import BacktestResult
from visualization.plots import BacktestVisualizer

try:
    from ml.evaluate import ModelEvaluator
except ImportError:
    ModelEvaluator = None

try:
    from ml.volatility import VolatilityPredictor
except ImportError:
    VolatilityPredictor = None


st.set_page_config(
    page_title="ndx-quant Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
SECTOR_OPTIONS = ["all", "tech", "consumer", "healthcare", "communication"]
MODEL_OPTIONS = ["random_forest", "gradient_boosting"] + (["xgboost"] if HAS_XGB else [])


@st.cache_resource
def get_fetcher() -> DataFetcher:
    return DataFetcher()


@st.cache_resource
def get_analyzer() -> PortfolioAnalyzer:
    return PortfolioAnalyzer()


@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, period: str) -> pd.DataFrame:
    df = get_fetcher().fetch(ticker, period=period)
    return add_all_indicators(df) if not df.empty else df


def compute_backtest(
    df: pd.DataFrame,
    ticker: str,
    strategy_name: str,
    initial_capital: float,
    model_type: str,
    lookback: int,
    horizon: int,
    confidence: str,
    retrain: int,
    n_estimators: int,
    max_depth: int,
) -> tuple[BacktestResult | dict, object]:
    kwargs = {}
    if strategy_name == "ml_signal":
        kwargs.update(
            model_type=model_type,
            lookback=lookback,
            forecast_horizon=horizon,
            confidence_threshold=confidence,
            retrain_interval=retrain,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

    strategy = get_strategy(strategy_name, **kwargs)
    if strategy_name == "trend_following_v3":
        from strategies.trend_following_v3 import TrendFollowingV3Engine

        engine = TrendFollowingV3Engine(strategy=strategy, initial_capital=initial_capital)
        return engine.run(df, ticker=ticker), strategy

    engine = BacktestEngine(strategy=strategy, initial_capital=initial_capital)
    return engine.run(df, ticker=ticker), strategy


def summarize_backtest(result: BacktestResult | dict) -> dict[str, float]:
    metrics = result["metrics"].copy() if isinstance(result, dict) else result.metrics.copy()
    metrics["sharpe"] = metrics.get("sharpe_ratio", 0.0)
    metrics["commission"] = metrics.get("total_commission", 0.0)
    return metrics


def build_trade_frame(result: BacktestResult | dict) -> pd.DataFrame:
    if isinstance(result, dict):
        return pd.DataFrame(result.get("trades", []))

    records = [
        {
            "entry_date": trade.entry_date,
            "exit_date": trade.exit_date,
            "hold_days": trade.hold_period_days,
            "hold_bars": trade.hold_period_bars,
            "shares": trade.shares,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "gross_pnl": trade.gross_pnl,
            "net_pnl": trade.net_pnl,
            "return_pct": trade.return_pct,
            "mfe_pct": trade.max_favorable_excursion_pct,
            "mae_pct": trade.max_adverse_excursion_pct,
            "commission": trade.total_commission,
            "exit_reason": trade.exit_reason,
        }
        for trade in result.completed_trades
    ]
    return pd.DataFrame(records)


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def metric_row(metrics: dict[str, float]) -> None:
    cols = st.columns(5)
    cols[0].metric("Last Close", f"${metrics['last_close']:,.2f}", format_pct(metrics["period_return"]))
    cols[1].metric("RSI", f"{metrics['rsi']:.1f}")
    cols[2].metric("MACD Hist", f"{metrics['macd_hist']:.3f}")
    cols[3].metric("ATR", f"{metrics['atr']:.2f}")
    cols[4].metric("Volume", f"{metrics['volume']:,.0f}")


def plot_price_panel(df: pd.DataFrame, ticker: str) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    axes[0].plot(df.index, df["Close"], label="Close", color="#1f77b4", linewidth=1.8)
    axes[0].plot(df.index, df["SMA_20"], label="SMA 20", color="#ff7f0e", linewidth=1.1)
    axes[0].plot(df.index, df["SMA_50"], label="SMA 50", color="#2ca02c", linewidth=1.1)
    axes[0].fill_between(df.index, df["BB_Lower"], df["BB_Upper"], color="#cfe2f3", alpha=0.35, label="Bollinger")
    axes[0].set_title(f"{ticker} Price and Trend")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.2)

    axes[1].plot(df.index, df["RSI"], color="#9467bd", linewidth=1.2)
    axes[1].axhline(70, color="#d62728", linestyle="--", linewidth=0.9)
    axes[1].axhline(30, color="#2ca02c", linestyle="--", linewidth=0.9)
    axes[1].set_title("RSI")
    axes[1].grid(alpha=0.2)

    axes[2].plot(df.index, df["MACD"], label="MACD", color="#17becf", linewidth=1.1)
    axes[2].plot(df.index, df["MACD_Signal"], label="Signal", color="#7f7f7f", linewidth=1.0)
    axes[2].bar(df.index, df["MACD_Hist"], color="#9ecae1", alpha=0.8)
    axes[2].set_title("MACD")
    axes[2].legend(loc="upper left")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_equity_panel(result: BacktestResult | dict, ticker: str, df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    if isinstance(result, dict):
        equity_curve = result["equity_curve"]
        drawdown = (equity_curve / equity_curve.cummax()) - 1
        buy_hold = (df["Close"] / df["Close"].iloc[0]) * equity_curve.iloc[0]

        axes[0].plot(equity_curve.index, equity_curve.values, label="Strategy", color="#1f77b4", linewidth=1.8)
        axes[0].plot(buy_hold.index, buy_hold.values, label="Buy & Hold", color="#7f7f7f", linestyle="--", linewidth=1.0)
        axes[0].set_title(f"{ticker} Equity Curve")
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.2)

        axes[1].fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.35)
        axes[1].set_title("Drawdown")
        axes[1].grid(alpha=0.2)
        fig.tight_layout()
        return fig

    visualizer = BacktestVisualizer()
    visualizer._plot_equity_curve(axes[0], result, ticker, df)
    visualizer._plot_drawdown(axes[1], result)
    fig.tight_layout()
    return fig


def render_header() -> None:
    st.title("ndx-quant Dashboard")
    st.caption("Interactive research workspace for Nasdaq-100 data, backtests, risk analysis, and ML signals.")


def render_sidebar() -> dict[str, object]:
    with st.sidebar:
        st.header("Controls")
        ticker = st.selectbox("Ticker", ["QQQ"] + NDX_TICKERS, index=0)
        period = st.selectbox("Period", PERIOD_OPTIONS, index=4)
        strategy = st.selectbox("Strategy", list(STRATEGIES.keys()), index=0)
        initial_capital = st.number_input("Initial capital", min_value=10_000.0, value=float(INITIAL_CAPITAL), step=10_000.0)

        st.subheader("ML Strategy")
        model_type = st.selectbox("Model type", MODEL_OPTIONS, index=min(MODEL_OPTIONS.index(ML_MODEL_TYPE), len(MODEL_OPTIONS) - 1) if ML_MODEL_TYPE in MODEL_OPTIONS else 0)
        lookback = st.slider("Lookback", 5, 60, ML_LOOKBACK)
        horizon = st.slider("Forecast horizon", 1, 20, ML_FORECAST_HORIZON)
        confidence = st.selectbox("Confidence filter", ["low", "medium", "high"], index=0)
        retrain = st.slider("Retrain interval", 0, 120, 0, 5)
        n_estimators = st.slider("Estimators", 50, 500, ML_N_ESTIMATORS, 25)
        max_depth = st.slider("Max depth", 2, 12, ML_MAX_DEPTH)

        st.subheader("Universe")
        sector = st.selectbox("Sector filter", SECTOR_OPTIONS, index=0)
        top_n = st.slider("Momentum top N", 3, 20, 10)
        compare_tickers = st.multiselect(
            "Correlation basket",
            NDX_TICKERS,
            default=["AAPL", "MSFT", "NVDA", "AMZN"],
        )

    return {
        "ticker": ticker,
        "period": period,
        "strategy": strategy,
        "initial_capital": initial_capital,
        "model_type": model_type,
        "lookback": lookback,
        "horizon": horizon,
        "confidence": confidence,
        "retrain": retrain,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "sector": None if sector == "all" else sector,
        "top_n": top_n,
        "compare_tickers": compare_tickers,
    }


def render_market_overview(df: pd.DataFrame, ticker: str) -> None:
    latest = df.iloc[-1]
    metrics = {
        "last_close": float(latest["Close"]),
        "period_return": float(df["Close"].iloc[-1] / df["Close"].iloc[0] - 1),
        "rsi": float(latest["RSI"]),
        "macd_hist": float(latest["MACD_Hist"]),
        "atr": float(latest["ATR"]) if pd.notna(latest.get("ATR")) else 0.0,
        "volume": float(latest["Volume"]) if pd.notna(latest.get("Volume")) else 0.0,
    }
    metric_row(metrics)
    st.pyplot(plot_price_panel(df.tail(252), ticker), use_container_width=True)

    st.subheader("Latest bars")
    st.dataframe(
        df.tail(20)[["Open", "High", "Low", "Close", "Volume", "SMA_20", "SMA_50", "RSI", "MACD", "MACD_Hist"]]
        .sort_index(ascending=False),
        use_container_width=True,
    )


def render_backtest_tab(df: pd.DataFrame, controls: dict[str, object]) -> None:
    result, strategy = compute_backtest(
        df=df,
        ticker=controls["ticker"],
        strategy_name=controls["strategy"],
        initial_capital=controls["initial_capital"],
        model_type=controls["model_type"],
        lookback=controls["lookback"],
        horizon=controls["horizon"],
        confidence=controls["confidence"],
        retrain=controls["retrain"],
        n_estimators=controls["n_estimators"],
        max_depth=controls["max_depth"],
    )
    summary = summarize_backtest(result)

    cols = st.columns(5)
    cols[0].metric("Total return", format_pct(summary["total_return"]))
    cols[1].metric("Final equity", f"${summary['final_equity']:,.0f}")
    cols[2].metric("Sharpe", f"{summary['sharpe']:.2f}")
    cols[3].metric("Max drawdown", format_pct(summary["max_drawdown"]))
    cols[4].metric("Win rate", format_pct(summary["win_rate"]))

    detail_cols = st.columns(5)
    detail_cols[0].metric("Profit factor", f"{summary.get('profit_factor', 0.0):.2f}")
    detail_cols[1].metric("Expectancy", f"${summary.get('expectancy', 0.0):,.0f}")
    detail_cols[2].metric("Avg trade", f"${summary.get('average_trade_pnl', 0.0):,.0f}")
    detail_cols[3].metric("Avg hold", f"{summary.get('average_hold_days', 0.0):.1f}d")
    detail_cols[4].metric("Exposure", format_pct(summary["exposure_ratio"]))

    st.pyplot(plot_equity_panel(result, controls["ticker"], df), use_container_width=True)

    trade_df = build_trade_frame(result)
    info_col, trade_col = st.columns([1, 2])
    with info_col:
        st.markdown(f"**Strategy:** `{strategy.name}`")
        st.markdown(f"**Buy trades:** `{summary.get('buy_trades', summary.get('total_buys', 0))}`")
        st.markdown(f"**Sell trades:** `{summary.get('sell_trades', summary.get('total_sells', 0))}`")
        st.markdown(f"**Completed trades:** `{summary.get('completed_trades', summary.get('round_trips', 0))}`")
        st.markdown(f"**Commission paid:** `${summary['commission']:,.2f}`")
        st.markdown(
            f"**Best / Worst trade:** "
            f"`${summary.get('best_trade', 0.0):,.0f}` / `${summary.get('worst_trade', 0.0):,.0f}`"
        )
        st.markdown(
            f"**Consecutive wins / losses:** "
            f"`{summary.get('max_consecutive_wins', 0)}` / `{summary.get('max_consecutive_losses', 0)}`"
        )
        if hasattr(strategy, "training_info"):
            st.json(strategy.training_info)
    with trade_col:
        st.subheader("Completed trades")
        st.dataframe(
            trade_df.sort_values("exit_date", ascending=False) if not trade_df.empty else trade_df,
            use_container_width=True,
        )


def render_risk_tab(controls: dict[str, object]) -> None:
    analyzer = get_analyzer()
    metrics = analyzer.risk_metrics(controls["ticker"], period=controls["period"])
    if not metrics:
        st.warning("No risk data available for the selected ticker.")
        return

    cols = st.columns(5)
    cols[0].metric("Annual return", format_pct(metrics["annual_return"]))
    cols[1].metric("Annual vol", format_pct(metrics["annual_volatility"]))
    cols[2].metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
    cols[3].metric("Sortino", f"{metrics['sortino_ratio']:.2f}")
    cols[4].metric("Max drawdown", format_pct(metrics["max_drawdown"]))

    stats_df = pd.DataFrame(
        [
            ("VaR 95%", format_pct(metrics["var_95"])),
            ("Best day", format_pct(metrics["best_day"])),
            ("Worst day", format_pct(metrics["worst_day"])),
            ("Positive days", format_pct(metrics["positive_days"])),
        ],
        columns=["Metric", "Value"],
    )

    left, right = st.columns([1, 2])
    with left:
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    with right:
        basket = controls["compare_tickers"] or ["AAPL", "MSFT", "NVDA", "AMZN"]
        corr = analyzer.correlation_matrix(basket, period=controls["period"])
        if corr.empty:
            st.info("Correlation matrix is unavailable for the current basket.")
        else:
            st.subheader("Correlation matrix")
            st.dataframe(corr.style.format("{:.2f}").background_gradient(cmap="Blues"), use_container_width=True)


def render_momentum_tab(controls: dict[str, object]) -> None:
    analyzer = get_analyzer()
    tickers = get_tickers(controls["sector"])
    st.caption(f"Universe size: {len(tickers)} tickers")
    with st.spinner("Scanning momentum leaders..."):
        top = analyzer.top_momentum(tickers, period=controls["period"], top_n=controls["top_n"])

    if not top:
        st.warning("Momentum scan returned no results.")
        return

    momentum_df = pd.DataFrame(top, columns=["Ticker", "Return"]).set_index("Ticker")
    st.bar_chart(momentum_df)
    st.dataframe(momentum_df.style.format({"Return": "{:.2%}"}), use_container_width=True)


def render_ml_tab(df: pd.DataFrame, controls: dict[str, object]) -> None:
    predictor = TrendPredictor(
        model_type=controls["model_type"],
        lookback=controls["lookback"],
        forecast_horizon=controls["horizon"],
        test_size=ML_TEST_SIZE,
        n_estimators=controls["n_estimators"],
        max_depth=controls["max_depth"],
    )

    try:
        training = predictor.train(df)
        predictions = predictor.predict(df, top_n=5)
    except Exception as exc:
        st.error(f"ML workflow failed: {exc}")
        return

    cols = st.columns(4)
    cols[0].metric("Train accuracy", format_pct(training.train_accuracy))
    cols[1].metric("Test accuracy", format_pct(training.test_accuracy))
    cols[2].metric("CV mean", format_pct(sum(training.cross_val_scores) / len(training.cross_val_scores)))
    cols[3].metric("Features", str(training.n_features))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Latest predictions")
        pred_df = pd.DataFrame(
            [
                {
                    "date": pred.date,
                    "signal": pred.signal,
                    "prob_up": pred.probability,
                    "confidence": pred.confidence,
                }
                for pred in predictions
            ]
        )
        st.dataframe(pred_df.sort_values("date", ascending=False).style.format({"prob_up": "{:.2%}"}), use_container_width=True)

    with right:
        st.subheader("Top features")
        feat_df = pd.DataFrame(
            list(training.feature_importances.items()),
            columns=["feature", "importance"],
        ).head(10)
        if feat_df.empty:
            st.info("Feature importance is unavailable for this model.")
        else:
            st.bar_chart(feat_df.set_index("feature"))

    if ModelEvaluator is not None:
        try:
            evaluator = ModelEvaluator(predictor)
            metrics = evaluator.evaluate(df)
            st.subheader("Evaluation")
            st.json(asdict(metrics))
        except Exception:
            pass


def render_volatility_tab(df: pd.DataFrame) -> None:
    if VolatilityPredictor is None:
        st.warning("Volatility analysis requires the `arch` package.")
        return

    predictor = VolatilityPredictor(
        model_type=GARCH_MODEL_TYPE,
        p=GARCH_P,
        q=GARCH_Q,
        vol_target=VOL_TARGET,
    )
    try:
        result = predictor.fit(df)
    except Exception as exc:
        st.error(f"Volatility analysis failed: {exc}")
        return

    sizing = predictor.position_sizing_signal(result.forecast)
    cols = st.columns(5)
    cols[0].metric("Current vol", format_pct(result.forecast.current_vol))
    cols[1].metric("Forecast vol", format_pct(result.forecast.forecast_vol))
    cols[2].metric("Regime", result.forecast.regime.upper())
    cols[3].metric("Risk score", f"{result.forecast.risk_score:.0f}/100")
    cols[4].metric("Scale factor", f"{sizing['scale_factor']}x")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.conditional_vol.index, result.conditional_vol.values, color="#d62728", linewidth=1.3)
    ax.set_title("Conditional Volatility")
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)

    st.json(
        {
            "forecast": asdict(result.forecast),
            "persistence": result.persistence,
            "half_life_days": result.half_life,
            "position_sizing": sizing,
        }
    )


def main() -> None:
    render_header()
    controls = render_sidebar()

    with st.spinner(f"Loading {controls['ticker']} price history..."):
        df = load_price_data(controls["ticker"], controls["period"])

    if df.empty:
        st.error("No price data was returned for the selected ticker.")
        return

    tabs = st.tabs(["Market", "Backtest", "Risk", "Momentum", "ML", "Volatility"])
    with tabs[0]:
        render_market_overview(df, controls["ticker"])
    with tabs[1]:
        render_backtest_tab(df, controls)
    with tabs[2]:
        render_risk_tab(controls)
    with tabs[3]:
        render_momentum_tab(controls)
    with tabs[4]:
        render_ml_tab(df, controls)
    with tabs[5]:
        render_volatility_tab(df)


if __name__ == "__main__":
    main()
