# ndx-quant

Nasdaq-100 (NDX) Quantitative Trading Toolkit.

Data-driven strategies, backtesting, and portfolio analysis for the Nasdaq-100 index.

## Features

- **Data Fetching** — Historical price data via Yahoo Finance (free, no API key), SQLite caching
- **Technical Indicators** — SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more
- **Strategy Engine** — Modular strategy framework (momentum, mean reversion, RSI+MACD, ML-driven)
- **Backtesting** — Event-driven backtester with performance metrics and trade blotter
- **Portfolio Analysis** — Correlation, risk metrics, sector breakdown
- **ML Prediction** — XGBoost / Random Forest / Gradient Boosting trend prediction with walk-forward validation
- **Volatility Modeling** — GARCH/GJR-GARCH/EGARCH volatility forecasting and position sizing
- **Visualization** — Publication-quality charts (equity curve, drawdown, monthly heatmap, trade analysis)
- **Interactive Dashboard** — Streamlit-based research workspace with 6 analysis tabs

## Quick Start

```bash
pip install -r requirements.txt

python main.py fetch
python main.py backtest --strategy momentum
python main.py analyze
streamlit run streamlit_app.py
```

## CLI Commands

```bash
python main.py fetch [TICKER]          # Fetch price data (default: QQQ)
python main.py backtest -s momentum    # Run backtest with a strategy
python main.py train [TICKER] --save   # Train & save ML model
python main.py predict [TICKER]        # Predict trend direction
python main.py backtest-ml [TICKER]    # Backtest with ML-driven strategy
python main.py volatility [TICKER]     # GARCH volatility analysis
python main.py analyze [TICKER]        # Risk metrics
python main.py momentum               # Top momentum stocks in NDX-100
python main.py strategies             # List available strategies
python main.py visualize [TICKER]     # Generate backtest charts
python main.py cache stats            # Cache statistics
```

## Dashboard

Launch the interactive research dashboard with:

```bash
streamlit run streamlit_app.py
```

The dashboard includes:

- Market view with price, Bollinger Bands, RSI, and MACD
- Strategy backtesting with trade blotter and equity/drawdown charts
- Risk metrics and correlation matrix
- Momentum scan across the NDX universe or sector subsets
- ML model training, recent predictions, and feature importance
- Volatility regime analysis and position sizing guidance

## Project Structure

```
ndx-quant/
├── data/              # Price data fetching & SQLite caching
├── indicators/        # Technical analysis indicators
├── strategies/        # Trading strategies (momentum, mean_reversion, rsi_macd, ml_signal)
├── backtest/          # Event-driven backtesting engine
├── portfolio/         # Portfolio risk & correlation analysis
├── ml/                # ML prediction (features, predictor, evaluator, volatility/GARCH)
├── visualization/     # Backtest charts (equity, drawdown, heatmap, trade analysis)
├── cache/             # SQLite market data cache
├── models/            # Saved ML models (.pkl)
├── streamlit_app.py   # Interactive dashboard (streamlit run streamlit_app.py)
├── config.py          # Configuration & defaults
├── main.py            # CLI entry point
└── requirements.txt   # Python dependencies
```

## Disclaimer

This is an educational/research tool. Not financial advice. Past performance does not guarantee future results.

---
Built by [BaoShi GEM](https://github.com/gem2-maker) 💎
