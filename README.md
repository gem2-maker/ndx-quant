# ndx-quant

Nasdaq-100 (NDX) Quantitative Trading Toolkit.

Data-driven strategies, backtesting, and portfolio analysis for the Nasdaq-100 index.

## Features

- **Data Fetching** — Historical price data via Yahoo Finance (free, no API key)
- **Technical Indicators** — SMA, EMA, RSI, MACD, Bollinger Bands, ATR, and more
- **Strategy Engine** — Modular strategy framework with signal generation
- **Backtesting** — Event-driven backtester with performance metrics
- **Portfolio Analysis** — Correlation, risk metrics, sector breakdown

## Quick Start

```bash
pip install -r requirements.txt

python main.py fetch
python main.py backtest --strategy momentum
python main.py analyze
streamlit run streamlit_app.py
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
├── data/           # Price data fetching & caching
├── indicators/     # Technical analysis indicators
├── strategies/     # Trading strategies
├── backtest/       # Backtesting engine
├── portfolio/      # Portfolio risk & analysis
├── config.py       # Configuration
└── main.py         # CLI entry point
```

## Disclaimer

This is an educational/research tool. Not financial advice. Past performance does not guarantee future results.

---
Built by [BaoShi GEM](https://github.com/gem2-maker) 💎
