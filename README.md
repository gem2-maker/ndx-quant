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

# Fetch NDX constituents and price data
python main.py fetch

# Run a backtest
python main.py backtest --strategy momentum

# Analyze portfolio
python main.py analyze
```

## Project Structure

```
ndx-quant/
├── data/
│   ├── fetcher.py      # Price data fetching & caching
│   └── constituents.py # NDX-100 stock list
├── indicators/
│   └── technical.py    # Technical analysis indicators
├── strategies/
│   ├── base.py         # Base strategy class
│   ├── momentum.py     # Momentum strategy
│   ├── mean_reversion.py # Mean reversion strategy
│   └── rsi_macd.py     # RSI + MACD combo
├── backtest/
│   └── engine.py       # Backtesting engine
├── portfolio/
│   └── analyzer.py     # Portfolio risk & analysis
├── config.py           # Configuration
├── main.py             # CLI entry point
└── requirements.txt
```

## Disclaimer

This is an educational/research tool. Not financial advice. Past performance does not guarantee future results.

---
Built with 💎 by OpenClaw
