# Codex Task: Integrate TrendFollowingV3 into ndx-quant pipeline

## Background
We have created a new strategy `strategies/trend_following_v3.py` that improves on the original `trend_following.py`:
- Uses SMA50/200 crossover + ADX filter + 63-day momentum confirmation + monthly rebalance entry
- Precomputes all indicators for O(1) access during backtest (faster)
- Proven results on QQQ 5y data: Sharpe 0.99, MaxDD -15.22%, +53% total return, 75% win rate

## Tasks

### 1. Register V3 in `strategies/__init__.py`
Add TrendFollowingV3 and TrendFollowingV3Engine to the module exports so other code can import them.

### 2. Add CLI command `backtest-trend-v3` in `main.py`
Following the existing pattern (e.g., `backtest-trend`, `backtest-quick`), add a new command:
```
backtest-trend-v3 --ticker QQQ --years 5 --adx-entry 21 --adx-exit 17 --trailing-stop 0.10 --no-monthly
```
Parameters:
- `--ticker` (default: QQQ)
- `--years` (default: 5) - how many years of data
- `--adx-entry` (default: 21.0) - ADX threshold for entry
- `--adx-exit` (default: 17.0) - ADX threshold for exit
- `--trailing-stop` (default: 0.10) - trailing stop percentage
- `--monthly/--no-monthly` (default: monthly) - monthly rebalance entry mode
- `--momentum-period` (default: 63) - momentum lookback in days
- `--initial-capital` (default: 100000)

### 3. Add V3 to the streamlit dashboard (if applicable)
If `streamlit_app.py` has strategy selection, add V3 as an option.

### 4. Add a comparison mode
Add command `compare-trend` that runs V1, V2, V3 side-by-side and prints a comparison table (the logic already exists in `compare_trend.py`, just integrate it into main.py as a CLI command).

## File locations
- Strategy code: `D:\openclaw\workspace\ndx-quant\strategies\trend_following_v3.py` (already exists)
- Entry point: `D:\openclaw\workspace\ndx-quant\main.py`
- Strategies init: `D:\openclaw\workspace\ndx-quant\strategies\__init__.py`
- Comparison script: `D:\openclaw\workspace\ndx-quant\compare_trend.py` (reference for comparison logic)
- Config: `D:\openclaw\workspace\ndx-quant\config.py`

## Code style
- Follow existing patterns in the codebase
- Use DataFetcher for data retrieval
- Use the same output formatting as other backtest commands
- Don't add unnecessary dependencies

## IMPORTANT
- Do NOT modify trend_following.py or trend_following_v2.py
- Do NOT modify the base strategy class
- Only add/modify files needed for integration
- Test that `python main.py backtest-trend-v3` runs without errors
