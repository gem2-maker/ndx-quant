"""Configuration for ndx-quant."""

# Data settings
DATA_DIR = "cache"
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"

# Backtesting
INITIAL_CAPITAL = 100_000.0
COMMISSION_RATE = 0.001  # 0.1% per trade
SLIPPAGE = 0.0005        # 0.05% slippage

# Risk management
MAX_POSITION_PCT = 0.10   # Max 10% in single stock
STOP_LOSS_PCT = 0.05      # 5% stop loss
TAKE_PROFIT_PCT = 0.15    # 15% take profit

# Indicators
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
SMA_SHORT = 20
SMA_LONG = 50
EMA_SHORT = 12
EMA_LONG = 26

# ML / Prediction
ML_MODEL_TYPE = "xgboost"       # "xgboost" | "random_forest" | "gradient_boosting"
ML_LOOKBACK = 20                # Rolling window for features
ML_FORECAST_HORIZON = 5         # Bars ahead to predict
ML_TEST_SIZE = 0.2              # Test set fraction
ML_N_ESTIMATORS = 200           # Number of trees
ML_MAX_DEPTH = 6                # Tree depth
MODEL_DIR = "models"

# GARCH / Volatility
GARCH_MODEL_TYPE = "garch"      # "garch" | "gjr" | "egarch"
GARCH_P = 1                     # ARCH order
GARCH_Q = 1                     # GARCH order
VOL_TARGET = 0.15               # Target annualized volatility for position sizing
