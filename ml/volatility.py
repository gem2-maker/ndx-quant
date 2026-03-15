"""Volatility prediction using GARCH models.

Provides GARCH(1,1) volatility forecasting with rolling window estimation,
volatility regime detection, and risk-adjusted position sizing signals.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal


@dataclass
class VolatilityForecast:
    """Container for volatility forecast results."""
    current_vol: float          # Current realized volatility (annualized)
    forecast_vol: float         # Forecasted volatility (annualized)
    forecast_horizon: int       # Forecast horizon in days
    regime: str                 # "low", "normal", "high", "extreme"
    confidence_interval: tuple  # (lower, upper) bounds at 95%
    risk_score: float           # 0-100 risk score


@dataclass
class VolatilityResult:
    """Full volatility analysis result."""
    ticker: str
    conditional_vol: pd.Series  # GARCH conditional volatility (annualized)
    forecast: VolatilityForecast
    model_params: dict          # GARCH model parameters
    half_life: float            # Volatility half-life in days
    persistence: float          # GARCH persistence (alpha + beta)


class VolatilityPredictor:
    """GARCH-based volatility prediction engine.

    Supports GARCH(1,1), GJR-GARCH (asymmetric), and EGARCH models.
    Provides regime detection and risk scoring for position sizing.
    """

    # Volatility regime thresholds (annualized vol)
    REGIME_THRESHOLDS = {
        "low": 0.10,       # < 10% annualized
        "normal": 0.20,    # 10-20%
        "high": 0.35,      # 20-35%
        # > 35% = extreme
    }

    def __init__(
        self,
        model_type: str = "garch",
        p: int = 1,
        q: int = 1,
        vol_target: float = 0.15,
    ):
        """
        Args:
            model_type: 'garch', 'gjr' (GJR-GARCH), or 'egarch'
            p: ARCH order
            q: GARCH order
            vol_target: Target annualized volatility for position sizing
        """
        self.model_type = model_type
        self.p = p
        self.q = q
        self.vol_target = vol_target
        self._model = None
        self._result = None

    def fit(self, df: pd.DataFrame, ticker: str = "TICKER") -> VolatilityResult:
        """Fit GARCH model on price data.

        Args:
            df: DataFrame with 'Close' column
            ticker: Ticker symbol for labeling

        Returns:
            VolatilityResult with conditional volatility and forecast
        """
        # Calculate log returns (percentage)
        returns = 100 * np.log(df["Close"] / df["Close"].shift(1)).dropna()

        # Build GARCH model
        if self.model_type == "egarch":
            vol = GARCH(p=self.p, q=self.q, o=1, power=1.0)
        elif self.model_type == "gjr":
            vol = GARCH(p=self.p, q=self.q, o=1)
        else:
            vol = GARCH(p=self.p, q=self.q)

        self._model = ConstantMean(returns, volatility=vol, distribution=Normal())
        self._result = self._model.fit(disp="off")

        # Extract conditional volatility (annualized)
        cond_vol = self._result.conditional_volatility * np.sqrt(252)

        # Forecast next period
        forecast_result = self._model.forecast(horizon=5, reindex=False)
        forecast_var = forecast_result.variance.iloc[-1, 0]  # Next-day variance
        forecast_vol = np.sqrt(forecast_var * 252)  # Annualized

        # Current realized volatility (20-day rolling)
        realized_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

        # Confidence interval (assume normal, 95%)
        forecast_std = np.sqrt(forecast_var) * np.sqrt(252) * 0.5  # Rough estimate
        ci_lower = max(0, forecast_vol - 1.96 * forecast_std)
        ci_upper = forecast_vol + 1.96 * forecast_std

        # Regime detection
        regime = self._detect_regime(forecast_vol)

        # Risk score (0-100)
        risk_score = min(100, max(0, (forecast_vol / 0.50) * 100))

        # Model parameters
        params = dict(self._result.params)

        # Persistence and half-life
        persistence = self._calc_persistence(params)
        half_life = self._calc_half_life(persistence)

        forecast = VolatilityForecast(
            current_vol=realized_vol if not np.isnan(realized_vol) else cond_vol.iloc[-1],
            forecast_vol=forecast_vol,
            forecast_horizon=5,
            regime=regime,
            confidence_interval=(ci_lower, ci_upper),
            risk_score=risk_score,
        )

        return VolatilityResult(
            ticker=ticker,
            conditional_vol=cond_vol,
            forecast=forecast,
            model_params=params,
            half_life=half_life,
            persistence=persistence,
        )

    def rolling_forecast(
        self,
        df: pd.DataFrame,
        window: int = 252,
        horizon: int = 5,
    ) -> pd.Series:
        """Rolling out-of-sample volatility forecasts.

        Fits GARCH on a rolling window and forecasts `horizon` days ahead
        at each step. Expensive but gives true out-of-sample predictions.

        Args:
            df: DataFrame with 'Close' column
            window: Rolling window size (bars)
            horizon: Forecast horizon (bars)

        Returns:
            Series of forecasted volatilities (annualized)
        """
        returns = 100 * np.log(df["Close"] / df["Close"].shift(1)).dropna()
        forecasts = []
        dates = []

        for i in range(window, len(returns) - horizon):
            train = returns.iloc[i - window:i]
            try:
                vol = GARCH(p=self.p, q=self.q)
                model = ConstantMean(train, volatility=vol, distribution=Normal())
                res = model.fit(disp="off")
                fcast = model.forecast(horizon=horizon, reindex=False)
                fcast_var = fcast.variance.iloc[-1, 0]
                fcast_vol = np.sqrt(fcast_var * 252)
                forecasts.append(fcast_vol)
                dates.append(returns.index[i])
            except Exception:
                forecasts.append(np.nan)
                dates.append(returns.index[i])

        return pd.Series(forecasts, index=dates, name="vol_forecast")

    def position_sizing_signal(self, forecast: VolatilityForecast) -> dict:
        """Generate position sizing recommendation based on volatility.

        Returns:
            dict with 'scale_factor', 'regime', 'action', 'reason'
        """
        # Kelly-style: reduce size when vol > target, increase when vol < target
        scale = self.vol_target / max(forecast.forecast_vol, 0.01)
        scale = min(2.0, max(0.1, scale))  # Clamp between 0.1x and 2.0x

        if forecast.regime == "extreme":
            action = "REDUCE"
            reason = f"Extreme volatility ({forecast.forecast_vol:.0%}), reduce exposure"
        elif forecast.regime == "high":
            action = "CAUTIOUS"
            reason = f"High volatility ({forecast.forecast_vol:.0%}), tighten stops"
        elif forecast.regime == "low":
            action = "OPPORTUNITY"
            reason = f"Low volatility ({forecast.forecast_vol:.0%}), potential for expansion"
        else:
            action = "NORMAL"
            reason = f"Normal volatility ({forecast.forecast_vol:.0%})"

        return {
            "scale_factor": round(scale, 2),
            "regime": forecast.regime,
            "action": action,
            "reason": reason,
        }

    def summary(self, result: VolatilityResult) -> str:
        """Human-readable summary."""
        f = result.forecast
        pos = self.position_sizing_signal(f)

        return f"""
{'='*50}
Volatility Analysis: {result.ticker}
{'='*50}
Model: {self.model_type.upper}({self.p},{self.q})
Current Volatility (realized): {f.current_vol:.1%}
Forecast Volatility (5-day): {f.forecast_vol:.1%}
95% CI: [{f.confidence_interval[0]:.1%}, {f.confidence_interval[1]:.1%}]
Regime: {f.regime.upper()}
Risk Score: {f.risk_score:.0f}/100

Model Parameters:
  Persistence: {result.persistence:.4f}
  Half-life: {result.half_life:.1f} days

Position Sizing:
  Scale Factor: {pos['scale_factor']}x
  Action: {pos['action']}
  {pos['reason']}
{'='*50}
"""

    def _detect_regime(self, vol: float) -> str:
        """Classify volatility regime."""
        if vol < self.REGIME_THRESHOLDS["low"]:
            return "low"
        elif vol < self.REGIME_THRESHOLDS["normal"]:
            return "normal"
        elif vol < self.REGIME_THRESHOLDS["high"]:
            return "high"
        else:
            return "extreme"

    def _calc_persistence(self, params: dict) -> float:
        """Calculate GARCH persistence (alpha[1] + beta[1])."""
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)
        return alpha + beta

    def _calc_half_life(self, persistence: float) -> float:
        """Calculate volatility half-life in days."""
        if persistence >= 1:
            return float("inf")
        if persistence <= 0:
            return 0
        return -np.log(2) / np.log(persistence)
