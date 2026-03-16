"""Portfolio Optimizer — lightweight mean-variance optimization inspired by Riskfolio-Lib.

Provides modern portfolio theory optimization without heavy dependencies.
Uses scipy.optimize instead of cvxpy, sklearn for shrinkage estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf, OAS


class PortfolioOptimizer:
    """
    Portfolio optimization using Mean-Variance and Risk Parity approaches.

    Unlike Riskfolio-Lib (which requires CVXPY), this uses scipy.optimize
    for lightweight deployment on low-spec machines.

    Parameters:
        returns: DataFrame of asset returns (rows=dates, cols=assets)
        risk_free_rate: Annual risk-free rate (default 0.05 = 5%)
        cov_method: Covariance estimation method ('ledoit', 'oas', 'hist')
        mean_method: Mean return estimation ('hist', 'ewma')
        freq: Trading frequency per year (default 252)
    """

    def __init__(
        self,
        returns: pd.DataFrame | None = None,
        risk_free_rate: float = 0.05,
        cov_method: str = "ledoit",
        mean_method: str = "hist",
        freq: int = 252,
    ):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.cov_method = cov_method
        self.mean_method = mean_method
        self.freq = freq

        self._mu: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._asset_names: list[str] = []
        self._n_assets: int = 0

        if returns is not None:
            self._fit(returns)

    def _fit(self, returns: pd.DataFrame) -> None:
        """Estimate mean vector and covariance matrix."""
        self._asset_names = list(returns.columns)
        self._n_assets = len(self._asset_names)
        daily_rf = self.risk_free_rate / self.freq

        # Mean estimation
        if self.mean_method == "ewma":
            self._mu = returns.ewm(span=60).mean().iloc[-1].values - daily_rf
        else:  # hist
            self._mu = returns.mean().values - daily_rf

        # Covariance estimation with shrinkage
        if self.cov_method == "ledoit":
            lw = LedoitWolf().fit(returns.values)
            self._cov = lw.covariance_
        elif self.cov_method == "oas":
            oas = OAS().fit(returns.values)
            self._cov = oas.covariance_
        else:  # hist
            self._cov = returns.cov().values

        # Annualize
        self._mu_annual = self._mu * self.freq
        self._cov_annual = self._cov * self.freq

    def optimize_sharpe(
        self,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
        long_only: bool = True,
    ) -> pd.Series:
        """Maximize Sharpe ratio (risk-adjusted return).

        Args:
            max_weight: Maximum weight per asset (default 40%)
            min_weight: Minimum weight per asset (default 0%)
            long_only: No short selling (default True)

        Returns:
            Series of optimal weights
        """
        n = self._n_assets

        def neg_sharpe(w):
            ret = w @ self._mu_annual
            vol = np.sqrt(w @ self._cov_annual @ w)
            return -(ret / vol) if vol > 1e-8 else 0

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(min_weight, max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return pd.Series(result.x, index=self._asset_names, name="weight")

    def optimize_min_risk(
        self,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
    ) -> pd.Series:
        """Minimize portfolio volatility.

        Returns:
            Series of optimal weights
        """
        n = self._n_assets

        def portfolio_vol(w):
            return np.sqrt(w @ self._cov_annual @ w)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(min_weight, max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            portfolio_vol, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return pd.Series(result.x, index=self._asset_names, name="weight")

    def optimize_risk_parity(self) -> pd.Series:
        """Risk Parity: equalize risk contribution from each asset.

        Each asset contributes equally to total portfolio risk.
        Classic Risk Parity / Equal Risk Contribution.

        Returns:
            Series of optimal weights
        """
        n = self._n_assets

        def risk_parity_objective(w):
            port_vol = np.sqrt(w @ self._cov_annual @ w)
            marginal_contrib = self._cov_annual @ w
            risk_contrib = w * marginal_contrib
            target = port_vol / n
            return np.sum((risk_contrib - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.01, 1.0)] * n  # Need min 1% to avoid zero risk contribution
        w0 = np.ones(n) / n

        result = minimize(
            risk_parity_objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return pd.Series(result.x, index=self._asset_names, name="weight")

    def efficient_frontier(
        self,
        n_points: int = 50,
        max_weight: float = 0.40,
    ) -> pd.DataFrame:
        """Generate efficient frontier (volatility vs return for optimal portfolios).

        Returns:
            DataFrame with columns: return, volatility, sharpe, and weight columns
        """
        min_ret = self._mu_annual.min()
        max_ret = self._mu_annual.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier = []
        for target_ret in target_returns:
            try:
                w = self._min_risk_for_return(target_ret, max_weight)
                port_ret = w @ self._mu_annual
                port_vol = np.sqrt(w @ self._cov_annual @ w)
                sharpe = port_ret / port_vol if port_vol > 1e-8 else 0
                row = {"return": port_ret, "volatility": port_vol, "sharpe": sharpe}
                for i, name in enumerate(self._asset_names):
                    row[name] = w[i]
                frontier.append(row)
            except Exception:
                continue

        return pd.DataFrame(frontier)

    def _min_risk_for_return(self, target_ret: float, max_weight: float) -> np.ndarray:
        """Find minimum risk portfolio for given target return."""
        n = self._n_assets

        def portfolio_vol(w):
            return np.sqrt(w @ self._cov_annual @ w)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: w @ self._mu_annual - target_ret},
        ]
        bounds = [(0, max_weight)] * n
        w0 = np.ones(n) / n

        result = minimize(
            portfolio_vol, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return result.x

    def portfolio_stats(self, weights: pd.Series | np.ndarray) -> dict:
        """Calculate portfolio statistics for given weights.

        Returns:
            dict with annual_return, volatility, sharpe, max_drawdown_estimate
        """
        if isinstance(weights, pd.Series):
            w = weights.values
        else:
            w = np.array(weights)

        port_ret = w @ self._mu_annual
        port_vol = np.sqrt(w @ self._cov_annual @ w)
        sharpe = port_ret / port_vol if port_vol > 1e-8 else 0

        return {
            "annual_return": port_ret,
            "volatility": port_vol,
            "sharpe": sharpe,
            "weights": dict(zip(self._asset_names, w)),
        }

    def compare_methods(self, max_weight: float = 0.40) -> pd.DataFrame:
        """Compare Sharpe, MinRisk, and Risk Parity allocations.

        Returns:
            DataFrame comparing different optimization methods
        """
        results = {}

        w_sharpe = self.optimize_sharpe(max_weight=max_weight)
        stats_sharpe = self.portfolio_stats(w_sharpe)
        results["Max Sharpe"] = {
            **stats_sharpe,
            "weights": w_sharpe.to_dict(),
        }

        w_minrisk = self.optimize_min_risk(max_weight=max_weight)
        stats_minrisk = self.portfolio_stats(w_minrisk)
        results["Min Risk"] = {
            **stats_minrisk,
            "weights": w_minrisk.to_dict(),
        }

        w_rp = self.optimize_risk_parity()
        stats_rp = self.portfolio_stats(w_rp)
        results["Risk Parity"] = {
            **stats_rp,
            "weights": w_rp.to_dict(),
        }

        # Equal weight benchmark
        w_eq = pd.Series(1.0 / self._n_assets, index=self._asset_names)
        stats_eq = self.portfolio_stats(w_eq)
        results["Equal Weight"] = {
            **stats_eq,
            "weights": w_eq.to_dict(),
        }

        comparison = pd.DataFrame({
            name: {
                "Return": data["annual_return"],
                "Volatility": data["volatility"],
                "Sharpe": data["sharpe"],
            }
            for name, data in results.items()
        }).T

        return comparison


def walk_forward_optimize(
    returns: pd.DataFrame,
    window_days: int = 252,
    rebalance_days: int = 63,
    method: str = "sharpe",
    max_weight: float = 0.40,
    cov_method: str = "ledoit",
) -> pd.DataFrame:
    """
    Walk-forward portfolio optimization.

    Trains on a rolling window, applies weights, rebalances periodically.
    Simulates realistic out-of-sample portfolio management.

    Args:
        returns: Full return series
        window_days: Training window (default 252 = 1 year)
        rebalance_days: Rebalance every N days (default 63 = quarterly)
        method: 'sharpe', 'minrisk', or 'risk_parity'
        max_weight: Max weight per asset
        cov_method: Covariance estimation method

    Returns:
        DataFrame with daily portfolio returns and weights over time
    """
    dates = returns.index
    assets = returns.columns
    n = len(assets)

    results = []
    current_weights = np.ones(n) / n

    for i in range(window_days, len(dates), rebalance_days):
        # Training window
        train = returns.iloc[i - window_days:i]

        try:
            opt = PortfolioOptimizer(train, cov_method=cov_method)

            if method == "sharpe":
                w = opt.optimize_sharpe(max_weight=max_weight)
            elif method == "minrisk":
                w = opt.optimize_min_risk(max_weight=max_weight)
            elif method == "risk_parity":
                w = opt.optimize_risk_parity()
            else:
                w = pd.Series(1.0 / n, index=assets)

            current_weights = w.values
        except Exception:
            pass  # Keep previous weights

        # Apply weights for the next rebalance period
        end_idx = min(i + rebalance_days, len(dates))
        period_returns = returns.iloc[i:end_idx]

        for date, row in period_returns.iterrows():
            port_ret = (current_weights * row.values).sum()
            results.append({
                "date": date,
                "portfolio_return": port_ret,
                **{f"w_{a}": current_weights[j] for j, a in enumerate(assets)},
            })

    return pd.DataFrame(results).set_index("date")
