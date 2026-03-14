"""Model evaluation — accuracy, precision, recall, F1, backtest validation."""

import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from ml.features import FeatureEngineer
from ml.predictor import TrendPredictor


@dataclass
class EvalMetrics:
    """Comprehensive evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray
    total_predictions: int
    correct_predictions: int

    # Backtest-style metrics
    strategy_return: float = 0.0
    buy_hold_return: float = 0.0
    excess_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0


class ModelEvaluator:
    """Evaluate trained prediction models with ML metrics + backtest validation."""

    def __init__(self, predictor: TrendPredictor):
        self.predictor = predictor

    def evaluate(self, df: pd.DataFrame) -> EvalMetrics:
        """Full evaluation on a DataFrame.

        Uses the predictor's test split logic to generate predictions
        and compare against actual outcomes.
        """
        featured_df = self.predictor.feature_engineer.build_features(df)
        X, y = self.predictor.feature_engineer.get_X_y(featured_df)

        if not self.predictor._trained:
            raise RuntimeError("Model not trained")

        # Get test split (same logic as training)
        split_idx = int(len(X) * (1 - self.predictor.test_size))
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        if len(X_test) == 0:
            raise ValueError("No test data available")

        X_scaled = self.predictor.scaler.transform(X_test)
        y_pred = self.predictor.model.predict(X_scaled)
        y_prob = self.predictor.model.predict_proba(X_scaled)[:, 1]

        # Classification metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.0
        cm = confusion_matrix(y_test, y_pred)

        # Backtest-style: simulate following ML signals
        test_dates = featured_df.index[split_idx:]
        strategy_return, buy_hold_return, wins, losses, win_pnl, loss_pnl = (
            self._simulate_strategy(df, test_dates, y_pred)
        )

        total = len(y_pred)
        correct = int((y_pred == y_test.values).sum())

        profit_factor = (
            win_pnl / abs(loss_pnl) if abs(loss_pnl) > 1e-10 else float("inf")
        )
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        return EvalMetrics(
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1=f1,
            roc_auc=auc,
            confusion=cm,
            total_predictions=total,
            correct_predictions=correct,
            strategy_return=strategy_return,
            buy_hold_return=buy_hold_return,
            excess_return=strategy_return - buy_hold_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )

    def _simulate_strategy(
        self,
        df: pd.DataFrame,
        test_dates: pd.DatetimeIndex,
        predictions: np.ndarray,
    ) -> tuple[float, float, int, int, float, float]:
        """Simulate trading based on ML predictions.

        Simple: go long when predict up, go to cash when predict down.
        Returns (strategy_return, buy_hold_return, wins, losses, win_pnl, loss_pnl).
        """
        # Get close prices for test period
        close_prices = df.loc[test_dates, "Close"].values

        if len(close_prices) < 2:
            return 0.0, 0.0, 0, 0, 0.0, 0.0

        # Buy-and-hold return
        buy_hold = (close_prices[-1] / close_prices[0]) - 1

        # Strategy return (long when predict up, flat when predict down)
        daily_returns = np.diff(close_prices) / close_prices[:-1]
        pred_for_returns = predictions[:-1]  # align with daily_returns

        strategy_returns = daily_returns * pred_for_returns
        total_strategy = np.prod(1 + strategy_returns) - 1

        # Win/loss tracking
        trade_returns = strategy_returns[strategy_returns != 0]
        wins = int((trade_returns > 0).sum())
        losses = int((trade_returns < 0).sum())
        win_pnl = float(trade_returns[trade_returns > 0].sum())
        loss_pnl = float(trade_returns[trade_returns < 0].sum())

        return float(total_strategy), float(buy_hold), wins, losses, win_pnl, loss_pnl

    def classification_report(self, df: pd.DataFrame) -> str:
        """Generate sklearn classification report."""
        featured_df = self.predictor.feature_engineer.build_features(df)
        X, y = self.predictor.feature_engineer.get_X_y(featured_df)

        split_idx = int(len(X) * (1 - self.predictor.test_size))
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

        X_scaled = self.predictor.scaler.transform(X_test)
        y_pred = self.predictor.model.predict(X_scaled)

        return classification_report(
            y_test, y_pred,
            target_names=["Down", "Up"],
            digits=4,
        )

    def format_metrics(self, metrics: EvalMetrics) -> str:
        """Format metrics as readable text."""
        cm = metrics.confusion
        lines = [
            "Model Evaluation Results",
            "=" * 55,
            "",
            "Classification Metrics:",
            f"  Accuracy:   {metrics.accuracy:.2%}  ({metrics.correct_predictions}/{metrics.total_predictions})",
            f"  Precision:  {metrics.precision:.2%}",
            f"  Recall:     {metrics.recall:.2%}",
            f"  F1 Score:   {metrics.f1:.2%}",
            f"  ROC AUC:    {metrics.roc_auc:.4f}",
            "",
            "Confusion Matrix:",
            f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}",
            f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}",
            "",
            "Backtest Validation (ML signal → trade):",
            f"  Strategy return:  {metrics.strategy_return:+.2%}",
            f"  Buy & Hold:       {metrics.buy_hold_return:+.2%}",
            f"  Excess return:    {metrics.excess_return:+.2%}",
            f"  Win rate:         {metrics.win_rate:.2%}",
            f"  Profit factor:    {metrics.profit_factor:.2f}",
        ]
        return "\n".join(lines)
