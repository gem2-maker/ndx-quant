"""ML-based trend prediction models.

Supports XGBoost and Random Forest for predicting price direction.
Includes model persistence (save/load) and prediction confidence.
"""

import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from ml.features import FeatureEngineer


MODEL_DIR = Path("models")


@dataclass
class PredictionResult:
    """Result from a prediction."""
    date: pd.Timestamp
    prediction: int           # 1 = up, 0 = down
    probability: float        # P(up)
    confidence: str           # "high", "medium", "low"
    features_used: int

    @property
    def signal(self) -> str:
        return "BUY" if self.prediction == 1 else "SELL"


@dataclass
class TrainingResult:
    """Result from model training."""
    model_name: str
    train_accuracy: float
    test_accuracy: float
    feature_importances: dict[str, float] = field(default_factory=dict)
    cross_val_scores: list[float] = field(default_factory=list)
    n_features: int = 0
    n_train_samples: int = 0
    n_test_samples: int = 0


class TrendPredictor:
    """Predict price trend direction using ensemble ML models."""

    SUPPORTED_MODELS = ["random_forest", "gradient_boosting"]
    if HAS_XGB:
        SUPPORTED_MODELS.append("xgboost")

    def __init__(
        self,
        model_type: str = "xgboost" if HAS_XGB else "random_forest",
        lookback: int = 20,
        forecast_horizon: int = 5,
        test_size: float = 0.2,
        n_estimators: int = 200,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        if model_type == "xgboost" and not HAS_XGB:
            print("Warning: xgboost not installed, falling back to random_forest")
            model_type = "random_forest"

        self.model_type = model_type
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.test_size = test_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.feature_engineer = FeatureEngineer(
            lookback=lookback,
            forecast_horizon=forecast_horizon,
            target_type="direction",
        )
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names: list[str] = []
        self._trained = False

    def _create_model(self):
        """Instantiate the ML model."""
        if self.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric="logloss",
                verbosity=0,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def build_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build the full featured DataFrame once for reuse."""
        return self.feature_engineer.build_features(df)

    def prepare_feature_matrix(
        self,
        featured_df: pd.DataFrame,
        feature_names: Optional[list[str]] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Extract X/y and align columns to a trained model when needed."""
        X, y = self.feature_engineer.get_X_y(featured_df)

        if feature_names:
            X = X.reindex(columns=feature_names, fill_value=0.0)

        return X, y

    def train_from_featured_df(self, featured_df: pd.DataFrame) -> TrainingResult:
        """Train the prediction model from a precomputed feature table."""
        X, y = self.prepare_feature_matrix(featured_df)
        return self.train_from_matrix(X, y)

    def train_from_matrix(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train the prediction model from an aligned feature matrix."""

        if len(X) < 50:
            raise ValueError(f"Not enough data: {len(X)} samples (need >= 50)")

        self.feature_names = list(X.columns)

        # Time-series aware split (no future leakage)
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self._trained = True

        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)

        # Feature importances
        importances = {}
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            for name, val in sorted(zip(self.feature_names, imp), key=lambda x: -x[1])[:20]:
                importances[name] = float(val)

        # Cross-validation (time-series)
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_cv_train = self.scaler.fit_transform(X.iloc[train_idx])
            X_cv_val = self.scaler.transform(X.iloc[val_idx])
            cv_model = self._create_model()
            cv_model.fit(X_cv_train, y.iloc[train_idx])
            cv_scores.append(cv_model.score(X_cv_val, y.iloc[val_idx]))

        # Refit scaler on full training set
        self.scaler.fit(X_train)

        return TrainingResult(
            model_name=self.model_type,
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            feature_importances=importances,
            cross_val_scores=cv_scores,
            n_features=len(self.feature_names),
            n_train_samples=len(X_train),
            n_test_samples=len(X_test),
        )

    def train(self, df: pd.DataFrame) -> TrainingResult:
        """Train the prediction model.

        Args:
            df: OHLCV DataFrame (from DataFetcher).

        Returns:
            TrainingResult with metrics.
        """
        featured_df = self.build_feature_frame(df)
        return self.train_from_featured_df(featured_df)

    def predict(self, df: pd.DataFrame, top_n: int = 5) -> list[PredictionResult]:
        """Predict trend for the most recent data points.

        Args:
            df: OHLCV DataFrame.
            top_n: Number of most recent predictions to return.

        Returns:
            List of PredictionResult for the latest bars.
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")

        featured_df = self.build_feature_frame(df)
        X, _ = self.prepare_feature_matrix(featured_df, self.feature_names)

        if X.empty:
            return []

        # Take most recent rows
        X_recent = X.tail(top_n)
        X_scaled = self.scaler.transform(X_recent)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        results = []
        for i, (date, row) in enumerate(X_recent.iterrows()):
            prob_up = float(probabilities[i][1])
            conf = self._confidence(prob_up)

            results.append(PredictionResult(
                date=date,
                prediction=int(predictions[i]),
                probability=prob_up,
                confidence=conf,
                features_used=len(self.feature_names),
            ))

        return results

    def predict_next(self, df: pd.DataFrame) -> PredictionResult:
        """Predict the NEXT bar (beyond the data)."""
        preds = self.predict(df, top_n=1)
        if not preds:
            raise RuntimeError("No prediction available")
        return preds[0]

    def _confidence(self, prob_up: float) -> str:
        """Classify prediction confidence."""
        dist_from_center = abs(prob_up - 0.5)
        if dist_from_center > 0.25:
            return "high"
        elif dist_from_center > 0.15:
            return "medium"
        else:
            return "low"

    def save(self, name: str = "latest"):
        """Save model to disk."""
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / f"{name}_{self.model_type}.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "lookback": self.lookback,
                "forecast_horizon": self.forecast_horizon,
            }, f)
        print(f"  Model saved to {path}")

    def load(self, name: str = "latest"):
        """Load model from disk."""
        path = MODEL_DIR / f"{name}_{self.model_type}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self._trained = True
        print(f"  Model loaded from {path}")

    def summary(self, result: TrainingResult) -> str:
        """Format training result as readable summary."""
        lines = [
            f"Trend Prediction — {result.model_name}",
            "=" * 50,
            f"  Train accuracy:  {result.train_accuracy:.2%}",
            f"  Test accuracy:   {result.test_accuracy:.2%}",
            f"  CV mean:         {np.mean(result.cross_val_scores):.2%} "
            f"(+/- {np.std(result.cross_val_scores):.2%})",
            f"  Features used:   {result.n_features}",
            f"  Train samples:   {result.n_train_samples}",
            f"  Test samples:    {result.n_test_samples}",
            "",
            "  Top features:",
        ]
        for name, imp in list(result.feature_importances.items())[:10]:
            bar = "+" * int(imp * 100)
            lines.append(f"    {name:30s} {imp:.4f} {bar}")

        return "\n".join(lines)
