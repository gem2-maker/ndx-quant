import sys
sys.path.insert(0, ".")
from data.fetcher import DataFetcher
from ml.predictor import TrendPredictor
import pickle

fetcher = DataFetcher()
df = fetcher.fetch("QQQ", period="2y", interval="1d")
print(f"Data: {len(df)} rows")

predictor = TrendPredictor(model_type="random_forest", n_estimators=50)
result = predictor.train(df)
print(f"Train acc: {result.train_accuracy:.2%}, Test acc: {result.test_accuracy:.2%}")

model_path = "models/QQQ_random_forest.pkl"
import os
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, "wb") as f:
    pickle.dump({"model": predictor.model, "scaler": predictor.scaler, "feature_names": predictor.feature_names}, f)
print(f"Model saved to {model_path}")
