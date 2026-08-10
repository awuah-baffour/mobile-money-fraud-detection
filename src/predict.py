"""Use the saved final fraud-detection pipeline for a single transaction."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from preprocessing import add_engineered_features


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "models" / "final_pipeline.pkl"
METADATA_PATH = ROOT / "models" / "final_model_metadata.pkl"


def predict_transaction(transaction: dict[str, object]) -> tuple[float, int, float]:
    """Return fraud probability, binary prediction, and threshold."""
    if not PIPELINE_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Final model artifacts were not found. Run `python src/tune.py` first."
        )

    pipeline = joblib.load(PIPELINE_PATH)
    metadata = joblib.load(METADATA_PATH)
    threshold = float(metadata["threshold"])

    df = pd.DataFrame([transaction])
    df = add_engineered_features(df)
    probability = float(pipeline.predict_proba(df[metadata["features"]])[:, 1][0])
    prediction = int(probability >= threshold)
    return probability, prediction, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fraud probability for one transaction.")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--type", required=True, choices=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    parser.add_argument("--amount", type=float, required=True)
    parser.add_argument("--oldbalanceOrg", type=float, required=True)
    parser.add_argument("--newbalanceOrig", type=float, default=0.0)
    parser.add_argument("--oldbalanceDest", type=float, required=True)
    parser.add_argument("--newbalanceDest", type=float, default=0.0)
    args = parser.parse_args()

    probability, prediction, threshold = predict_transaction(vars(args))
    label = "FRAUD" if prediction == 1 else "LEGITIMATE"
    print(f"Fraud Probability: {probability:.4f}")
    print(f"Prediction: {label}")
    print(f"Threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
