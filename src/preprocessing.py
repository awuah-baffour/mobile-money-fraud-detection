"""Feature engineering and preprocessing utilities for the PaySim fraud project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_COLUMN = "isFraud"
EPSILON = 1.0

RAW_IDENTIFIER_COLUMNS = ["nameOrig", "nameDest"]
RULE_BASED_COLUMNS = ["isFlaggedFraud"]
POST_TRANSACTION_COLUMNS = ["newbalanceOrig", "newbalanceDest"]

PRIMARY_NUMERIC_FEATURES = [
    "step",
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",
    "origin_balance_zero",
    "destination_balance_zero",
    "origin_balance_sufficient",
    "step_day",
]

PRIMARY_CATEGORICAL_FEATURES = ["type"]

POST_TRANSACTION_NUMERIC_FEATURES = PRIMARY_NUMERIC_FEATURES + [
    "newbalanceOrig",
    "newbalanceDest",
    "origin_balance_change",
    "destination_balance_change",
]


@dataclass(frozen=True)
class FeatureSet:
    """Container for feature-set metadata."""

    name: str
    numeric_features: list[str]
    categorical_features: list[str]

    @property
    def raw_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features


PRIMARY_FEATURE_SET = FeatureSet(
    name="primary_transaction_time",
    numeric_features=PRIMARY_NUMERIC_FEATURES,
    categorical_features=PRIMARY_CATEGORICAL_FEATURES,
)

POST_TRANSACTION_FEATURE_SET = FeatureSet(
    name="post_transaction_comparison",
    numeric_features=POST_TRANSACTION_NUMERIC_FEATURES,
    categorical_features=PRIMARY_CATEGORICAL_FEATURES,
)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the PaySim subset from CSV."""
    return pd.read_csv(path)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create explainable transaction-time and post-transaction analysis features."""
    engineered = df.copy()

    engineered["log_amount"] = np.log1p(engineered["amount"])
    engineered["amount_to_origin_balance_ratio"] = (
        engineered["amount"] / (engineered["oldbalanceOrg"] + EPSILON)
    )
    engineered["amount_to_destination_balance_ratio"] = (
        engineered["amount"] / (engineered["oldbalanceDest"] + EPSILON)
    )
    engineered["origin_balance_zero"] = (engineered["oldbalanceOrg"] == 0).astype(int)
    engineered["destination_balance_zero"] = (
        engineered["oldbalanceDest"] == 0
    ).astype(int)
    engineered["origin_balance_sufficient"] = (
        engineered["oldbalanceOrg"] >= engineered["amount"]
    ).astype(int)

    # PaySim documents one step as one hour, so this is a broad simulation-day index.
    engineered["step_day"] = ((engineered["step"] - 1) // 24 + 1).astype(int)

    # These use post-transaction balances and are excluded from the primary model.
    engineered["origin_balance_change"] = (
        engineered["oldbalanceOrg"] - engineered["newbalanceOrig"]
    )
    engineered["destination_balance_change"] = (
        engineered["newbalanceDest"] - engineered["oldbalanceDest"]
    )

    return engineered


def split_features_target(
    df: pd.DataFrame,
    feature_set: FeatureSet = PRIMARY_FEATURE_SET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X and y for a documented feature set."""
    missing = sorted(set(feature_set.raw_columns + [TARGET_COLUMN]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[feature_set.raw_columns].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
):
    """Create a stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a reusable preprocessing transformer.

    Scaling is useful for Logistic Regression. Tree-based models can use the same
    encoded design matrix, although scaling is not required for their splits.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("cat", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def validate_primary_features(X: pd.DataFrame) -> dict[str, object]:
    """Check that the primary X matrix does not contain known leakage columns."""
    forbidden = set(
        [TARGET_COLUMN]
        + RAW_IDENTIFIER_COLUMNS
        + RULE_BASED_COLUMNS
        + POST_TRANSACTION_COLUMNS
        + ["origin_balance_change", "destination_balance_change"]
    )
    present_forbidden = sorted(forbidden.intersection(X.columns))

    numeric_X = X.select_dtypes(include=[np.number])
    has_missing = bool(X.isna().any().any())
    has_infinite = bool(np.isinf(numeric_X.to_numpy()).any()) if not numeric_X.empty else False

    return {
        "row_count": int(X.shape[0]),
        "column_count_before_encoding": int(X.shape[1]),
        "missing_values": int(X.isna().sum().sum()),
        "has_missing": has_missing,
        "has_infinite_numeric_values": has_infinite,
        "forbidden_columns_present": present_forbidden,
        "categorical_type_values": sorted(X["type"].unique().tolist())
        if "type" in X.columns
        else [],
    }


def fit_preprocessor_for_sanity_check(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_set: FeatureSet = PRIMARY_FEATURE_SET,
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray, list[str]]:
    """Fit preprocessing on training data only and transform train/test features."""
    preprocessor = build_preprocessor(
        feature_set.numeric_features,
        feature_set.categorical_features,
        scale_numeric=True,
    )
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return preprocessor, X_train_encoded, X_test_encoded, feature_names
