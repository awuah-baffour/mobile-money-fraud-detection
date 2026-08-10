"""Train and evaluate Stage 4 baseline fraud-detection models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from evaluate import (
    THRESHOLD,
    classification_report_frame,
    evaluate_predictions,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from preprocessing import (
    PRIMARY_FEATURE_SET,
    RANDOM_STATE,
    add_engineered_features,
    build_preprocessor,
    load_dataset,
    make_train_test_split,
    split_features_target,
    validate_primary_features,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "paysim_100k.csv"
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def make_model_pipeline(classifier) -> Pipeline:
    """Create a full preprocessing + classifier pipeline."""
    preprocessor = build_preprocessor(
        PRIMARY_FEATURE_SET.numeric_features,
        PRIMARY_FEATURE_SET.categorical_features,
        scale_numeric=True,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def model_definitions(scale_pos_weight: float) -> dict[str, Pipeline]:
    """Return the baseline model pipelines."""
    return {
        "Logistic Regression": make_model_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            )
        ),
        "Random Forest": make_model_pipeline(
            RandomForestClassifier(
                n_estimators=150,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "XGBoost": make_model_pipeline(
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.10,
                subsample=0.90,
                colsample_bytree=0.90,
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def safe_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def confusion_matrix_values(matrix: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = matrix.ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a simple GitHub-style markdown table without optional dependencies."""
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(format(value, floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_markdown(
    comparison: pd.DataFrame,
    confusion_rows: list[dict[str, object]],
    feature_importance_notes: dict[str, list[str]],
    scale_pos_weight: float,
    strongest_candidate: str,
    suspicious_notes: list[str],
) -> None:
    """Write Stage 4 results in report-ready Markdown."""
    table_md = dataframe_to_markdown(comparison, floatfmt=".4f")
    confusion_md = dataframe_to_markdown(pd.DataFrame(confusion_rows), floatfmt=".4f")

    notes_md = "\n".join(f"- {note}" for note in suspicious_notes)
    feature_md = "\n\n".join(
        [
            f"### {model_name}\n"
            + "\n".join(f"- `{feature}`" for feature in features[:8])
            for model_name, features in feature_importance_notes.items()
        ]
    )

    content = f"""# Stage 4: Baseline Model Training and Evaluation

## Experimental Setup

The baseline models were trained on the reduced 100,000-record PaySim experimental dataset using the Stage 3 primary transaction-time feature set. The train/test split used `test_size=0.20`, `stratify=y`, and `random_state=42`, producing 80,000 training rows and 20,000 test rows. The preprocessing pipeline was fitted only on the training data.

The evaluation threshold for this baseline stage is `{THRESHOLD:.2f}`. Predicted probabilities were saved for each model so that threshold analysis can be performed later.

## Class Imbalance Strategy

Logistic Regression and Random Forest use `class_weight="balanced"`. XGBoost uses `scale_pos_weight={scale_pos_weight:.4f}`, calculated from the training set as legitimate training samples divided by fraudulent training samples.

Accuracy is reported, but it is not the main metric because the dataset is imbalanced. Precision, recall, F1-score, PR-AUC, and ROC-AUC are more informative for fraud detection.

## Baseline Model Comparison

{table_md}

## Confusion Matrices

In this project, false positives are legitimate transactions incorrectly flagged as fraud, which could inconvenience customers or delay valid transactions. False negatives are fraudulent transactions incorrectly treated as legitimate, which may lead to financial loss.

{confusion_md}

## ROC-AUC Comparison

The combined ROC curve is saved as `figures/roc_curve_comparison.png`. ROC-AUC measures ranking ability across thresholds, but a high ROC-AUC alone does not automatically identify the best fraud-detection model.

## Precision-Recall Comparison

The combined Precision-Recall curve is saved as `figures/precision_recall_curve_comparison.png`. PR-AUC is especially relevant because fraud is the minority class, so the precision-recall trade-off directly shows how well a model identifies fraud without overwhelming the system with false alarms.

## Preliminary Feature Importance

Feature importance plots were created for the three baseline models. These values indicate which features the models rely on more heavily; they do not prove causation.

{feature_md}

## Suspicious Performance and Leakage Review

{notes_md}

## Baseline Conclusion

The current strongest baseline candidate is **{strongest_candidate}** based on the Stage 4 balance of recall, precision, F1-score, PR-AUC, and ROC-AUC. This is not the final model. Final selection should wait until hyperparameter tuning, threshold analysis, and deeper feature-importance review are completed.

These results apply only to the reduced PaySim experimental dataset and should not be presented as performance on Ghanaian mobile-money systems or any real financial institution.
"""
    (RESULTS_DIR / "stage4_baseline_results.md").write_text(content, encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = add_engineered_features(load_dataset(DATA_PATH))
    X, y = split_features_target(df, PRIMARY_FEATURE_SET)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    validation = validate_primary_features(X)
    if validation["forbidden_columns_present"]:
        raise ValueError(
            "Primary features include forbidden columns: "
            f"{validation['forbidden_columns_present']}"
        )

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    models = model_definitions(scale_pos_weight=scale_pos_weight)
    comparison_rows = []
    confusion_rows = []
    probabilities = {}
    feature_importance_notes = {}
    suspicious_notes = []

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_probability = pipeline.predict_proba(X_test)[:, 1]
        probabilities[model_name] = y_probability

        comparison_rows.append(evaluate_predictions(model_name, y_test, y_probability))

        name = safe_name(model_name)
        pd.DataFrame(
            {
                "y_true": y_test.to_numpy(),
                "fraud_probability": y_probability,
                "prediction_at_0_50": (y_probability >= THRESHOLD).astype(int),
            }
        ).to_csv(RESULTS_DIR / f"stage4_{name}_predictions.csv", index=False)

        report = classification_report_frame(y_test, y_probability)
        report.to_csv(RESULTS_DIR / f"stage4_{name}_classification_report.csv")

        matrix = plot_confusion_matrix(
            y_test,
            y_probability,
            title=f"{model_name} Confusion Matrix",
            output_path=FIGURES_DIR / f"{name}_confusion_matrix.png",
        )
        confusion_rows.append({"Model": model_name, **confusion_matrix_values(matrix)})

        joblib.dump(pipeline, MODELS_DIR / f"{name}_baseline.pkl")

        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        classifier = pipeline.named_steps["classifier"]
        if model_name == "Logistic Regression":
            importance = plot_feature_importance(
                feature_names,
                classifier.coef_[0],
                "Logistic Regression Coefficients",
                FIGURES_DIR / "logistic_regression_coefficients.png",
                absolute=True,
            )
        else:
            importance = plot_feature_importance(
                feature_names,
                classifier.feature_importances_,
                f"{model_name} Feature Importance",
                FIGURES_DIR / f"{name}_feature_importance.png",
                absolute=True,
            )
        importance.to_csv(RESULTS_DIR / f"stage4_{name}_feature_importance.csv", index=False)
        feature_importance_notes[model_name] = importance["feature"].head(8).tolist()

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison[
        ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ]
    comparison.to_csv(RESULTS_DIR / "model_comparison_baseline.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(
        RESULTS_DIR / "stage4_confusion_matrices.csv", index=False
    )

    plot_roc_curves(
        y_test,
        probabilities,
        output_path=FIGURES_DIR / "roc_curve_comparison.png",
    )
    plot_precision_recall_curves(
        y_test,
        probabilities,
        output_path=FIGURES_DIR / "precision_recall_curve_comparison.png",
    )

    joblib.dump(
        {
            "feature_set": PRIMARY_FEATURE_SET,
            "threshold": THRESHOLD,
            "scale_pos_weight": scale_pos_weight,
            "validation": validation,
        },
        MODELS_DIR / "stage4_training_metadata.pkl",
    )

    for row in comparison.to_dict(orient="records"):
        if row["ROC-AUC"] >= 0.995 or row["PR-AUC"] >= 0.995 or row["Recall"] >= 0.995:
            suspicious_notes.append(
                f"{row['Model']} produced an unusually high metric "
                f"(Recall={row['Recall']:.4f}, ROC-AUC={row['ROC-AUC']:.4f}, "
                f"PR-AUC={row['PR-AUC']:.4f}). This should be discussed carefully."
            )

    if not suspicious_notes:
        suspicious_notes.append(
            "No forbidden primary features were present and no model crossed the configured extreme-performance review thresholds."
        )

    suspicious_notes.append(
        "The primary feature set excludes raw identifiers, isFlaggedFraud, and post-transaction balance variables."
    )
    suspicious_notes.append(
        "PaySim balance variables still require careful academic discussion because the dataset documentation warns that fraudulent transactions are cancelled."
    )

    ranked = comparison.sort_values(
        by=["PR-AUC", "F1", "Recall", "Precision"], ascending=False
    )
    strongest_candidate = str(ranked.iloc[0]["Model"])

    write_results_markdown(
        comparison=comparison,
        confusion_rows=confusion_rows,
        feature_importance_notes=feature_importance_notes,
        scale_pos_weight=scale_pos_weight,
        strongest_candidate=strongest_candidate,
        suspicious_notes=suspicious_notes,
    )

    summary = {
        "comparison": comparison.round(6).to_dict(orient="records"),
        "confusion_matrices": confusion_rows,
        "scale_pos_weight": scale_pos_weight,
        "strongest_baseline_candidate": strongest_candidate,
        "suspicious_notes": suspicious_notes,
        "saved_models": [
            str(path.relative_to(ROOT)) for path in sorted(MODELS_DIR.glob("*baseline.pkl"))
        ],
    }
    (RESULTS_DIR / "stage4_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
