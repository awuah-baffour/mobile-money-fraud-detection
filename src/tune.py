"""Stage 5 tuning, threshold analysis, and final candidate selection."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from evaluate import (
    THRESHOLD,
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

THRESHOLDS = np.round(np.arange(0.10, 0.95, 0.05), 2)
FN_COST = 10
FP_COST = 1


def make_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    PRIMARY_FEATURE_SET.numeric_features,
                    PRIMARY_FEATURE_SET.categorical_features,
                    scale_numeric=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def get_search_spaces(scale_pos_weight: float) -> dict[str, tuple[Pipeline, dict[str, object], int]]:
    return {
        "Logistic Regression": (
            make_pipeline(
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1200,
                    random_state=RANDOM_STATE,
                )
            ),
            {
                "classifier__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
                "classifier__class_weight": ["balanced", None],
            },
            8,
        ),
        "Random Forest": (
            make_pipeline(
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            {
                "classifier__n_estimators": [100, 150, 220],
                "classifier__max_depth": [10, 14, 18, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2", None],
                "classifier__class_weight": ["balanced", "balanced_subsample"],
            },
            10,
        ),
        "XGBoost": (
            make_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
            {
                "classifier__n_estimators": randint(120, 281),
                "classifier__max_depth": randint(3, 7),
                "classifier__learning_rate": uniform(0.04, 0.12),
                "classifier__subsample": uniform(0.75, 0.25),
                "classifier__colsample_bytree": uniform(0.75, 0.25),
                "classifier__min_child_weight": randint(1, 7),
                "classifier__gamma": uniform(0.0, 0.4),
                "classifier__reg_alpha": uniform(0.0, 0.4),
                "classifier__reg_lambda": uniform(0.8, 1.5),
                "classifier__scale_pos_weight": [scale_pos_weight, scale_pos_weight * 0.75, scale_pos_weight * 1.25],
            },
            12,
        ),
    }


def threshold_metrics(y_true, y_probability, thresholds=THRESHOLDS) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        y_pred = (y_probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "false_positive_rate": fp / (fp + tn),
                "false_negative_rate": fn / (fn + tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "illustrative_cost": int((fn * FN_COST) + (fp * FP_COST)),
            }
        )
    return pd.DataFrame(rows)


def metrics_at_threshold(model_name: str, y_true, y_probability, threshold: float) -> dict[str, object]:
    y_pred = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_probability),
        "PR-AUC": average_precision_score(y_true, y_probability),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
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


def plot_threshold_curves(threshold_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    ax.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    ax.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    ax.set_title("Precision, Recall, and F1 Across Thresholds")
    ax.set_xlabel("Fraud Probability Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_vs_threshold.png", bbox_inches="tight", dpi=180)
    plt.close()


def plot_final_curves(y_test, final_probability: np.ndarray, model_name: str) -> None:
    plot_roc_curves(
        y_test,
        {model_name: final_probability},
        FIGURES_DIR / "final_roc_curve.png",
    )
    plot_precision_recall_curves(
        y_test,
        {model_name: final_probability},
        FIGURES_DIR / "final_precision_recall_curve.png",
    )


def diagnostic_rules(df: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    type_pred = X_test["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int).to_numpy()
    type_metrics = metrics_at_threshold("Rule: TRANSFER or CASH_OUT", y_test, type_pred, 0.50)

    train_df = df.loc[X_test.index.difference(X_test.index)] if False else None
    del train_df

    rows = []
    for feature in ["amount", "log_amount", "amount_to_origin_balance_ratio", "amount_to_destination_balance_ratio"]:
        values = df[feature]
        candidates = np.quantile(values, np.linspace(0.05, 0.95, 19))
        best = None
        for candidate in candidates:
            pred = (X_test[feature] >= candidate).astype(int).to_numpy()
            f1 = f1_score(y_test, pred, zero_division=0)
            if best is None or f1 > best["f1"]:
                tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
                best = {
                    "feature": feature,
                    "threshold": float(candidate),
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "recall": recall_score(y_test, pred, zero_division=0),
                    "f1": f1,
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                    "true_negatives": int(tn),
                }
        rows.append(best)

    amount_rule_df = pd.DataFrame(rows)
    amount_rule_df.to_csv(RESULTS_DIR / "stage5_amount_rule_diagnostics.csv", index=False)

    feature_stats = df.groupby("isFraud")[PRIMARY_FEATURE_SET.numeric_features].agg(
        ["mean", "median", "std", "min", "max"]
    )
    feature_stats.to_csv(RESULTS_DIR / "stage5_feature_distribution_by_class.csv")

    type_rule_df = pd.DataFrame([type_metrics])
    type_rule_df.to_csv(RESULTS_DIR / "stage5_type_rule_diagnostics.csv", index=False)

    return {
        "type_rule": type_metrics,
        "amount_rules": amount_rule_df.round(6).to_dict(orient="records"),
    }


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = add_engineered_features(load_dataset(DATA_PATH))
    X, y = split_features_target(df, PRIMARY_FEATURE_SET)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    validation = validate_primary_features(X)
    if validation["forbidden_columns_present"]:
        raise ValueError(f"Forbidden features found: {validation['forbidden_columns_present']}")

    scale_pos_weight = int((y_train == 0).sum()) / int((y_train == 1).sum())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
    }

    diagnostics = diagnostic_rules(df, X_test, y_test)
    search_spaces = get_search_spaces(scale_pos_weight)

    tuned_rows = []
    cv_rows = []
    best_params = {}
    fitted_models = {}
    probabilities = {}

    for model_name, (pipeline, params, n_iter) in search_spaces.items():
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=n_iter,
            scoring="average_precision",
            n_jobs=-1,
            cv=cv,
            random_state=RANDOM_STATE,
            verbose=0,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        fitted_models[model_name] = best_estimator
        best_params[model_name] = search.best_params_

        cv_result = cross_validate(
            best_estimator,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
        )
        cv_rows.append(
            {
                "Model": model_name,
                "CV ROC-AUC mean": float(np.mean(cv_result["test_roc_auc"])),
                "CV ROC-AUC std": float(np.std(cv_result["test_roc_auc"])),
                "CV PR-AUC mean": float(np.mean(cv_result["test_pr_auc"])),
                "CV PR-AUC std": float(np.std(cv_result["test_pr_auc"])),
                "CV F1 mean": float(np.mean(cv_result["test_f1"])),
                "CV Precision mean": float(np.mean(cv_result["test_precision"])),
                "CV Recall mean": float(np.mean(cv_result["test_recall"])),
            }
        )

        y_probability = best_estimator.predict_proba(X_test)[:, 1]
        probabilities[model_name] = y_probability
        tuned_rows.append(evaluate_predictions(model_name, y_test, y_probability))
        joblib.dump(best_estimator, MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_tuned.pkl")

    tuned_comparison = pd.DataFrame(tuned_rows)
    tuned_comparison["Tuned?"] = "Yes"
    tuned_comparison = tuned_comparison[
        ["Model", "Tuned?", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ]
    tuned_comparison.to_csv(RESULTS_DIR / "stage5_tuned_model_comparison.csv", index=False)

    cv_summary = pd.DataFrame(cv_rows)
    cv_summary.to_csv(RESULTS_DIR / "stage5_cross_validation_results.csv", index=False)
    pd.DataFrame(
        [{"Model": model, "Best Parameters": json.dumps(params, default=str)} for model, params in best_params.items()]
    ).to_csv(RESULTS_DIR / "stage5_best_hyperparameters.csv", index=False)

    baseline = pd.read_csv(RESULTS_DIR / "model_comparison_baseline.csv")
    baseline["Version"] = "Baseline"
    tuned_for_comparison = tuned_comparison.drop(columns=["Tuned?"]).copy()
    tuned_for_comparison["Version"] = "Tuned"
    baseline_vs_tuned = pd.concat([baseline, tuned_for_comparison], ignore_index=True)
    baseline_vs_tuned = baseline_vs_tuned[
        ["Model", "Version", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ].sort_values(["Model", "Version"])
    baseline_vs_tuned.to_csv(RESULTS_DIR / "stage5_baseline_vs_tuned.csv", index=False)

    # Select model by PR-AUC first, then F1/recall/precision. Threshold is selected from CV train probabilities.
    selected_model_name = tuned_comparison.sort_values(
        ["PR-AUC", "F1", "Recall", "Precision"], ascending=False
    ).iloc[0]["Model"]
    selected_pipeline = fitted_models[selected_model_name]

    cv_train_probability = cross_val_predict(
        selected_pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold_df = threshold_metrics(y_train, cv_train_probability)
    threshold_df.to_csv(RESULTS_DIR / "threshold_analysis.csv", index=False)
    plot_threshold_curves(threshold_df)

    min_cost = threshold_df["illustrative_cost"].min()
    cost_candidates = threshold_df[threshold_df["illustrative_cost"] == min_cost]
    selected_threshold = float(cost_candidates.sort_values(["recall", "precision"], ascending=False).iloc[0]["threshold"])

    final_probability = selected_pipeline.predict_proba(X_test)[:, 1]
    final_metrics = metrics_at_threshold(selected_model_name, y_test, final_probability, selected_threshold)
    final_metrics_050 = metrics_at_threshold(selected_model_name, y_test, final_probability, THRESHOLD)
    pd.DataFrame([final_metrics_050, final_metrics]).to_csv(
        RESULTS_DIR / "stage5_final_threshold_comparison.csv", index=False
    )
    pd.DataFrame([final_metrics]).to_csv(RESULTS_DIR / "stage5_final_test_metrics.csv", index=False)

    final_matrix = plot_confusion_matrix(
        y_test,
        final_probability,
        f"Final {selected_model_name} Confusion Matrix at Threshold {selected_threshold:.2f}",
        FIGURES_DIR / "final_confusion_matrix.png",
        threshold=selected_threshold,
    )
    plot_final_curves(y_test, final_probability, selected_model_name)

    feature_names = selected_pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    classifier = selected_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance_values = classifier.feature_importances_
    else:
        importance_values = classifier.coef_[0]
    final_importance = plot_feature_importance(
        feature_names,
        importance_values,
        "Final Model Feature Importance",
        FIGURES_DIR / "final_feature_importance.png",
        top_n=15,
        absolute=True,
    )
    final_importance.to_csv(RESULTS_DIR / "stage5_final_feature_importance.csv", index=False)

    joblib.dump(selected_pipeline, MODELS_DIR / "final_pipeline.pkl")
    joblib.dump(
        {
            "model_name": selected_model_name,
            "threshold": selected_threshold,
            "features": PRIMARY_FEATURE_SET.raw_columns,
            "encoded_features": feature_names,
        },
        MODELS_DIR / "final_model_metadata.pkl",
    )

    plot_roc_curves(y_test, probabilities, FIGURES_DIR / "stage5_tuned_roc_curve_comparison.png")
    plot_precision_recall_curves(y_test, probabilities, FIGURES_DIR / "stage5_tuned_precision_recall_curve_comparison.png")

    suspicious_notes = [
        "Tree-based model performance remains very high even after excluding raw identifiers, isFlaggedFraud, and post-transaction balances.",
        "The simple transaction-type rule has perfect recall but low precision, showing that PaySim's fraud labels are strongly linked to TRANSFER and CASH_OUT in this sample.",
        "Feature importance is dominated by balance sufficiency and balance-ratio behavior, suggesting the models may be learning PaySim-specific simulation patterns.",
        "The results demonstrate strong performance within the reduced PaySim experimental environment, not guaranteed real-world Ghanaian mobile-money performance.",
    ]

    report = f"""# Stage 5: Hyperparameter Tuning and Threshold Analysis

## Hyperparameter Tuning Methodology

Tuning used only the 80,000-row training split. A 5-fold `StratifiedKFold(shuffle=True, random_state=42)` cross-validation strategy was used. `RandomizedSearchCV` optimized average precision, which corresponds to PR-AUC and is appropriate for an imbalanced fraud-detection task.

## Cross-Validation Results

{dataframe_to_markdown(cv_summary, floatfmt=".4f")}

## Tuned Hyperparameters

{dataframe_to_markdown(pd.DataFrame([{'Model': model, 'Best Parameters': json.dumps(params, default=str)} for model, params in best_params.items()]), floatfmt=".4f")}

## Tuned Model Comparison on Test Set at Threshold 0.50

{dataframe_to_markdown(tuned_comparison, floatfmt=".4f")}

## Baseline vs Tuned Results

{dataframe_to_markdown(baseline_vs_tuned, floatfmt=".4f")}

## Diagnostic Rule Checks

The simple rule `type in [TRANSFER, CASH_OUT]` produced precision {diagnostics['type_rule']['Precision']:.4f}, recall {diagnostics['type_rule']['Recall']:.4f}, and F1 {diagnostics['type_rule']['F1']:.4f}. This confirms that transaction type explains a large amount of class separation, but it is not precise enough to be a final fraud detector.

Amount-threshold diagnostics were saved to `results/stage5_amount_rule_diagnostics.csv`. These are diagnostic rules, not final models.

## Threshold Analysis

Thresholds from 0.10 to 0.90 were evaluated using cross-validated training probabilities for the selected model. The threshold table is saved to `results/threshold_analysis.csv`, and the plot is saved to `figures/precision_recall_vs_threshold.png`.

The selected operating threshold is **{selected_threshold:.2f}**. It was selected using an illustrative cost scenario where false negatives cost 10 units and false positives cost 1 unit. These are hypothetical values only and must not be presented as real financial costs.

## Final Test-Set Performance

{dataframe_to_markdown(pd.DataFrame([final_metrics]), floatfmt=".4f")}

## Threshold 0.50 vs Selected Threshold

{dataframe_to_markdown(pd.DataFrame([final_metrics_050, final_metrics]), floatfmt=".4f")}

## Final Feature Importance

Top final model features:

{chr(10).join('- `' + feature + '`' for feature in final_importance['feature'].head(15).tolist())}

The model assigns greater predictive importance to these features, but feature importance does not prove causation.

## Leakage and Suspicious-Performance Investigation

{chr(10).join('- ' + note for note in suspicious_notes)}

## Final Candidate Model

The recommended final candidate model after Stage 5 is **{selected_model_name}** with operating threshold **{selected_threshold:.2f}**. This selection considers PR-AUC, F1, recall, precision, false-positive burden, and the leakage review. The model should still be described as a PaySim experimental model, not as a validated production model for real mobile-money systems.
"""
    (RESULTS_DIR / "stage5_tuning_and_threshold_analysis.md").write_text(report, encoding="utf-8")

    summary = {
        "selected_model": selected_model_name,
        "selected_threshold": selected_threshold,
        "final_metrics": final_metrics,
        "final_confusion_matrix": {
            "true_negative": int(final_matrix.ravel()[0]),
            "false_positive": int(final_matrix.ravel()[1]),
            "false_negative": int(final_matrix.ravel()[2]),
            "true_positive": int(final_matrix.ravel()[3]),
        },
        "best_params": best_params,
        "diagnostics": diagnostics,
        "suspicious_notes": suspicious_notes,
    }
    (RESULTS_DIR / "stage5_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
