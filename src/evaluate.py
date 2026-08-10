"""Evaluation helpers for baseline fraud-detection models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


THRESHOLD = 0.50


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, float | str]:
    """Calculate standard baseline classification metrics."""
    y_pred = (y_probability >= threshold).astype(int)
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_probability),
        "PR-AUC": average_precision_score(y_true, y_probability),
    }


def classification_report_frame(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Return a classification report as a dataframe."""
    y_pred = (y_probability >= threshold).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate", "Fraudulent"],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    title: str,
    output_path: str | Path,
    threshold: float = THRESHOLD,
) -> np.ndarray:
    """Save a labeled confusion matrix plot."""
    y_pred = (y_probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Legitimate", "Predicted Fraud"],
        yticklabels=["Actual Legitimate", "Actual Fraud"],
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close()
    return matrix


def plot_roc_curves(
    y_true: pd.Series | np.ndarray,
    probabilities: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Save a combined ROC curve plot."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for model_name, y_probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, y_probability)
        auc_value = roc_auc_score(y_true, y_probability)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close()


def plot_precision_recall_curves(
    y_true: pd.Series | np.ndarray,
    probabilities: dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    """Save a combined Precision-Recall curve plot."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for model_name, y_probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, y_probability)
        pr_auc = average_precision_score(y_true, y_probability)
        ax.plot(recall, precision, linewidth=2, label=f"{model_name} (PR-AUC={pr_auc:.4f})")

    baseline = float(np.mean(y_true))
    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        linewidth=1,
        label=f"Fraud prevalence baseline ({baseline:.2%})",
    )
    ax.set_title("Precision-Recall Curve Comparison")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close()


def plot_feature_importance(
    feature_names: list[str],
    values: np.ndarray,
    title: str,
    output_path: str | Path,
    top_n: int = 15,
    absolute: bool = True,
) -> pd.DataFrame:
    """Save a horizontal feature-importance plot and return sorted values."""
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
            "absolute_importance": np.abs(values),
        }
    )
    sort_column = "absolute_importance" if absolute else "importance"
    importance = importance.sort_values(sort_column, ascending=False)
    plot_data = importance.head(top_n).sort_values(sort_column, ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.barplot(data=plot_data, x="importance", y="feature", color="#4E79A7", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance" if not absolute else "Coefficient / Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close()
    return importance
