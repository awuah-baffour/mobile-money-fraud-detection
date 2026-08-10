# Stage 5: Hyperparameter Tuning and Threshold Analysis

## Hyperparameter Tuning Methodology

Tuning used only the 80,000-row training split. A 5-fold `StratifiedKFold(shuffle=True, random_state=42)` cross-validation strategy was used. `RandomizedSearchCV` optimized average precision, which corresponds to PR-AUC and is appropriate for an imbalanced fraud-detection task.

## Cross-Validation Results

| Model | CV ROC-AUC mean | CV ROC-AUC std | CV PR-AUC mean | CV PR-AUC std | CV F1 mean | CV Precision mean | CV Recall mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9962 | 0.0005 | 0.9507 | 0.0022 | 0.8706 | 0.9110 | 0.8337 |
| Random Forest | 0.9995 | 0.0004 | 0.9969 | 0.0008 | 0.9704 | 0.9455 | 0.9967 |
| XGBoost | 0.9998 | 0.0002 | 0.9990 | 0.0008 | 0.9950 | 0.9926 | 0.9975 |

## Tuned Hyperparameters

| Model | Best Parameters |
| --- | --- |
| Logistic Regression | {"classifier__class_weight": null, "classifier__C": 5.0} |
| Random Forest | {"classifier__n_estimators": 150, "classifier__min_samples_split": 10, "classifier__min_samples_leaf": 1, "classifier__max_features": "log2", "classifier__max_depth": null, "classifier__class_weight": "balanced"} |
| XGBoost | {"classifier__colsample_bytree": 0.8391883316733973, "classifier__gamma": 0.11237380387495231, "classifier__learning_rate": 0.10512352997898983, "classifier__max_depth": 3, "classifier__min_child_weight": 5, "classifier__n_estimators": 134, "classifier__reg_alpha": 0.066106775625201, "classifier__reg_lambda": 0.823454610111791, "classifier__scale_pos_weight": 19.0, "classifier__subsample": 0.9430611923241643} |

## Tuned Model Comparison on Test Set at Threshold 0.50

| Model | Tuned? | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Yes | 0.9871 | 0.8972 | 0.8380 | 0.8666 | 0.9938 | 0.9422 |
| Random Forest | Yes | 0.9967 | 0.9431 | 0.9940 | 0.9679 | 0.9975 | 0.9935 |
| XGBoost | Yes | 0.9994 | 0.9950 | 0.9940 | 0.9945 | 0.9990 | 0.9963 |

## Baseline vs Tuned Results

| Model | Version | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | Baseline | 0.9583 | 0.5457 | 0.9920 | 0.7040 | 0.9937 | 0.9349 |
| Logistic Regression | Tuned | 0.9871 | 0.8972 | 0.8380 | 0.8666 | 0.9938 | 0.9422 |
| Random Forest | Baseline | 0.9957 | 0.9247 | 0.9940 | 0.9581 | 0.9976 | 0.9935 |
| Random Forest | Tuned | 0.9967 | 0.9431 | 0.9940 | 0.9679 | 0.9975 | 0.9935 |
| XGBoost | Baseline | 0.9996 | 0.9980 | 0.9940 | 0.9960 | 0.9988 | 0.9959 |
| XGBoost | Tuned | 0.9994 | 0.9950 | 0.9940 | 0.9945 | 0.9990 | 0.9963 |

## Diagnostic Rule Checks

The simple rule `type in [TRANSFER, CASH_OUT]` produced precision 0.1082, recall 1.0000, and F1 0.1953. This confirms that transaction type explains a large amount of class separation, but it is not precise enough to be a final fraud detector.

Amount-threshold diagnostics were saved to `results/stage5_amount_rule_diagnostics.csv`. These are diagnostic rules, not final models.

## Threshold Analysis

Thresholds from 0.10 to 0.90 were evaluated using cross-validated training probabilities for the selected model. The threshold table is saved to `results/threshold_analysis.csv`, and the plot is saved to `figures/precision_recall_vs_threshold.png`.

The selected operating threshold is **0.55**. It was selected using an illustrative cost scenario where false negatives cost 10 units and false positives cost 1 unit. These are hypothetical values only and must not be presented as real financial costs.

## Final Test-Set Performance

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.5500 | 0.9994 | 0.9950 | 0.9940 | 0.9945 | 0.9990 | 0.9963 | 18995 | 5 | 6 | 994 |

## Threshold 0.50 vs Selected Threshold

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.5000 | 0.9994 | 0.9950 | 0.9940 | 0.9945 | 0.9990 | 0.9963 | 18995 | 5 | 6 | 994 |
| XGBoost | 0.5500 | 0.9994 | 0.9950 | 0.9940 | 0.9945 | 0.9990 | 0.9963 | 18995 | 5 | 6 | 994 |

## Final Feature Importance

Top final model features:

- `origin_balance_sufficient`
- `amount_to_origin_balance_ratio`
- `type_CASH_IN`
- `type_PAYMENT`
- `log_amount`
- `oldbalanceOrg`
- `amount_to_destination_balance_ratio`
- `amount`
- `type_TRANSFER`
- `origin_balance_zero`
- `oldbalanceDest`
- `type_CASH_OUT`
- `step`
- `step_day`
- `destination_balance_zero`

The model assigns greater predictive importance to these features, but feature importance does not prove causation.

## Leakage and Suspicious-Performance Investigation

- Tree-based model performance remains very high even after excluding raw identifiers, isFlaggedFraud, and post-transaction balances.
- The simple transaction-type rule has perfect recall but low precision, showing that PaySim's fraud labels are strongly linked to TRANSFER and CASH_OUT in this sample.
- Feature importance is dominated by balance sufficiency and balance-ratio behavior, suggesting the models may be learning PaySim-specific simulation patterns.
- The results demonstrate strong performance within the reduced PaySim experimental environment, not guaranteed real-world Ghanaian mobile-money performance.

## Final Candidate Model

The recommended final candidate model after Stage 5 is **XGBoost** with operating threshold **0.55**. This selection considers PR-AUC, F1, recall, precision, false-positive burden, and the leakage review. The model should still be described as a PaySim experimental model, not as a validated production model for real mobile-money systems.
