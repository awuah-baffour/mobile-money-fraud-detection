# Stage 4: Baseline Model Training and Evaluation

## Experimental Setup

The baseline models were trained on the reduced 100,000-record PaySim experimental dataset using the Stage 3 primary transaction-time feature set. The train/test split used `test_size=0.20`, `stratify=y`, and `random_state=42`, producing 80,000 training rows and 20,000 test rows. The preprocessing pipeline was fitted only on the training data.

The evaluation threshold for this baseline stage is `0.50`. Predicted probabilities were saved for each model so that threshold analysis can be performed later.

## Class Imbalance Strategy

Logistic Regression and Random Forest use `class_weight="balanced"`. XGBoost uses `scale_pos_weight=19.0000`, calculated from the training set as legitimate training samples divided by fraudulent training samples.

Accuracy is reported, but it is not the main metric because the dataset is imbalanced. Precision, recall, F1-score, PR-AUC, and ROC-AUC are more informative for fraud detection.

## Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9583 | 0.5457 | 0.9920 | 0.7040 | 0.9937 | 0.9349 |
| Random Forest | 0.9957 | 0.9247 | 0.9940 | 0.9581 | 0.9976 | 0.9935 |
| XGBoost | 0.9996 | 0.9980 | 0.9940 | 0.9960 | 0.9988 | 0.9959 |

## Confusion Matrices

In this project, false positives are legitimate transactions incorrectly flagged as fraud, which could inconvenience customers or delay valid transactions. False negatives are fraudulent transactions incorrectly treated as legitimate, which may lead to financial loss.

| Model | true_negative | false_positive | false_negative | true_positive |
| --- | --- | --- | --- | --- |
| Logistic Regression | 18174 | 826 | 8 | 992 |
| Random Forest | 18919 | 81 | 6 | 994 |
| XGBoost | 18998 | 2 | 6 | 994 |

## ROC-AUC Comparison

The combined ROC curve is saved as `figures/roc_curve_comparison.png`. ROC-AUC measures ranking ability across thresholds, but a high ROC-AUC alone does not automatically identify the best fraud-detection model.

## Precision-Recall Comparison

The combined Precision-Recall curve is saved as `figures/precision_recall_curve_comparison.png`. PR-AUC is especially relevant because fraud is the minority class, so the precision-recall trade-off directly shows how well a model identifies fraud without overwhelming the system with false alarms.

## Preliminary Feature Importance

Feature importance plots were created for the three baseline models. These values indicate which features the models rely on more heavily; they do not prove causation.

### Logistic Regression
- `origin_balance_sufficient`
- `origin_balance_zero`
- `step_day`
- `step`
- `type_PAYMENT`
- `type_CASH_IN`
- `type_TRANSFER`
- `type_CASH_OUT`

### Random Forest
- `origin_balance_sufficient`
- `oldbalanceOrg`
- `type_PAYMENT`
- `type_CASH_IN`
- `amount_to_destination_balance_ratio`
- `amount_to_origin_balance_ratio`
- `type_CASH_OUT`
- `log_amount`

### XGBoost
- `origin_balance_sufficient`
- `amount_to_origin_balance_ratio`
- `type_CASH_IN`
- `amount_to_destination_balance_ratio`
- `oldbalanceOrg`
- `amount`
- `type_PAYMENT`
- `origin_balance_zero`

## Suspicious Performance and Leakage Review

- Random Forest produced an unusually high metric (Recall=0.9940, ROC-AUC=0.9976, PR-AUC=0.9935). This should be discussed carefully.
- XGBoost produced an unusually high metric (Recall=0.9940, ROC-AUC=0.9988, PR-AUC=0.9959). This should be discussed carefully.
- The primary feature set excludes raw identifiers, isFlaggedFraud, and post-transaction balance variables.
- PaySim balance variables still require careful academic discussion because the dataset documentation warns that fraudulent transactions are cancelled.

## Baseline Conclusion

The current strongest baseline candidate is **XGBoost** based on the Stage 4 balance of recall, precision, F1-score, PR-AUC, and ROC-AUC. This is not the final model. Final selection should wait until hyperparameter tuning, threshold analysis, and deeper feature-importance review are completed.

These results apply only to the reduced PaySim experimental dataset and should not be presented as performance on Ghanaian mobile-money systems or any real financial institution.
