# Stage 2: Data Cleaning and Exploratory Data Analysis

## Cleaning and Validation Summary

The raw dataset was copied before analysis. No rows or columns were removed, and no valid values were changed.

| Check | Result |
|---|---:|
| Rows | 100,000 |
| Columns | 11 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Negative transaction amounts | 0 |
| Negative balance values | 0 |
| Unexpected transaction types | 0 |
| Unexpected target values | 0 |

The 5% fraud representation should be described as a deliberately constructed reduced PaySim subset for machine-learning experimentation, not as the natural fraud prevalence of PaySim or Ghana.

## Leakage Investigation

`isFlaggedFraud` flagged 10 transactions. All 10 were fraudulent, giving precision of 1.0000, but it detected only 10 of 5,000 fraud cases, giving recall of 0.0020. This column should be analyzed separately as a PaySim rule-based indicator and excluded from the main ML feature set unless the report explicitly justifies a separate experiment with it.

`nameOrig` has 99,999 unique values and only one repeated sender identifier. `nameDest` has 92,971 unique values, with repeated recipients appearing more often but still as identifiers. These columns should not be converted into arbitrary numeric codes for modeling because such codes would impose false numerical order and may not generalize.

Balance variables require caution. Pre-transaction balances may be transaction-time information, while post-transaction balances such as `newbalanceOrig` and `newbalanceDest` may reflect information available only after the transaction is processed. Derived balance-change variables are useful for EDA, but they should be treated carefully during Stage 3 feature engineering.

## Key Findings From Exploratory Data Analysis

1. Observation: The dataset contains 95,000 legitimate transactions and 5,000 fraudulent transactions.
   Interpretation: The dataset is imbalanced, so accuracy alone is insufficient for evaluation.
   Limitation: The 5% fraud representation was intentionally constructed and should not be treated as real-world fraud prevalence.

2. Observation: Fraud occurs only in `TRANSFER` and `CASH_OUT` transactions in this sample.
   Interpretation: Transaction type is likely to be useful for fraud detection.
   Limitation: This is an observation from the PaySim subset and not a universal claim about mobile-money fraud.

3. Observation: `TRANSFER` has the highest fraud rate at 23.5860%, while `CASH_OUT` has a fraud rate of 7.1058%.
   Interpretation: Fraud rate provides a clearer risk comparison than fraud count alone.
   Limitation: The rate depends on this sampled dataset and the synthetic PaySim generation process.

4. Observation: Fraudulent transactions have a higher mean amount than legitimate transactions.
   Interpretation: Transaction amount may help distinguish fraud from legitimate behavior.
   Limitation: Higher transaction amount should not be interpreted as causing fraud.

5. Observation: The highest amount quintile has a fraud rate of 15.2550%.
   Interpretation: Large-value transactions appear more associated with fraud in this sample.
   Limitation: Some legitimate transactions are also large, so amount alone is not enough for classification.

6. Observation: Fraud cases have a median origin balance change of 447,531.94, compared with 0.00 for legitimate transactions.
   Interpretation: Origin-account balance behavior may be informative.
   Limitation: Balance-change features may include post-transaction information and need leakage review before final modeling.

7. Observation: `isFlaggedFraud` has perfect precision in this sample but very low recall.
   Interpretation: It catches a very small number of highly suspicious transactions.
   Limitation: It is likely a rule-based PaySim signal and should not be blindly used as an ML feature.

8. Observation: The final time decile contains 2,269 fraud cases and a fraud rate of 22.69%.
   Interpretation: Fraud appears more concentrated in later simulation steps.
   Limitation: `step` is a simulation time variable and may not map directly to real calendar time.

## Recommended Stage 3 Direction

Stage 3 should create explainable, transaction-time-aware features. Candidate features include transaction type encoding, transaction amount transformations, sender pre-transaction balance features, and carefully justified balance-change indicators. `isFlaggedFraud`, raw identifiers, and post-transaction balance variables should be treated as potential leakage risks and not included in the final feature set without explicit justification.
