# Stage 3: Feature Engineering and Final Model Features

## Prediction Scenario

The primary model is designed to identify potentially fraudulent mobile-money transactions using information that could reasonably be available when the transaction is being evaluated. This means the main feature set prioritizes transaction-time information and excludes post-transaction balances from the primary experiment.

## Final Primary Feature List

These are the raw and engineered columns that will be used for the main machine-learning experiment before one-hot encoding:

- `step`
- `amount`
- `log_amount`
- `oldbalanceOrg`
- `oldbalanceDest`
- `amount_to_origin_balance_ratio`
- `amount_to_destination_balance_ratio`
- `origin_balance_zero`
- `destination_balance_zero`
- `origin_balance_sufficient`
- `step_day`
- `type`

After preprocessing, `type` is one-hot encoded, producing 16 total model input columns:

- `step`
- `amount`
- `log_amount`
- `oldbalanceOrg`
- `oldbalanceDest`
- `amount_to_origin_balance_ratio`
- `amount_to_destination_balance_ratio`
- `origin_balance_zero`
- `destination_balance_zero`
- `origin_balance_sufficient`
- `step_day`
- `type_CASH_IN`
- `type_CASH_OUT`
- `type_DEBIT`
- `type_PAYMENT`
- `type_TRANSFER`

## Excluded Features

- `nameOrig`: High-cardinality sender identifier; raw encoding would create arbitrary numeric relationships and poor generalization.
- `nameDest`: High-cardinality recipient identifier; raw encoding would create arbitrary numeric relationships and poor generalization.
- `isFlaggedFraud`: Existing PaySim rule-based fraud flag; Stage 2 precision was 1.0000 but recall was only 0.0020, so it is treated separately.
- `newbalanceOrig`: Post-transaction sender balance; may not be available during real-time transaction evaluation.
- `newbalanceDest`: Post-transaction recipient balance; may not be available during real-time transaction evaluation.
- `origin_balance_change`: Requires newbalanceOrig, so it is a post-transaction feature excluded from the primary model.
- `destination_balance_change`: Requires newbalanceDest, so it is a post-transaction feature excluded from the primary model.

## Preprocessing Approach

The preprocessing code is saved in `src/preprocessing.py`. The primary preprocessing pipeline uses a `ColumnTransformer` with two branches. Numerical features are median-imputed and scaled with `StandardScaler`, which supports Logistic Regression in the next modeling stage. The categorical `type` feature is imputed with the most frequent value and encoded with `OneHotEncoder(handle_unknown="ignore")`. The preprocessor is fitted only on `X_train` during the sanity check, then applied to `X_test`.

## Train/Test Split

The split uses `train_test_split(test_size=0.20, stratify=y, random_state=42)`.

| Split | Rows | Legitimate | Fraudulent | Fraud Rate |
|---|---:|---:|---:|---:|
| Train | 80,000 | 76,000 | 4,000 | 5.00% |
| Test | 20,000 | 19,000 | 1,000 | 5.00% |

## Sanity Checks

- Missing values in primary X: 0
- Infinite numeric values in primary X: False
- Forbidden leakage columns in primary X: []
- Raw feature count before encoding: 12
- Encoded feature count after preprocessing: 16
- Training matrix after preprocessing: 80,000 rows x 16 columns
- Test matrix after preprocessing: 20,000 rows x 16 columns

## Feature Engineering Decisions

`amount` is retained because it is a core transaction attribute and Stage 2 showed that fraudulent transactions in this dataset tended to have larger transaction values. `log_amount` is added because transaction amounts are highly right-skewed, and the log transform reduces the influence of extreme values while keeping the original amount available.

`oldbalanceOrg` and `oldbalanceDest` are retained in the primary set because they are pre-transaction balances. However, they are marked for careful discussion because PaySim documentation warns that balance columns can be problematic for fraud detection. The post-transaction balances are excluded from the primary model.

The ratio features compare transaction value with recorded pre-transaction balances. A denominator constant of 1.0 is used to avoid division by zero. These features are easy to explain: they show whether a transaction is small or large relative to the sender or destination balance.

The zero-balance indicators are included because Stage 2 showed many zero balance values. These binary features let models identify the presence of zero balances without relying only on large ratio values.

`origin_balance_sufficient` is included because it captures whether the sender's recorded starting balance can cover the transaction amount. It uses only `amount` and `oldbalanceOrg`, so it is available at transaction time.

`type` is encoded with one-hot encoding rather than integer encoding. This avoids creating a false order among transaction categories such as `CASH_IN`, `PAYMENT`, and `TRANSFER`.

`step` is retained as a simulation time variable. Because PaySim documents one step as one hour, `step_day` is also created as a broad day-level period. No more detailed time-of-day behavioral claim is made at this stage.

A secondary post-transaction comparison feature set is documented but not selected as the main feature set. It contains `newbalanceOrig`, `newbalanceDest`, `origin_balance_change`, and `destination_balance_change` so that later experiments can explicitly show the effect of including post-transaction information.

## Remaining Methodological Concerns

The PaySim documentation notes that fraudulent transactions are cancelled and warns that balance columns can create issues for fraud detection. For that reason, Stage 4 should train the primary transaction-time model first and, only if useful, run a clearly labeled post-transaction comparison. Any performance improvement from post-transaction features should be discussed as potentially caused by information availability rather than simply as a better real-time model.
