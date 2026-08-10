# Machine Learning-Based Mobile Money Fraud Detection

## Contributors

1. ##### Name: BAFFOUR ABRAHAM AWUAH | Index Number: UEB3512623 | GitHub: https://github.com/awuah-baffour

2. ##### Name: ABDUL WAKIL MAGIWABA DI-ELUMA | Index Number: UEB3502723| GitHub: https://github.com/mawdieluma-design

3. ##### Name: SARBAH DANIEL| Index Number: UEB3512723 | GitHub: https://github.com/Sarbah08


## Overview

This project investigates machine-learning approaches for detecting fraudulent mobile-money transactions using a reduced 100,000-record experimental subset of the synthetic PaySim dataset.

The selected project title is:

**Machine Learning-Based Mobile Money Fraud Detection: An Experimental Study Using the PaySim Dataset**

The project does **not** use real Ghanaian mobile-money transaction data and does **not** claim deployment-ready performance for MTN Mobile Money, Telecel Cash, AirtelTigo Money, or any financial institution.

## Objectives

1. Analyze transaction patterns associated with fraudulent activity.
2. Engineer meaningful transaction-time features.
3. Compare Logistic Regression, Random Forest, and XGBoost for fraud classification.
4. Evaluate and select a suitable model using fraud-sensitive metrics.

## Dataset

The project uses the **PaySim Synthetic Financial Dataset for Fraud Detection**, originally obtained from Kaggle.

Due to computational limitations, a reduced 100,000-record experimental dataset was prepared from the PaySim data and saved as:

```text
paysim_100k.csv
```

The reduced dataset contains:

| Class      |       Count |
| ---------- | ----------: |
| Legitimate |      95,000 |
| Fraudulent |       5,000 |
| **Total**  | **100,000** |

The dataset therefore contains a 5% fraudulent class representation.

The dataset included in this repository allows the complete notebook workflow to be reproduced without requiring the original multi-million-record PaySim dataset.


## Methodology

- Data cleaning and validation
- Exploratory data analysis
- Leakage-aware feature engineering
- Scikit-learn preprocessing pipeline
- Baseline model training
- Cross-validation
- Hyperparameter tuning
- Threshold analysis
- Final model selection

## Models

- Logistic Regression
- Random Forest
- XGBoost

## Final Results

Final selected model: **XGBoost**

Selected threshold: **0.55**

| Metric | Value |
|---|---:|
| Accuracy | 0.99945 |
| Precision | 0.99499 |
| Recall | 0.99400 |
| F1 | 0.99450 |
| ROC-AUC | 0.99904 |
| PR-AUC | 0.99630 |

Final confusion matrix:

| TN | FP | FN | TP |
|---:|---:|---:|---:|
| 18,995 | 5 | 6 | 994 |

## Project Structure

```text
AI/
├── paysim_100k.csv
├── notebooks/
│   └── mobile_money_fraud_detection.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   └── predict.py
├── models/
│   └── final_pipeline.pkl
├── figures/
├── results/
├── report/
│   └── mobile_money_fraud_detection_report.docx
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Notebook

Open:

```text
notebooks/mobile_money_fraud_detection.ipynb
```

The notebook contains the complete end-to-end machine-learning workflow and uses the included `paysim_100k.csv` dataset.

The workflow includes:

* Dataset loading and inspection
* Data quality validation
* Exploratory data analysis
* Feature engineering
* Leakage analysis
* Data preprocessing
* Baseline model training
* Cross-validation
* Hyperparameter tuning
* Threshold analysis
* Final XGBoost evaluation
* Feature importance analysis
* Transaction-type rule baseline

The notebook can be executed from beginning to end to reproduce the experiment.


## Train Baseline Models

```bash
python src/train.py
```

## Run Tuning and Threshold Analysis

```bash
python src/tune.py
```

## Make a Prediction

Example:

```bash
python src/predict.py --step 274 --type TRANSFER --amount 379057.93 --oldbalanceOrg 379057.93 --oldbalanceDest 0
```

The script returns:

```text
Fraud Probability
Prediction
Threshold
```

## Limitations

- PaySim is synthetic.
- The dataset is a reduced 100,000-record subset.
- The 5% fraud representation is artificial and experimental.
- The project does not use real Ghanaian transaction data.
- High performance may partly reflect PaySim-specific simulation patterns.
- External validation is required before real-world deployment claims.


