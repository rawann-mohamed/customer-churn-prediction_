# Customer Churn Prediction Pipeline
### Lloyds Banking Group — Neural Networks Assignment

An end-to-end ML pipeline that identifies banking customers at risk of churning, built on a synthetic multi-source dataset simulating a real banking environment.

---

## Pipeline Overview

```
Raw Excel (5 sheets)
       │
       ▼
  Task 1: ETL + EDA + Preprocessing
  ├── Merge 5 data sources on CustomerID
  ├── Aggregate transactions & service interactions
  ├── Feature engineering (SpendPerLogin, DaysSinceLastLogin, ActivityLevel)
  ├── Missing value imputation + IQR outlier capping
  └── One-hot encoding → preprocessed_data.csv
       │
       ▼
  Task 2: Modelling + Business Output
  ├── Stratified 70/15/15 train-val-test split
  ├── SMOTE inside pipeline (no data leakage)
  ├── 6 classifiers evaluated on validation set
  ├── GridSearchCV hyperparameter tuning on best model
  ├── Decision threshold tuning (optimised for F1)
  └── High-risk customer list exported → high_risk_customers.csv
```

---

## Models Evaluated

| Model               | Tuned |
|---------------------|-------|
| Logistic Regression | ✓     |
| Decision Tree       | ✓     |
| Random Forest       | ✓     |
| SVM                 | ✓     |
| XGBoost             | ✓     |
| Neural Network (MLP)| ✓     |

Model selection: top-2 by F1-Score, tie-broken by ROC-AUC.

---

## Key Design Decisions

- **SMOTE inside ImbPipeline** — prevents synthetic samples from leaking into val/test sets
- **Threshold tuning** — default 0.5 threshold replaced by the value maximising validation F1
- **SelectKBest (k=15)** — drops noisy features before training distance-based models
- **Business output** — customers with churn probability > 0.70 exported for retention team

---

## How to Run

```bash
pip install -r requirements.txt

# Step 1 — Data prep & EDA (generates preprocessed_data.csv + figures/)
python task1.py

# Step 2 — Modelling (generates high_risk_customers.csv)
python task2.py
```

> The source data file `Customer_Churn_Data_Large.xlsx` must be in the same directory.

---

## Tech Stack

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `imbalanced-learn` · `Matplotlib` · `Seaborn`
