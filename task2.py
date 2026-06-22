import pandas as pd
import numpy as np
import warnings
# Suppressing warnings to keep the terminal output clean and professional
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix)

# imblearn pipeline is crucial here instead of standard sklearn pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ============================================================
# 1. LOAD DATA & SPLIT (70/15/15)
# ============================================================
# Load the preprocessed data (cleaned and encoded in Task 1)
df = pd.read_csv("preprocessed_data.csv")

# Separate features (X) and target variable (y)
X = df.drop(columns=['ChurnStatus'])
y = df['ChurnStatus']

# --- STRATIFIED SPLITTING ---
# We use stratify=y to ensure the exact same proportion of churners (imbalanced class) 
# is maintained across the Train, Validation, and Test sets.

# First split: 70% Train, 30% Temp (Temp will be halved into Val/Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Second split: Divide Temp exactly in half -> 15% Validation, 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Data Split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ============================================================
# 2. DEFINE PIPELINES
# ============================================================
# WHY WE USE AN IMBLEARN PIPELINE: 
# If we apply SMOTE to the whole dataset before splitting, synthetic data will leak 
# into the validation/test sets, giving falsely high scores (Data Leakage). 
# The pipeline ensures SMOTE is ONLY applied to the training data during cross-validation.

def create_pipeline(model):
    return ImbPipeline([
        ('scaler', StandardScaler()), # Step 1: Distance-based models and SMOTE need scaled data
        ('smote', SMOTE(random_state=42)), # Step 2: Handle class imbalance by synthesizing minority class
        ('select', SelectKBest(score_func=f_classif, k=15)), # Step 3: Dimensionality reduction (drop noisy features)
        ('model', model) # Step 4: The chosen classifier
    ])

# Initialize a dictionary of models to evaluate
pipelines = {
    "Logistic Regression": create_pipeline(LogisticRegression(max_iter=1000, random_state=42)),
    "Decision Tree": create_pipeline(DecisionTreeClassifier(random_state=42)),
    "Random Forest": create_pipeline(RandomForestClassifier(random_state=42)),
    "SVM": create_pipeline(SVC(probability=True, random_state=42)),
    "XGBoost": create_pipeline(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    "Neural Network": create_pipeline(MLPClassifier(max_iter=500, random_state=42))
}

# ============================================================
# 3. MODEL EVALUATION (ON VALIDATION SET)
# ============================================================
print("\nEvaluating base models on Validation Set...")
results = []

for name, pipe in pipelines.items():
    # Train strictly on X_train to keep validation set unseen
    pipe.fit(X_train, y_train)
    
    # Predict on X_val for unbiased model selection
    y_val_pred = pipe.predict(X_val)
    y_val_prob = pipe.predict_proba(X_val)[:, 1]
    
    # Collect a comprehensive suite of metrics for comparison
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred, zero_division=0),
        'Recall': recall_score(y_val, y_val_pred, zero_division=0),
        'F1-Score': f1_score(y_val, y_val_pred, zero_division=0), # Harmonic mean of Precision & Recall
        'ROC-AUC': roc_auc_score(y_val, y_val_prob)
    })

# Rank models by F1-Score because we want a balanced model.
# Focusing only on Recall would flag too many false positives (wasting retention budget).
results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False)
print("\nBase Model Performance (Ranked by F1-Score):")
print(results_df.to_string(index=False, float_format="%.4f"))

# ============================================================
# 4. HYPERPARAMETER TUNING (ON BEST MODEL)
# ============================================================
# Smart Model Selection: Take the top 2 models by F1-Score, 
# then pick the one with the higher ROC-AUC to break the tie.
results_df = results_df.sort_values(by='F1-Score', ascending=False)
top_models = results_df.head(2)
best_model_name = top_models.sort_values(by='ROC-AUC', ascending=False).iloc[0]['Model']

print(f"\nSelecting best model based on F1-Score/AUC logic: {best_model_name}")

# Define hyperparameter grids to optimize our winning model
param_grids = {
    "Logistic Regression": {'model__C': [0.01, 0.1, 1, 10]},
    "Decision Tree": {'model__max_depth': [5, 10, None], 'model__min_samples_split': [2, 5, 10]},
    "Random Forest": {'model__n_estimators': [100, 200], 'model__max_depth': [10, None]},
    "SVM": {'model__C': [0.1, 1, 10], 'model__gamma': ['scale', 'auto']},
    "XGBoost": {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1], 'model__max_depth': [3, 6]},
    "Neural Network": {'model__hidden_layer_sizes': [(64,), (64, 32)], 'model__alpha': [0.0001, 0.001]}
}

print(f"Running GridSearchCV for {best_model_name}...")
# GridSearchCV systematically tests combinations of parameters using Cross-Validation
grid_search = GridSearchCV(
    pipelines[best_model_name], 
    param_grids[best_model_name], 
    cv=5, 
    scoring='f1', # Explicitly optimize for F1-score during tuning
    n_jobs=-1     # Use all available CPU cores for faster execution
)

grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

# ============================================================
# 5. THRESHOLD TUNING (ON VALIDATION SET)
# ============================================================
print("\nTuning Decision Threshold on Validation Set...")
# Default ML threshold is 0.5. For imbalanced churn datasets, shifting the threshold 
# (e.g., to 0.3 or 0.4) often yields a much better F1-Score by catching more risk.
y_val_prob_best = best_pipeline.predict_proba(X_val)[:, 1]

thresholds = [0.2, 0.3, 0.4, 0.5]
best_threshold = 0.3 # Fallback default
max_val_f1 = 0.0

for thresh in thresholds:
    # Convert probability to binary 1 or 0 based on the current threshold loop
    val_preds_adjusted = (y_val_prob_best >= thresh).astype(int)
    current_f1 = f1_score(y_val, val_preds_adjusted, zero_division=0)
    
    print(f"Threshold: {thresh:.1f} | Validation F1-Score: {current_f1:.4f}")
    
    # Save the threshold that gives us the peak F1 performance
    if current_f1 > max_val_f1:
        max_val_f1 = current_f1
        best_threshold = thresh

print(f"\nOptimal Threshold Selected: {best_threshold} (F1: {max_val_f1:.4f})")

# ============================================================
# 6. FINAL EVALUATION (ON TEST SET)
# ============================================================
# on the X_test set, which the model has NEVER seen before.
print("\n============================================================")
print("FINAL TEST SET EVALUATION")
print("============================================================")

y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
y_test_pred_final = (y_test_prob >= best_threshold).astype(int)

# Unpack the confusion matrix to see exactly where the model succeeds/fails
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_final).ravel()

print(classification_report(y_test, y_test_pred_final, target_names=['Retained (0)', 'Churned (1)']))

print(f"ROC-AUC Score  : {roc_auc_score(y_test, y_test_prob):.4f}")
print(f"F1-Score       : {f1_score(y_test, y_test_pred_final):.4f}\n")

print("Confusion Matrix Breakdown:")
print(f"  True Positives (Caught Churners)    : {tp}") # Business Win: Churners we can save
print(f"  False Positives (False Alarms)      : {fp}") # Business Cost: Wasted retention offers
print(f"  True Negatives (Correct Retentions) : {tn}") # Safe customers
print(f"  False Negatives (Missed Churners)   : {fn}") # Business Loss: Churners that escaped

# ============================================================
# 7. BUSINESS OUTPUT: HIGH-RISK CUSTOMERS
# ============================================================
# Generating actionable insights for the Customer Retention Team.
# We isolate customers with a >70% probability of leaving for immediate intervention.
business_df = X_test.copy()
business_df['Actual_Churn_Status'] = y_test
business_df['Churn_Probability'] = y_test_prob

# Filter and sort so the highest risk customers are at the top of the list
high_risk_customers = business_df[business_df['Churn_Probability'] > 0.70]
high_risk_customers = high_risk_customers.sort_values(by='Churn_Probability', ascending=False)

print(f"\nGenerated {len(high_risk_customers)} high-risk customers (Probability > 0.7).")

# Export to CSV for business stakeholders
high_risk_customers.to_csv("high_risk_customers.csv", index=False)
print("Saved to 'high_risk_customers.csv'")

# ============================================================
# 8. FINAL MODEL INFO
# ============================================================
print("\nFinal Selected Model:")
print(best_model_name)

# ============================================================
# 9. HIGH-RISK BUSINESS ANALYSIS
# ============================================================
# Checking the ROI (Return on Investment) of our high-risk list.
# If the business calls everyone on the high-risk list, what percentage were actually churning?
tp_hr = len(high_risk_customers[high_risk_customers['Actual_Churn_Status'] == 1])

if len(high_risk_customers) > 0:
    precision_hr = tp_hr / len(high_risk_customers)
else:
    precision_hr = 0

print(f"\nHigh-Risk Customers Count: {len(high_risk_customers)}")
print(f"High-Risk Precision (Business Value): {precision_hr:.2f} (Meaning {precision_hr*100:.0f}% of this list actually churned)")

# ============================================================
# 10. EXTRA METRICS
# ============================================================
print("\nAdditional Metrics:")
print(f"Precision (Churn): {precision_score(y_test, y_test_pred_final):.4f} -> Reliability of a churn prediction.")
print(f"Recall (Churn)   : {recall_score(y_test, y_test_pred_final):.4f} -> Percentage of total churners successfully caught.")