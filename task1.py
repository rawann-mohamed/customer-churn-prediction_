# ============================================================
# TASK 1 — Data Preparation, EDA, and Preprocessing
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs("figures", exist_ok=True)


# ============================================================
# PART 1: LOAD AND MERGE DATA
# ============================================================

df_demo    = pd.read_excel("Customer_Churn_Data_Large.xlsx", sheet_name="Customer_Demographics")
df_trans   = pd.read_excel("Customer_Churn_Data_Large.xlsx", sheet_name="Transaction_History")
df_service = pd.read_excel("Customer_Churn_Data_Large.xlsx", sheet_name="Customer_Service")
df_online  = pd.read_excel("Customer_Churn_Data_Large.xlsx", sheet_name="Online_Activity")
df_churn   = pd.read_excel("Customer_Churn_Data_Large.xlsx", sheet_name="Churn_Status")

# aggregate multi-row sheets to one row per customer
trans_agg   = df_trans.groupby('CustomerID').agg(AmountSpent=('AmountSpent', 'sum')).reset_index()
service_agg = df_service.groupby('CustomerID').agg(ServiceInteractions=('InteractionID', 'count')).reset_index()

# merge all sheets on CustomerID
df = df_demo \
    .merge(trans_agg,   on='CustomerID', how='left') \
    .merge(service_agg, on='CustomerID', how='left') \
    .merge(df_online,   on='CustomerID', how='left') \
    .merge(df_churn,    on='CustomerID', how='left')

# CustomerID was only needed for merging
df.drop(columns=['CustomerID'], inplace=True)

# convert LastLoginDate to number of days since last login
# use a dataset-based reference date so reruns stay reproducible
df['LastLoginDate'] = pd.to_datetime(df['LastLoginDate'])
reference_date = df['LastLoginDate'].max() + pd.Timedelta(days=1)
df['DaysSinceLastLogin'] = (reference_date - df['LastLoginDate']).dt.days
df.drop(columns=['LastLoginDate'], inplace=True)

print("Dataset shape:", df.shape)
print(df.head())


# ============================================================
# PART 2: EDA — EXPLORATORY DATA ANALYSIS
# ============================================================

print(df.describe())
print("\nChurn rate:", df['ChurnStatus'].mean() * 100, "%")


# Plot 1: Churn Distribution
counts = df['ChurnStatus'].value_counts().sort_index()
plt.bar(['Retained (0)', 'Churned (1)'], counts.values, color=['steelblue', 'tomato'])
plt.title("Churn Distribution")
plt.ylabel("Count")
plt.savefig("figures/01_churn_distribution.png")
plt.close()

# Plot 2: Histograms
df[['Age', 'AmountSpent', 'LoginFrequency', 'ServiceInteractions', 'DaysSinceLastLogin']].hist(
    figsize=(14, 6), color='steelblue', edgecolor='white')
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("figures/02_histograms.png")
plt.close()

# Plot 3: Box Plots vs Churn
features = ['Age', 'AmountSpent', 'LoginFrequency', 'ServiceInteractions', 'DaysSinceLastLogin']
fig, axes = plt.subplots(1, len(features), figsize=(20, 5))
for i, col in enumerate(features):
    sns.boxplot(x='ChurnStatus', y=col, data=df, palette=["steelblue", "tomato"], ax=axes[i])
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel("Churn (0=No, 1=Yes)")
plt.suptitle("Box Plots by Churn Status")
plt.tight_layout()
plt.savefig("figures/03_boxplots.png")
plt.close()

# Plot 4: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features + ['ChurnStatus']].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("figures/04_correlation_heatmap.png")
plt.close()

# Plot 5: Scatter — LoginFrequency vs AmountSpent
colors = df['ChurnStatus'].map({0: "steelblue", 1: "tomato"})
plt.figure(figsize=(7, 5))
plt.scatter(df['LoginFrequency'], df['AmountSpent'], c=colors, alpha=0.4)
plt.xlabel("Login Frequency")
plt.ylabel("Amount Spent")
plt.title("Login Frequency vs Amount Spent")
plt.tight_layout()
plt.savefig("figures/05_scatter.png")
plt.close()

# Plot 6: Churn Rate by Gender
if 'Gender' in df.columns:
    churn_by_gender = df.groupby('Gender')['ChurnStatus'].mean() * 100
    churn_by_gender.plot(kind='bar', color='steelblue', edgecolor='white')
    plt.title("Churn Rate by Gender")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/06_churn_by_gender.png")
    plt.close()

print("All figures saved to figures/")


# ============================================================
# PART 3: PREPROCESSING
# ============================================================

# --- Step 1: Missing Values (selective fillna) ---
print("\nMissing values before fill:")
print(df.isnull().sum())

# AmountSpent and ServiceInteractions: missing means 0 (no transaction / no contact)
df['AmountSpent']         = df['AmountSpent'].fillna(0)
df['ServiceInteractions'] = df['ServiceInteractions'].fillna(0)

# numeric columns: fill with median
for col in ['Age', 'LoginFrequency', 'DaysSinceLastLogin']:
    df[col] = df[col].fillna(df[col].median())

# categorical columns: fill with mode
for col in ['Gender', 'MaritalStatus', 'IncomeLevel', 'ServiceUsage']:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after fill:", df.isnull().sum().sum())


# --- Step 2: Feature Engineering ---
# SpendPerLogin captures how much a customer spends per visit
# +1 avoids division by zero for customers with LoginFrequency = 0
df['SpendPerLogin'] = df['AmountSpent'] / (df['LoginFrequency'] + 1)

# ActivityLevel groups customers into Low / Medium / High engagement
df['ActivityLevel'] = pd.cut(df['LoginFrequency'], bins=3,
                              labels=['Low', 'Medium', 'High'])


# --- Step 3: Outlier Detection and Capping (IQR method) ---
print("\nOutlier Detection (IQR):")
num_cols = ['Age', 'AmountSpent', 'LoginFrequency', 'ServiceInteractions',
            'DaysSinceLastLogin', 'SpendPerLogin']

for col in num_cols:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"  {col}: {n_out} outliers — capped to [{lower:.1f}, {upper:.1f}]")
    df[col] = df[col].clip(lower=lower, upper=upper)


# save a cleaned, human-readable dataset before one-hot encoding
df.to_csv("cleaned_data.csv", index=False)
print("\nCleaned data saved - shape:", df.shape)

# --- Step 4: One-Hot Encoding ---
# convert categorical text columns to numeric binary columns
cat_cols = ['Gender', 'MaritalStatus', 'IncomeLevel', 'ServiceUsage', 'ActivityLevel']
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

print("\nColumns after encoding:", df.columns.tolist())


# --- No Scaling here — Scaling happens inside the model Pipeline in Task 2 ---

df.to_csv("preprocessed_data.csv", index=False)
print("\nPreprocessed data saved — shape:", df.shape)

print("AmountSpent -> Mean:", df['AmountSpent'].mean(), "Median:", df['AmountSpent'].median())
print("LoginFrequency -> Mean:", df['LoginFrequency'].mean(), "Median:", df['LoginFrequency'].median())