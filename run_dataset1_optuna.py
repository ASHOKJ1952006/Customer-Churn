"""
run_dataset1_optuna.py — Telco Customer Churn, tuned with Optuna instead of RandomizedSearchCV.
Run this AFTER run_dataset1.py to compare whether Optuna's smarter search finds a better model.
Saves files with prefix 'ds1opt_' so they don't overwrite your existing ds1_ results.
"""

import pandas as pd
import joblib
from pipeline import run_full_pipeline_optuna, save_model_files

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df.drop('customerID', axis=1, inplace=True)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

result1_optuna = run_full_pipeline_optuna(
    X, y, numeric_cols, dataset_name="Telco Customer Churn (Dataset 1, Optuna)", n_trials=40
)
save_model_files(result1_optuna, prefix='ds1opt')

joblib.dump(result1_optuna['final_metrics'], 'ds1opt_final_metrics.pkl')
joblib.dump(result1_optuna['overfit_df'], 'ds1opt_overfit_df.pkl')
joblib.dump(result1_optuna['efficiency_df'], 'ds1opt_efficiency_df.pkl')

print("\n\nDataset 1 (Optuna) pipeline complete.")
print("\nCompare against your existing ds1_final_metrics.pkl (RandomizedSearchCV) to see if Optuna improved results.")