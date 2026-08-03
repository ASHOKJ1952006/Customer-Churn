"""
run_dataset1.py — Telco Customer Churn (primary dataset, comparable to the base paper)
"""

import pandas as pd
from pipeline import run_full_pipeline, save_model_files

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

result1 = run_full_pipeline(X, y, numeric_cols, dataset_name="Telco Customer Churn (Dataset 1)")
save_model_files(result1, prefix='ds1')

# Save results summary for the final comparison report
import joblib
joblib.dump(result1['final_metrics'], 'ds1_final_metrics.pkl')
joblib.dump(result1['overfit_df'], 'ds1_overfit_df.pkl')
joblib.dump(result1['efficiency_df'], 'ds1_efficiency_df.pkl')

print("\n\nDataset 1 pipeline complete.")