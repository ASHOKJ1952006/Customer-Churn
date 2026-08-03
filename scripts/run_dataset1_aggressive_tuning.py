"""
run_dataset1_aggressive_tuning.py — Aggressive hyperparameter tuning with wider search ranges
and more trials to potentially achieve higher accuracy than 77.22%.
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

# Wider search bounds for aggressive tuning
rf_bounds = {
    'n_estimators': (100, 800),
    'max_depth': (3, 30),
    'min_samples_split': (2, 50),
    'min_samples_leaf': (1, 30)
}

xgb_bounds = {
    'n_estimators': (100, 800),
    'max_depth': (2, 12),
    'learning_rate': (0.001, 0.3),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.3, 1.0),
    'min_child_weight': (1, 20),
    'reg_alpha': (1e-4, 10.0),
    'reg_lambda': (1e-4, 10.0)
}

lgbm_bounds = {
    'n_estimators': (100, 800),
    'max_depth': (3, 15),
    'num_leaves': (7, 150),
    'learning_rate': (0.001, 0.3),
    'min_child_samples': (5, 100),
    'subsample': (0.5, 1.0),
    'reg_alpha': (1e-4, 10.0),
    'reg_lambda': (1e-4, 10.0)
}

cat_bounds = {
    'iterations': (100, 800),
    'depth': (2, 12),
    'learning_rate': (0.001, 0.3),
    'l2_leaf_reg': (1.0, 15.0),
    'subsample': (0.5, 1.0)
}

result_aggressive = run_full_pipeline_optuna(
    X, y, numeric_cols, dataset_name="Telco Customer Churn (Aggressive Tuning)", 
    n_trials=100,  # More trials for better search
    rf_bounds=rf_bounds, xgb_bounds=xgb_bounds, lgbm_bounds=lgbm_bounds, cat_bounds=cat_bounds
)
save_model_files(result_aggressive, prefix='ds1agg')

joblib.dump(result_aggressive['final_metrics'], 'ds1agg_final_metrics.pkl')
joblib.dump(result_aggressive['overfit_df'], 'ds1agg_overfit_df.pkl')
joblib.dump(result_aggressive['efficiency_df'], 'ds1agg_efficiency_df.pkl')

print("\n\nAggressive tuning pipeline complete.")
