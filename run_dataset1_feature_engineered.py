"""
run_dataset1_feature_engineered.py — Same Telco dataset and same 3-model Optuna pipeline
as your primary result, but with engineered features added. Saves under 'ds1fe_' prefix
so it doesn't overwrite your existing primary 'ds1opt_' result — run this to COMPARE,
not replace, until you've confirmed it's actually better.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from pipeline import (
    split_and_scale, tune_random_forest_optuna, tune_xgboost_optuna, tune_lightgbm_optuna,
    build_final_models_from_optuna_3, tune_threshold, evaluate,
    overfitting_check, efficiency_check, save_model_files
)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df.drop('customerID', axis=1, inplace=True)

# =========================================================
# FEATURE ENGINEERING — new columns, built BEFORE encoding
# =========================================================

# 1. Average monthly spend across the customer's whole tenure (+1 avoids divide-by-zero
#    for brand-new customers). Compare this to current MonthlyCharges: a customer paying
#    much more NOW than their historical average may be a recent price-hike churn risk.
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlySpend']

# 2. Service engagement count — how many optional add-on services (out of 6) the
#    customer has. Low engagement customers may be easier to lose since they have
#    less invested in the platform.
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['NumServices'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)

# 3. Tenure bucket — captures non-linear tenure effects (e.g., a sharp risk cliff in
#    the first few months) that a single continuous tenure feature might smooth over.
df['TenureGroup'] = pd.cut(
    df['tenure'], bins=[-1, 6, 12, 24, 48, 72],
    labels=['0-6mo', '7-12mo', '13-24mo', '25-48mo', '49-72mo']
)

# 4. Explicit interaction flag: combines two features SHAP already identified as top
#    individual risk drivers (month-to-month contract + electronic check payment).
#    A model CAN learn interactions on its own from trees, but giving it directly
#    can help, especially for the linear meta-learner in the stack.
df['HighRiskCombo'] = (
    (df['Contract'] == 'Month-to-month') & (df['PaymentMethod'] == 'Electronic check')
).astype(int)

# 5. New customer flag — very low tenure customers behave differently (still in an
#    onboarding/trial mindset) than the continuous tenure value alone captures.
df['IsNewCustomer'] = (df['tenure'] <= 3).astype(int)

print("New engineered features added:", ['AvgMonthlySpend', 'ChargeIncrease', 'NumServices',
                                          'TenureGroup', 'HighRiskCombo', 'IsNewCustomer'])

# =========================================================
# Standard encoding (same as before, plus the new columns)
# =========================================================
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod', 'TenureGroup']
df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# All continuous numeric columns, including the new engineered ones, get scaled
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend',
                 'ChargeIncrease', 'NumServices']

print(f"\nFinal feature count: {X.shape[1]} (was 30 without feature engineering)")

# =========================================================
# Run the SAME pipeline as your primary result (3-model Optuna stack)
# =========================================================
print("\nTuning and training with feature-engineered data...")

X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, numeric_cols)
print(f"Train: {X_train.shape}, churn rate {y_train.mean():.4f} | Test: {X_test.shape}, churn rate {y_test.mean():.4f}")

print("\nTuning Random Forest (Optuna)...")
rf_study = tune_random_forest_optuna(X_train, y_train, n_trials=40)
print("Best RF CV F1:", rf_study.best_value)

print("\nTuning XGBoost (Optuna)...")
xgb_study = tune_xgboost_optuna(X_train, y_train, n_trials=40)
print("Best XGBoost CV F1:", xgb_study.best_value)

print("\nTuning LightGBM (Optuna)...")
lgbm_study = tune_lightgbm_optuna(X_train, y_train, n_trials=40)
print("Best LightGBM CV F1:", lgbm_study.best_value)

rf_final, xgb_final, lgbm_final = build_final_models_from_optuna_3(rf_study, xgb_study, lgbm_study)

X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

stack_model = StackingClassifier(
    estimators=[('rf', rf_final), ('xgb', xgb_final), ('lgbm', lgbm_final)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, n_jobs=-1
)
print("\nTraining stacked ensemble...")
stack_model.fit(X_train_bal, y_train_bal)

best_threshold, probs = tune_threshold(stack_model, X_test, y_test)
preds_tuned = (probs >= best_threshold).astype(int)
final_metrics = evaluate(y_test, preds_tuned)
print(f"\nBest threshold: {best_threshold:.4f}")
print("Final metrics (feature-engineered):", final_metrics)

xgb_final.fit(X_train_bal, y_train_bal)
rf_final.fit(X_train_bal, y_train_bal)
lgbm_final.fit(X_train_bal, y_train_bal)

models_dict = {'Random Forest': rf_final, 'XGBoost': xgb_final,
                'LightGBM': lgbm_final, 'Stacked Ensemble': stack_model}

print("\n=== Overfitting Check ===")
overfit_df = overfitting_check(models_dict, X_train, y_train, X_test, y_test)
print(overfit_df.to_string(index=False))

# --- Feature importance check: did the new engineered features actually matter? ---
print("\n=== XGBoost Feature Importance (top 15) — checking if engineered features rank highly ===")
importances = pd.Series(xgb_final.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(15))

result = {
    'scaler': scaler, 'xgb_final': xgb_final, 'stack_model': stack_model,
    'best_threshold': best_threshold, 'feature_columns': X_train.columns.tolist()
}
save_model_files(result, prefix='ds1fe')

joblib.dump(final_metrics, 'ds1fe_final_metrics.pkl')
joblib.dump(overfit_df, 'ds1fe_overfit_df.pkl')

print("\n\nFeature-engineered pipeline complete. Compare ds1fe_final_metrics.pkl against ds1opt_final_metrics.pkl.")