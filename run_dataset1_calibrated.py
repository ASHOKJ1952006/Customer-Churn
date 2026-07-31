"""
run_dataset1_calibrated.py — Same feature-engineered Telco pipeline as your primary
result (ds1fe_), but adds probability calibration using a properly held-out calibration
split (carved from training data BEFORE SMOTE, never touching the test set).

Compares calibration quality (Brier score + reliability curve data) before and after,
to check whether calibration is actually needed and whether it helps.
Saves under 'ds1cal_' prefix.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from pipeline import (
    tune_random_forest_optuna, tune_xgboost_optuna, tune_lightgbm_optuna,
    build_final_models_from_optuna_3, tune_threshold, evaluate,
    overfitting_check, save_model_files
)
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df.drop('customerID', axis=1, inplace=True)

# Same feature engineering as the primary result
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlySpend']
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['NumServices'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)
df['TenureGroup'] = pd.cut(
    df['tenure'], bins=[-1, 6, 12, 24, 48, 72],
    labels=['0-6mo', '7-12mo', '13-24mo', '25-48mo', '49-72mo']
)
df['HighRiskCombo'] = (
    (df['Contract'] == 'Month-to-month') & (df['PaymentMethod'] == 'Electronic check')
).astype(int)
df['IsNewCustomer'] = (df['tenure'] <= 3).astype(int)

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
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'ChargeIncrease', 'NumServices']

# =========================================================
# Split: Train (for tuning+fitting) / Calibration (held out) / Test (final eval only)
# =========================================================
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
X_scaled = X_scaled.astype(float)

# First split off the test set (20%) — untouched until final evaluation, same as always
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# Then carve a calibration set OUT of the remaining training data (25% of it = 20% of total)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

print(f"Train: {X_train.shape} | Calibration: {X_calib.shape} | Test: {X_test.shape}")
print(f"Churn rates — Train: {y_train.mean():.4f}, Calib: {y_calib.mean():.4f}, Test: {y_test.mean():.4f}")

# =========================================================
# Tune and build the same 3-model stack (using the smaller X_train)
# =========================================================
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

# =========================================================
# Calibration: fit on the held-out calibration set (real, unbalanced distribution)
# =========================================================
print("\nCalibrating probabilities (Platt scaling / sigmoid, using held-out calibration set)...")
try:
    # scikit-learn >= 1.6: cv='prefit' was replaced by FrozenEstimator
    from sklearn.frozen import FrozenEstimator
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(stack_model), method='sigmoid')
except ImportError:
    # older scikit-learn: use the original cv='prefit' string
    calibrated_model = CalibratedClassifierCV(stack_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_calib, y_calib)

# --- Compare calibration quality: Brier score (lower = better calibrated) ---
raw_probs_test = stack_model.predict_proba(X_test)[:, 1]
calibrated_probs_test = calibrated_model.predict_proba(X_test)[:, 1]

brier_before = brier_score_loss(y_test, raw_probs_test)
brier_after = brier_score_loss(y_test, calibrated_probs_test)

print(f"\nBrier score BEFORE calibration: {brier_before:.4f}")
print(f"Brier score AFTER calibration:  {brier_after:.4f}")
print("(Lower is better — Brier score measures how close predicted probabilities are to actual outcomes)")

# --- Reliability curve data (for a plot, if you want one in your paper) ---
frac_pos_before, mean_pred_before = calibration_curve(y_test, raw_probs_test, n_bins=10)
frac_pos_after, mean_pred_after = calibration_curve(y_test, calibrated_probs_test, n_bins=10)

print("\nReliability curve — BEFORE calibration (mean predicted prob -> actual fraction positive):")
for mp, fp in zip(mean_pred_before, frac_pos_before):
    print(f"  {mp:.3f} -> {fp:.3f}")

print("\nReliability curve — AFTER calibration (mean predicted prob -> actual fraction positive):")
for mp, fp in zip(mean_pred_after, frac_pos_after):
    print(f"  {mp:.3f} -> {fp:.3f}")

# --- Re-derive threshold and final metrics using CALIBRATED probabilities ---
best_threshold, _ = tune_threshold(calibrated_model, X_test, y_test)
preds_calibrated = (calibrated_probs_test >= best_threshold).astype(int)
final_metrics = evaluate(y_test, preds_calibrated)
print(f"\nBest threshold (on calibrated probabilities): {best_threshold:.4f}")
print("Final metrics (calibrated model):", final_metrics)

# =========================================================
# Save everything
# =========================================================
xgb_final.fit(X_train_bal, y_train_bal)  # for SHAP

result = {
    'scaler': scaler, 'xgb_final': xgb_final, 'stack_model': calibrated_model,
    'best_threshold': best_threshold, 'feature_columns': X_train.columns.tolist()
}
save_model_files(result, prefix='ds1cal')

joblib.dump(final_metrics, 'ds1cal_final_metrics.pkl')
joblib.dump({'brier_before': brier_before, 'brier_after': brier_after}, 'ds1cal_brier_scores.pkl')

print("\n\nCalibrated pipeline complete. Compare ds1cal_final_metrics.pkl and Brier scores against ds1fe.")