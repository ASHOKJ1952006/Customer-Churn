"""
run_dataset1_calibrated_cv.py — Same feature-engineered Telco pipeline and same full
training set as your primary result (ds1fe_), but adds probability calibration via
CalibratedClassifierCV's internal 3-fold CV instead of a separate held-out split —
avoiding the training-data loss (and resulting recall drop) from the earlier approach.

SMOTE is applied INSIDE the calibration pipeline (per fold), same leakage-safe pattern
used everywhere else in this project — not pre-applied to the whole training set.

Saves under 'ds1calcv_' prefix.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from pipeline import (
    split_and_scale, tune_random_forest_optuna, tune_xgboost_optuna, tune_lightgbm_optuna,
    build_final_models_from_optuna_3, tune_threshold, evaluate,
    overfitting_check, save_model_files
)

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

# Same 80/20 split as the primary result — NO calibration split carved out this time
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, numeric_cols)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Churn rates — Train: {y_train.mean():.4f}, Test: {y_test.mean():.4f}")

# =========================================================
# Tune (same as primary — full X_train, no data held back)
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

# =========================================================
# Build SMOTE + Stack pipeline, then wrap in CV-based calibration
# (SMOTE happens INSIDE each calibration fold — no pre-resampling, no leakage)
# =========================================================
stack_model = StackingClassifier(
    estimators=[('rf', rf_final), ('xgb', xgb_final), ('lgbm', lgbm_final)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3,  # reduced from 5 to keep the nested CV (3 outer x 3 inner) computationally reasonable
    n_jobs=-1
)

smote_stack_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('stack', stack_model)
])

print("\nCalibrating via 3-fold CV (SMOTE + stack refit per fold — this will take a while)...")
calibrated_model = CalibratedClassifierCV(smote_stack_pipeline, method='sigmoid', cv=3, n_jobs=-1)
calibrated_model.fit(X_train, y_train)

# =========================================================
# Compare calibration quality against the uncalibrated primary approach
# =========================================================
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
uncalibrated_stack = StackingClassifier(
    estimators=[('rf', rf_final), ('xgb', xgb_final), ('lgbm', lgbm_final)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, n_jobs=-1
)
uncalibrated_stack.fit(X_train_bal, y_train_bal)

raw_probs_test = uncalibrated_stack.predict_proba(X_test)[:, 1]
calibrated_probs_test = calibrated_model.predict_proba(X_test)[:, 1]

brier_before = brier_score_loss(y_test, raw_probs_test)
brier_after = brier_score_loss(y_test, calibrated_probs_test)
print(f"\nBrier score BEFORE calibration: {brier_before:.4f}")
print(f"Brier score AFTER calibration (CV-based):  {brier_after:.4f}")

frac_pos_before, mean_pred_before = calibration_curve(y_test, raw_probs_test, n_bins=10)
frac_pos_after, mean_pred_after = calibration_curve(y_test, calibrated_probs_test, n_bins=10)

print("\nReliability curve — BEFORE calibration:")
for mp, fp in zip(mean_pred_before, frac_pos_before):
    print(f"  {mp:.3f} -> {fp:.3f}")

print("\nReliability curve — AFTER calibration (CV-based):")
for mp, fp in zip(mean_pred_after, frac_pos_after):
    print(f"  {mp:.3f} -> {fp:.3f}")

# =========================================================
# Final metrics using calibrated probabilities
# =========================================================
best_threshold, _ = tune_threshold(calibrated_model, X_test, y_test)
preds_calibrated = (calibrated_probs_test >= best_threshold).astype(int)
final_metrics = evaluate(y_test, preds_calibrated)
print(f"\nBest threshold (on CV-calibrated probabilities): {best_threshold:.4f}")
print("Final metrics (CV-calibrated model):", final_metrics)
print("\nFor comparison, primary (ds1fe, uncalibrated) was: "
      "Accuracy 0.7466, Precision 0.5142, Recall 0.8209, F1 0.6323")

# =========================================================
# Save
# =========================================================
xgb_final.fit(X_train_bal, y_train_bal)  # separate model just for SHAP, same pattern as before

result = {
    'scaler': scaler, 'xgb_final': xgb_final, 'stack_model': calibrated_model,
    'best_threshold': best_threshold, 'feature_columns': X_train.columns.tolist()
}
save_model_files(result, prefix='ds1calcv')

joblib.dump(final_metrics, 'ds1calcv_final_metrics.pkl')
joblib.dump({'brier_before': brier_before, 'brier_after': brier_after}, 'ds1calcv_brier_scores.pkl')

print("\n\nCV-calibrated pipeline complete.")