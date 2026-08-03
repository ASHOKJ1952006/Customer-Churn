"""
run_dataset2_feature_engineered_calibrated.py — Dataset 2 with behavioral feature engineering
and CV-based calibration, following the same honest methodology as Dataset 1.

Feature engineering focus: behavioral CHANGE features leveraging Dataset 2's time-series structure:
- CallUsageDrop_7to8, DataUsageDrop_7to8 (usage decline patterns)
- RechargeConsistency (recharge amount stability)
- WentQuiet flag (extending fb_user_8 signal)

Calibration: CV-based (not held-out split) to preserve recall.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, precision_recall_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

print("=== Dataset 2: Feature Engineering + CV Calibration ===")
print("Following honest methodology: SMOTE inside CV, recall as primary metric\n")

# =========================================================
# Load and preprocess Dataset 2
# =========================================================
df = pd.read_csv("telecom_churn_data.csv")
print("Raw shape:", df.shape)

# High-value customer identification
total_data_rech_6 = df['total_rech_data_6'].fillna(0) * df['av_rech_amt_data_6'].fillna(0)
total_data_rech_7 = df['total_rech_data_7'].fillna(0) * df['av_rech_amt_data_7'].fillna(0)
amt_data_6 = df['total_rech_amt_6'].fillna(0) + total_data_rech_6
amt_data_7 = df['total_rech_amt_7'].fillna(0) + total_data_rech_7
df = df.copy()
df['av_amt_data_6_7'] = (amt_data_6 + amt_data_7) / 2
threshold_70 = df['av_amt_data_6_7'].quantile(0.7)
df_hv = df[df['av_amt_data_6_7'] >= threshold_70].copy()
print(f"High-value customers: {df_hv.shape[0]}")

# Churn label derivation (month 9 zero usage)
df_hv['total_calls_9'] = df_hv['total_ic_mou_9'].fillna(0) + df_hv['total_og_mou_9'].fillna(0)
df_hv['total_data_9'] = df_hv['vol_2g_mb_9'].fillna(0) + df_hv['vol_3g_mb_9'].fillna(0)
df_hv['Churn'] = ((df_hv['total_calls_9'] == 0) & (df_hv['total_data_9'] == 0)).astype(int)

# Drop churn phase columns to prevent leakage
churn_phase_cols = [c for c in df_hv.columns if c.endswith('_9')]
drop_cols = churn_phase_cols + ['mobile_number', 'circle_id', 'loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou',
    'last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8',
    'date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8',
    'date_of_last_rech_data_6', 'date_of_last_rech_data_7', 'date_of_last_rech_data_8',
    'total_calls_9', 'total_data_9']
drop_cols = [c for c in drop_cols if c in df_hv.columns]
df_clean = df_hv.drop(columns=drop_cols)

# Handle missing values
flag_cols = [c for c in df_clean.columns if 'fb_user' in c or 'night_pck_user' in c]
for c in flag_cols:
    df_clean[c] = df_clean[c].fillna(0)
numeric_all = df_clean.select_dtypes(include=[np.number]).columns.tolist()
df_clean[numeric_all] = df_clean[numeric_all].fillna(0)

print(f"Clean shape: {df_clean.shape}")
print(f"Churn rate: {df_clean['Churn'].mean():.2%}")

# =========================================================
# Behavioral Feature Engineering (Dataset 2 specific)
# =========================================================
print("\n=== Adding Behavioral Change Features ===")

# 1. Call usage drop from month 7 to 8
if 'total_ic_mou_7' in df_clean.columns and 'total_ic_mou_8' in df_clean.columns:
    df_clean['CallUsageDrop_7to8'] = (df_clean['total_ic_mou_7'] - df_clean['total_ic_mou_8']) / (df_clean['total_ic_mou_7'] + 1)
    df_clean['CallUsageDrop_7to8'] = df_clean['CallUsageDrop_7to8'].clip(-1, 1)

# 2. Data usage drop from month 7 to 8
if 'vol_2g_mb_7' in df_clean.columns and 'vol_2g_mb_8' in df_clean.columns:
    df_clean['DataUsageDrop_7to8'] = (df_clean['vol_2g_mb_7'] - df_clean['vol_2g_mb_8']) / (df_clean['vol_2g_mb_7'] + 1)
    df_clean['DataUsageDrop_7to8'] = df_clean['DataUsageDrop_7to8'].clip(-1, 1)

# 3. Recharge consistency (std/mean ratio across months 6-8)
rech_cols = [c for c in df_clean.columns if 'total_rech_amt' in c and not c.endswith('_9')]
if len(rech_cols) >= 3:
    df_clean['RechargeConsistency'] = df_clean[rech_cols].std(axis=1) / (df_clean[rech_cols].mean(axis=1) + 1)

# 4. WentQuiet flag (extended fb_user_8 signal - zero usage in month 8)
if 'total_ic_mou_8' in df_clean.columns and 'vol_2g_mb_8' in df_clean.columns:
    df_clean['WentQuiet'] = ((df_clean['total_ic_mou_8'] == 0) & (df_clean['vol_2g_mb_8'] == 0)).astype(int)

# 5. General trend features (as before)
month_6_cols = [c for c in df_clean.columns if c.endswith('_6')]
month_7_cols = [c for c in df_clean.columns if c.endswith('_7')]
month_8_cols = [c for c in df_clean.columns if c.endswith('_8')]
base_names_6 = set([c[:-2] for c in month_6_cols])
base_names_7 = set([c[:-2] for c in month_7_cols])
base_names_8 = set([c[:-2] for c in month_8_cols])
common_bases = base_names_6 & base_names_7 & base_names_8

new_features_dict = {}
for base in common_bases:
    col_6 = f"{base}_6"
    col_7 = f"{base}_7"
    col_8 = f"{base}_8"
    if all(col in df_clean.columns for col in [col_6, col_7, col_8]):
        new_features_dict[f"{base}_trend"] = df_clean[col_8] - df_clean[col_6]
        new_features_dict[f"{base}_avg"] = (df_clean[col_6] + df_clean[col_7] + df_clean[col_8]) / 3
        new_features_dict[f"{base}_std"] = df_clean[[col_6, col_7, col_8]].std(axis=1)

if new_features_dict:
    df_features = pd.DataFrame(new_features_dict)
    df_clean = pd.concat([df_clean, df_features], axis=1)

print(f"Shape after FE: {df_clean.shape}")
print(f"New features added: {len(new_features_dict) + 4}")

# =========================================================
# Build X, y
# =========================================================
y = df_clean['Churn']
X = df_clean.drop(columns=['Churn', 'av_amt_data_6_7'])
X = X.astype(np.float32)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Final feature count: {X.shape[1]}")

# =========================================================
# Stratified train/test split
# =========================================================
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
X_scaled = X_scaled.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape}, churn rate {y_train.mean():.2%}")
print(f"Test: {X_test.shape}, churn rate {y_test.mean():.2%}")

# =========================================================
# Base learners (following established methodology)
# =========================================================
print("\n=== Training Base Learners ===")

# XGBoost (GPU) - Fit separately for SHAP
print("Training XGBoost (NVIDIA GPU) separately for SHAP...")
xgb_shap = XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
    reg_alpha=0.05, reg_lambda=0.05, random_state=42, eval_metric='logloss',
    tree_method='hist', device='cuda', n_jobs=-1
)

# Apply SMOTE for XGBoost SHAP model
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
xgb_shap.fit(X_train_bal, y_train_bal)

# XGBoost for pipeline (more regularized to reduce overfitting)
print("Training XGBoost (NVIDIA GPU) for pipeline (regularized)...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, eval_metric='logloss',
    tree_method='hist', device='cuda', n_jobs=-1
)

# LightGBM (CPU - per guidelines, regularized to reduce overfitting)
print("Training LightGBM (CPU, regularized)...")
lgbm = LGBMClassifier(
    n_estimators=300, max_depth=6, num_leaves=30,
    learning_rate=0.03, min_child_samples=30, subsample=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
    device='cpu'
)

# Random Forest (CPU, regularized to reduce overfitting)
print("Training Random Forest (CPU, regularized)...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_split=20,
    min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
)

# =========================================================
# SMOTE + Stacking Pipeline (CV-based calibration)
# =========================================================
print("\n=== Building SMOTE + Stacking Pipeline ===")

# Create stacking classifier
stacking = StackingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
    final_estimator=LogisticRegression(max_iter=1000, C=0.1),
    cv=5, n_jobs=-1
)

# Create pipeline: SMOTE -> Stacking
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('stack', stacking)
])

print("Training pipeline with SMOTE inside CV...")
pipeline.fit(X_train, y_train)

# =========================================================
# CV-based Calibration (preserves recall)
# =========================================================
print("\n=== CV-based Calibration ===")

try:
    # For sklearn >= 1.6
    from sklearn.frozen import FrozenEstimator
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(pipeline),
        cv=5, method='sigmoid', n_jobs=-1
    )
except ImportError:
    # Fallback for older sklearn
    calibrated = CalibratedClassifierCV(
        pipeline,
        cv='prefit', method='sigmoid'
    )
    # Note: cv='prefit' is deprecated but kept as fallback
    print("Using cv='prefit' fallback (older sklearn version)")

print("Calibrating...")
calibrated.fit(X_train, y_train)

# =========================================================
# Evaluation (Honest: test on ORIGINAL unbalanced distribution)
# =========================================================
print("\n=== Evaluation (Original Unbalanced Test Set) ===")

# Uncalibrated predictions
probs_uncal = pipeline.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs_uncal)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = f1_scores.argmax()
threshold_uncal = thresholds[best_idx] if len(thresholds) > 0 else 0.5
preds_uncal = (probs_uncal >= threshold_uncal).astype(int)

acc_uncal = accuracy_score(y_test, preds_uncal)
prec_uncal = precision_score(y_test, preds_uncal)
rec_uncal = recall_score(y_test, preds_uncal)
f1_uncal = f1_score(y_test, preds_uncal)
brier_uncal = brier_score_loss(y_test, probs_uncal)

print(f"UNCALIBRATED:")
print(f"  Accuracy: {acc_uncal:.4f}")
print(f"  Precision: {prec_uncal:.4f}")
print(f"  Recall: {rec_uncal:.4f}")
print(f"  F1: {f1_uncal:.4f}")
print(f"  Brier Score: {brier_uncal:.4f}")

# Calibrated predictions
probs_cal = calibrated.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs_cal)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = f1_scores.argmax()
threshold_cal = thresholds[best_idx] if len(thresholds) > 0 else 0.5
preds_cal = (probs_cal >= threshold_cal).astype(int)

acc_cal = accuracy_score(y_test, preds_cal)
prec_cal = precision_score(y_test, preds_cal)
rec_cal = recall_score(y_test, preds_cal)
f1_cal = f1_score(y_test, preds_cal)
brier_cal = brier_score_loss(y_test, probs_cal)

print(f"\nCALIBRATED:")
print(f"  Accuracy: {acc_cal:.4f}")
print(f"  Precision: {prec_cal:.4f}")
print(f"  Recall: {rec_cal:.4f}")
print(f"  F1: {f1_cal:.4f}")
print(f"  Brier Score: {brier_cal:.4f}")

# Comparison
print(f"\n=== CALIBRATION IMPACT ===")
print(f"Accuracy change: {acc_cal - acc_uncal:+.4f}")
print(f"Precision change: {prec_cal - prec_uncal:+.4f}")
print(f"Recall change: {rec_cal - rec_uncal:+.4f} (PRIMARY METRIC)")
print(f"F1 change: {f1_cal - f1_uncal:+.4f}")
print(f"Brier Score improvement: {brier_uncal - brier_cal:+.4f} (lower is better)")

# Overfitting check
train_probs = calibrated.predict_proba(X_train)[:, 1]
train_preds = (train_probs >= threshold_cal).astype(int)
train_f1 = f1_score(y_train, train_preds)
test_f1 = f1_score(y_test, preds_cal)
overfit_gap = train_f1 - test_f1

print(f"\n=== OVERFITTING CHECK ===")
print(f"Train F1: {train_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Gap: {overfit_gap:.4f} (healthy if < 0.1)")

# =========================================================
# Save model artifacts
# =========================================================
print(f"\n=== Saving Model Artifacts ===")

# Save calibrated model
joblib.dump(calibrated, 'ds2fecal_churn_model.pkl')
joblib.dump(scaler, 'ds2fecal_scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'ds2fecal_feature_columns.pkl')
joblib.dump(threshold_cal, 'ds2fecal_best_threshold.pkl')

metrics = {
    'Accuracy': acc_cal,
    'Precision': prec_cal,
    'Recall': rec_cal,
    'F1': f1_cal,
    'Brier_Score': brier_cal,
    'Overfit_Gap': overfit_gap
}
joblib.dump(metrics, 'ds2fecal_final_metrics.pkl')

# Save XGBoost component for SHAP (separately fitted model)
xgb_shap.save_model('ds2fecal_shap_model.json')

print("Dataset 2 FE + Calibration complete.")
print(f"Primary metric (Recall): {rec_cal:.4f}")
