"""
run_dataset2_feature_selected.py — Dataset 2 with feature selection to fix overfitting
Applies top-N feature selection by importance, then trains with regularization.
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
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

print("=== Dataset 2: Feature Selection + Regularization (Fix Overfitting) ===")
print("Target: Reduce overfitting gap from 0.1082 to < 0.10\n")

# =========================================================
# Load and preprocess Dataset 2 (same as before)
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

# Churn label derivation
df_hv['total_calls_9'] = df_hv['total_ic_mou_9'].fillna(0) + df_hv['total_og_mou_9'].fillna(0)
df_hv['total_data_9'] = df_hv['vol_2g_mb_9'].fillna(0) + df_hv['vol_3g_mb_9'].fillna(0)
df_hv['Churn'] = ((df_hv['total_calls_9'] == 0) & (df_hv['total_data_9'] == 0)).astype(int)

# Drop churn phase columns
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
# Behavioral Feature Engineering (same as before)
# =========================================================
print("\n=== Adding Behavioral Change Features ===")

# 1. Call usage drop
if 'total_ic_mou_7' in df_clean.columns and 'total_ic_mou_8' in df_clean.columns:
    df_clean['CallUsageDrop_7to8'] = (df_clean['total_ic_mou_7'] - df_clean['total_ic_mou_8']) / (df_clean['total_ic_mou_7'] + 1)
    df_clean['CallUsageDrop_7to8'] = df_clean['CallUsageDrop_7to8'].clip(-1, 1)

# 2. Data usage drop
if 'vol_2g_mb_7' in df_clean.columns and 'vol_2g_mb_8' in df_clean.columns:
    df_clean['DataUsageDrop_7to8'] = (df_clean['vol_2g_mb_7'] - df_clean['vol_2g_mb_8']) / (df_clean['vol_2g_mb_7'] + 1)
    df_clean['DataUsageDrop_7to8'] = df_clean['DataUsageDrop_7to8'].clip(-1, 1)

# 3. Recharge consistency
rech_cols = [c for c in df_clean.columns if 'total_rech_amt' in c and not c.endswith('_9')]
if len(rech_cols) >= 3:
    df_clean['RechargeConsistency'] = df_clean[rech_cols].std(axis=1) / (df_clean[rech_cols].mean(axis=1) + 1)

# 4. WentQuiet flag
if 'total_ic_mou_8' in df_clean.columns and 'vol_2g_mb_8' in df_clean.columns:
    df_clean['WentQuiet'] = ((df_clean['total_ic_mou_8'] == 0) & (df_clean['vol_2g_mb_8'] == 0)).astype(int)

# 5. General trend features
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
print(f"Total features: {df_clean.shape[1] - 2}")  # -2 for Churn and av_amt_data_6_7

# =========================================================
# Build X, y
# =========================================================
y = df_clean['Churn']
X = df_clean.drop(columns=['Churn', 'av_amt_data_6_7'])
X = X.astype(np.float32)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Features before selection: {X.shape[1]}")

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
# Feature Selection using XGBoost importance
# =========================================================
print("\n=== Feature Selection ===")

# Train baseline XGBoost for feature importance
print("Training baseline XGBoost for feature importance...")
baseline_xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, eval_metric='logloss',
    tree_method='hist', device='cuda', n_jobs=-1
)

# Apply SMOTE for baseline training
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
baseline_xgb.fit(X_train_bal, y_train_bal)

# Get feature importance
importance = baseline_xgb.feature_importances_
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

print(f"\nTop 20 features by importance:")
print(feature_importance.head(20))

# Select top N features
top_n = 100  # Start with 100, can adjust
selected_features = feature_importance.head(top_n)['feature'].tolist()
print(f"\nSelected top {top_n} features")

# Create feature selector
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print(f"Train shape after selection: {X_train_selected.shape}")
print(f"Test shape after selection: {X_test_selected.shape}")

# =========================================================
# Train Stacked Ensemble with Regularization on Selected Features
# =========================================================
print("\n=== Training Stacked Ensemble (Regularized, Selected Features) ===")

# XGBoost (GPU, regularized)
print("Training XGBoost (NVIDIA GPU, regularized)...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, eval_metric='logloss',
    tree_method='hist', device='cuda', n_jobs=-1
)

# LightGBM (CPU, regularized)
print("Training LightGBM (CPU, regularized)...")
lgbm = LGBMClassifier(
    n_estimators=300, max_depth=6, num_leaves=30,
    learning_rate=0.03, min_child_samples=30, subsample=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
    device='cpu'
)

# Random Forest (CPU, regularized)
print("Training Random Forest (CPU, regularized)...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_split=20,
    min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
)

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
pipeline.fit(X_train_selected, y_train)

# =========================================================
# CV-based Calibration
# =========================================================
print("\n=== CV-based Calibration ===")

try:
    from sklearn.frozen import FrozenEstimator
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(pipeline),
        cv=5, method='sigmoid', n_jobs=-1
    )
except ImportError:
    calibrated = CalibratedClassifierCV(
        pipeline,
        cv='prefit', method='sigmoid'
    )

print("Calibrating...")
calibrated.fit(X_train_selected, y_train)

# =========================================================
# Evaluation
# =========================================================
print("\n=== Evaluation (Original Unbalanced Test Set) ===")

# Uncalibrated predictions
probs_uncal = pipeline.predict_proba(X_test_selected)[:, 1]
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
probs_cal = calibrated.predict_proba(X_test_selected)[:, 1]
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
print(f"\n=== COMPARISON WITH PREVIOUS (315 features, gap 0.1082) ===")
print(f"Previous Recall: 0.6865 | Current Recall: {rec_cal:.4f} | Change: {rec_cal - 0.6865:+.4f}")
print(f"Previous F1: 0.625 | Current F1: {f1_cal:.4f} | Change: {f1_cal - 0.625:+.4f}")

# Overfitting check
train_probs = calibrated.predict_proba(X_train_selected)[:, 1]
train_preds = (train_probs >= threshold_cal).astype(int)
train_f1 = f1_score(y_train, train_preds)
test_f1 = f1_score(y_test, preds_cal)
overfit_gap = train_f1 - test_f1

print(f"\n=== OVERFITTING CHECK ===")
print(f"Train F1: {train_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Gap: {overfit_gap:.4f} (target: < 0.10)")

if overfit_gap < 0.10:
    print("✅ OVERFITTING FIXED: Gap below 0.10 threshold")
else:
    print(f"⚠️ OVERFITTING STILL HIGH: Gap {overfit_gap:.4f} > 0.10")
    print("Consider reducing top_n further or increasing regularization")

# =========================================================
# Save model artifacts
# =========================================================
print(f"\n=== Saving Model Artifacts ===")

joblib.dump(calibrated, 'ds2fs_selected_churn_model.pkl')
joblib.dump(scaler, 'ds2fs_selected_scaler.pkl')
joblib.dump(selected_features, 'ds2fs_selected_feature_columns.pkl')
joblib.dump(threshold_cal, 'ds2fs_selected_best_threshold.pkl')

metrics = {
    'Accuracy': acc_cal,
    'Precision': prec_cal,
    'Recall': rec_cal,
    'F1': f1_cal,
    'Brier_Score': brier_cal,
    'Overfit_Gap': overfit_gap,
    'Num_Features': len(selected_features)
}
joblib.dump(metrics, 'ds2fs_selected_final_metrics.pkl')

# Save XGBoost component for SHAP
baseline_xgb.save_model('ds2fs_selected_shap_model.json')

print("Dataset 2 Feature Selection + Regularization complete.")
