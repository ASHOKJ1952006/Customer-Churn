"""
run_dataset2.py — High-Value Customer Churn (Indian telecom, 4-month usage data)

This dataset has NO explicit churn column. Per the dataset's own methodology:
  - Months 6, 7 = "good phase" (used to identify high-value customers)
  - Month 8     = "action phase"
  - Month 9     = "churn phase" (used ONLY to derive the churn label, then dropped)

High-value customers = top 30% by average recharge spend in months 6-7
(the dataset description cites this "top 20% of customers = 80% of revenue" framing).

Churn definition = zero calls (total_ic_mou_9 + total_og_mou_9 == 0)
                    AND zero data usage (vol_2g_mb_9 + vol_3g_mb_9 == 0) in month 9.

CRITICAL: every other month-9 (*_9) column is dropped after deriving the label —
otherwise the model would be trained on information that defines the target itself.
"""

import pandas as pd
import numpy as np
from pipeline import run_full_pipeline, save_model_files

df = pd.read_csv("telecom_churn_data.csv")
print("Raw shape:", df.shape)

# =========================================================
# STEP 1: Identify high-value customers (top 30% by avg recharge, months 6-7)
# =========================================================
# Total recharge = call recharge + data recharge (data recharge = count * avg amount)
total_data_rech_6 = df['total_rech_data_6'].fillna(0) * df['av_rech_amt_data_6'].fillna(0)
total_data_rech_7 = df['total_rech_data_7'].fillna(0) * df['av_rech_amt_data_7'].fillna(0)

amt_data_6 = df['total_rech_amt_6'].fillna(0) + total_data_rech_6
amt_data_7 = df['total_rech_amt_7'].fillna(0) + total_data_rech_7

df = df.copy()
df['av_amt_data_6_7'] = (amt_data_6 + amt_data_7) / 2

threshold_70 = df['av_amt_data_6_7'].quantile(0.7)
print(f"\n70th percentile recharge (months 6-7): {threshold_70:.2f}")

df_hv = df[df['av_amt_data_6_7'] >= threshold_70].copy()
print(f"High-value customers: {df_hv.shape[0]} out of {df.shape[0]} total")

# =========================================================
# STEP 2: Derive churn label from month 9 (churn phase) usage
# =========================================================
df_hv['total_calls_9'] = df_hv['total_ic_mou_9'].fillna(0) + df_hv['total_og_mou_9'].fillna(0)
df_hv['total_data_9'] = df_hv['vol_2g_mb_9'].fillna(0) + df_hv['vol_3g_mb_9'].fillna(0)

df_hv['Churn'] = ((df_hv['total_calls_9'] == 0) & (df_hv['total_data_9'] == 0)).astype(int)

print("\nChurn distribution (high-value customers only):")
print(df_hv['Churn'].value_counts())
print(df_hv['Churn'].value_counts(normalize=True) * 100)

# =========================================================
# STEP 3: Drop all month-9 columns (churn phase) to prevent leakage,
# plus identifier / constant / date columns that add no predictive value
# =========================================================
churn_phase_cols = [c for c in df_hv.columns if c.endswith('_9')]

drop_cols = churn_phase_cols + [
    'mobile_number', 'circle_id',
    'loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou',  # near-constant/zero in this dataset
    'last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8',
    'date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8',
    'date_of_last_rech_data_6', 'date_of_last_rech_data_7', 'date_of_last_rech_data_8',
    'total_calls_9', 'total_data_9',  # helper columns used only to derive Churn
]
drop_cols = [c for c in drop_cols if c in df_hv.columns]  # safety: only drop what exists

df_clean = df_hv.drop(columns=drop_cols)
print(f"\nShape after dropping churn-phase & non-predictive columns: {df_clean.shape}")

# =========================================================
# STEP 4: Handle remaining missing values and non-numeric columns
# =========================================================
# fb_user_* and night_pck_user_* are categorical flags (0/1/NaN) — fill NaN with 0 (treated as "no")
flag_cols = [c for c in df_clean.columns if 'fb_user' in c or 'night_pck_user' in c]
for c in flag_cols:
    df_clean[c] = df_clean[c].fillna(0)

# Remaining numeric NaNs (mostly usage columns where NaN means "no usage") -> fill with 0
numeric_all = df_clean.select_dtypes(include=[np.number]).columns.tolist()
df_clean[numeric_all] = df_clean[numeric_all].fillna(0)

print("\nRemaining missing values after cleanup:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# =========================================================
# STEP 5: Build X, y and run the shared pipeline
# =========================================================
y = df_clean['Churn']
X = df_clean.drop(columns=['Churn'])

# All remaining columns are numeric (usage/recharge figures) — scale everything
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nFinal feature count (before selection): {X.shape[1]}")

# =========================================================
# STEP 5b: Feature selection — reduce from 159 to top 40 features
# =========================================================
# With only ~1,950 positive training examples, 159 features is a lot of room
# to overfit. Rank features by a quick XGBoost fit's importance, keep the top 40.
from xgboost import XGBClassifier as _XGBQuick
from sklearn.model_selection import train_test_split as _tts

X_fs_train, _, y_fs_train, _ = _tts(X, y, test_size=0.2, random_state=42, stratify=y)
quick_model = _XGBQuick(random_state=42, eval_metric='logloss', tree_method='hist', device='cuda')
quick_model.fit(X_fs_train, y_fs_train)

importances = pd.Series(quick_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importances.head(40).index.tolist()

print("\nTop 15 features by importance:")
print(importances.head(15))

X = X[top_features]
numeric_cols = [c for c in numeric_cols if c in top_features]
print(f"\nFeature count after selection: {X.shape[1]}")

# =========================================================
# STEP 6: Stricter regularization grids (combat overfitting on
# a high-dimensional, severely imbalanced dataset)
# =========================================================
rf_strict_params = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 4, 5, 6],              # shallower than default (was up to 20/None)
    'clf__min_samples_split': [10, 20, 30],       # higher than default (was 2-10)
    'clf__min_samples_leaf': [5, 10, 15, 20],     # higher than default (was 1-4)
    'clf__max_features': ['sqrt', 'log2']
}

xgb_strict_params = {
    'clf__n_estimators': [100, 150, 200],
    'clf__max_depth': [2, 3, 4],                  # shallower than default (was up to 8)
    'clf__learning_rate': [0.01, 0.03, 0.05],
    'clf__subsample': [0.6, 0.7, 0.8],
    'clf__colsample_bytree': [0.5, 0.6, 0.7],
    'clf__min_child_weight': [5, 10, 15],         # higher than default (was 1-5)
    'clf__reg_alpha': [0.5, 1.0, 2.0],            # stronger L1 than dataset1
    'clf__reg_lambda': [0.5, 1.0, 2.0],           # stronger L2 than dataset1
}

lgbm_strict_params = {
    'clf__n_estimators': [100, 150, 200],
    'clf__max_depth': [3, 4, 5],
    'clf__num_leaves': [7, 15, 20],               # much smaller than default (was up to 70)
    'clf__learning_rate': [0.01, 0.03, 0.05],
    'clf__min_child_samples': [30, 50, 70],       # higher than default (was 10-50)
    'clf__subsample': [0.6, 0.7, 0.8],
    'clf__reg_alpha': [0.5, 1.0, 2.0],
    'clf__reg_lambda': [0.5, 1.0, 2.0],
}

result2 = run_full_pipeline(
    X, y, numeric_cols, dataset_name="High-Value Customer Churn (Dataset 2, feature-selected)",
    n_iter=20,
    rf_custom_params=rf_strict_params,
    xgb_custom_params=xgb_strict_params,
    lgbm_custom_params=lgbm_strict_params
)
save_model_files(result2, prefix='ds2')

import joblib
joblib.dump(result2['final_metrics'], 'ds2_final_metrics.pkl')
joblib.dump(result2['overfit_df'], 'ds2_overfit_df.pkl')
joblib.dump(result2['efficiency_df'], 'ds2_efficiency_df.pkl')

print("\n\nDataset 2 pipeline complete.")