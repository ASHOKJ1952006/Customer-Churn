"""
run_dataset2_optuna.py — High-Value Customer Churn, tuned with Optuna instead of RandomizedSearchCV.
Same data preparation as run_dataset2.py (high-value filter, churn derivation, leakage prevention,
feature selection to top 40), but using Optuna's smarter TPE search for tuning.
Saves files with prefix 'ds2opt_'.
"""

import pandas as pd
import numpy as np
import joblib
from pipeline import run_full_pipeline_optuna, save_model_files

df = pd.read_csv("telecom_churn_data.csv")
print("Raw shape:", df.shape)

# --- High-value customer filtering ---
total_data_rech_6 = df['total_rech_data_6'].fillna(0) * df['av_rech_amt_data_6'].fillna(0)
total_data_rech_7 = df['total_rech_data_7'].fillna(0) * df['av_rech_amt_data_7'].fillna(0)
amt_data_6 = df['total_rech_amt_6'].fillna(0) + total_data_rech_6
amt_data_7 = df['total_rech_amt_7'].fillna(0) + total_data_rech_7

df = df.copy()
df['av_amt_data_6_7'] = (amt_data_6 + amt_data_7) / 2

threshold_70 = df['av_amt_data_6_7'].quantile(0.7)
df_hv = df[df['av_amt_data_6_7'] >= threshold_70].copy()
print(f"High-value customers: {df_hv.shape[0]} out of {df.shape[0]} total")

# --- Churn label derivation ---
df_hv['total_calls_9'] = df_hv['total_ic_mou_9'].fillna(0) + df_hv['total_og_mou_9'].fillna(0)
df_hv['total_data_9'] = df_hv['vol_2g_mb_9'].fillna(0) + df_hv['vol_3g_mb_9'].fillna(0)
df_hv['Churn'] = ((df_hv['total_calls_9'] == 0) & (df_hv['total_data_9'] == 0)).astype(int)

print("\nChurn distribution (high-value customers only):")
print(df_hv['Churn'].value_counts(normalize=True) * 100)

# --- Drop churn-phase and non-predictive columns ---
churn_phase_cols = [c for c in df_hv.columns if c.endswith('_9')]
drop_cols = churn_phase_cols + [
    'mobile_number', 'circle_id',
    'loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou',
    'last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8',
    'date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8',
    'date_of_last_rech_data_6', 'date_of_last_rech_data_7', 'date_of_last_rech_data_8',
    'total_calls_9', 'total_data_9',
]
drop_cols = [c for c in drop_cols if c in df_hv.columns]
df_clean = df_hv.drop(columns=drop_cols)

# --- Handle missing values ---
flag_cols = [c for c in df_clean.columns if 'fb_user' in c or 'night_pck_user' in c]
for c in flag_cols:
    df_clean[c] = df_clean[c].fillna(0)
numeric_all = df_clean.select_dtypes(include=[np.number]).columns.tolist()
df_clean[numeric_all] = df_clean[numeric_all].fillna(0)

y = df_clean['Churn']
X = df_clean.drop(columns=['Churn'])
print(f"\nFeature count (before selection): {X.shape[1]}")

# --- Feature selection: top 40 by quick XGBoost importance ---
from xgboost import XGBClassifier as _XGBQuick
from sklearn.model_selection import train_test_split as _tts

X_fs_train, _, y_fs_train, _ = _tts(X, y, test_size=0.2, random_state=42, stratify=y)
quick_model = _XGBQuick(random_state=42, eval_metric='logloss', tree_method='hist', device='cuda')
quick_model.fit(X_fs_train, y_fs_train)

importances = pd.Series(quick_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = importances.head(40).index.tolist()
print("\nTop 10 features by importance:")
print(importances.head(10))

X = X[top_features]
numeric_cols = top_features  # all remaining features are numeric usage/recharge figures
print(f"\nFeature count after selection: {X.shape[1]}")

# --- Run Optuna-tuned pipeline, with TIGHTENED search bounds ---
# Same reasoning as the earlier RandomizedSearchCV fix: 159->40 features on only ~1,950
# positive training examples needs constrained trees, not the wide default Optuna ranges.
rf_bounds = {
    'n_estimators': (100, 300),
    'max_depth': (3, 6),                # was (3, 25) — this is what caused the overfitting
    'min_samples_split': (10, 30),
    'min_samples_leaf': (5, 20),
}
xgb_bounds = {
    'n_estimators': (100, 200),
    'max_depth': (2, 4),                # was (2, 10)
    'learning_rate': (0.01, 0.05),
    'subsample': (0.6, 0.8),
    'colsample_bytree': (0.5, 0.7),
    'min_child_weight': (5, 15),
    'reg_alpha': (0.5, 3.0),            # stronger L1 floor than default
    'reg_lambda': (0.5, 3.0),
}
lgbm_bounds = {
    'n_estimators': (100, 200),
    'max_depth': (3, 5),                # was (3, 10)
    'num_leaves': (7, 20),              # was (7, 80)
    'learning_rate': (0.01, 0.05),
    'min_child_samples': (30, 70),
    'subsample': (0.6, 0.8),
    'reg_alpha': (0.5, 3.0),
    'reg_lambda': (0.5, 3.0),
}

result2_optuna = run_full_pipeline_optuna(
    X, y, numeric_cols, dataset_name="High-Value Customer Churn (Dataset 2, Optuna, regularized)",
    n_trials=25, rf_bounds=rf_bounds, xgb_bounds=xgb_bounds, lgbm_bounds=lgbm_bounds
)
save_model_files(result2_optuna, prefix='ds2opt')

joblib.dump(result2_optuna['final_metrics'], 'ds2opt_final_metrics.pkl')
joblib.dump(result2_optuna['overfit_df'], 'ds2opt_overfit_df.pkl')
joblib.dump(result2_optuna['efficiency_df'], 'ds2opt_efficiency_df.pkl')

print("\n\nDataset 2 (Optuna) pipeline complete.")
print("Compare against ds2_final_metrics.pkl (RandomizedSearchCV) to see if Optuna improved results.")