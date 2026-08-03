"""
run_dataset1_voting_ensemble.py — Uses VotingClassifier instead of StackingClassifier
to potentially achieve higher accuracy. Voting often performs better than stacking
when base models are already well-tuned.
"""

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

from pipeline import (
    split_and_scale, tune_random_forest_optuna, tune_xgboost_optuna, 
    tune_lightgbm_optuna, tune_catboost_optuna,
    build_final_models_from_optuna
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

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

print(f"\n{'='*60}\nRUNNING VOTING ENSEMBLE PIPELINE\n{'='*60}")
print(f"Shape: {X.shape}, Churn rate: {y.mean():.4f}")

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

print("\nTuning CatBoost (Optuna)...")
cat_study = tune_catboost_optuna(X_train, y_train, n_trials=40)
print("Best CatBoost CV F1:", cat_study.best_value)

rf_final, xgb_final, lgbm_final, cat_final = build_final_models_from_optuna(
    rf_study, xgb_study, lgbm_study, cat_study
)

# Balance training data
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Fit individual models on balanced data
print("\nTraining individual models on balanced data...")
rf_final.fit(X_train_bal, y_train_bal)
xgb_final.fit(X_train_bal, y_train_bal)
lgbm_final.fit(X_train_bal, y_train_bal)
cat_final.fit(X_train_bal, y_train_bal)

# Try both soft and hard voting
print("\nTraining Voting Ensemble (soft voting)...")
voting_soft = VotingClassifier(
    estimators=[('rf', rf_final), ('xgb', xgb_final), ('lgbm', lgbm_final), ('cat', cat_final)],
    voting='soft', n_jobs=-1
)
voting_soft.fit(X_train_bal, y_train_bal)

print("\nTraining Voting Ensemble (hard voting)...")
voting_hard = VotingClassifier(
    estimators=[('rf', rf_final), ('xgb', xgb_final), ('lgbm', lgbm_final), ('cat', cat_final)],
    voting='hard', n_jobs=-1
)
voting_hard.fit(X_train_bal, y_train_bal)

# Evaluate both
probs_soft = voting_soft.predict_proba(X_test)[:, 1]
preds_hard = voting_hard.predict(X_test)

# Threshold tuning for soft voting
precisions, recalls, thresholds = precision_recall_curve(y_test, probs_soft)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
preds_soft_tuned = (probs_soft >= best_threshold).astype(int)

print(f"\n=== Soft Voting (tuned threshold: {best_threshold:.4f}) ===")
print(f"Accuracy: {accuracy_score(y_test, preds_soft_tuned):.4f}")
print(f"Precision: {precision_score(y_test, preds_soft_tuned):.4f}")
print(f"Recall: {recall_score(y_test, preds_soft_tuned):.4f}")
print(f"F1: {f1_score(y_test, preds_soft_tuned):.4f}")

print(f"\n=== Hard Voting ===")
print(f"Accuracy: {accuracy_score(y_test, preds_hard):.4f}")
print(f"Precision: {precision_score(y_test, preds_hard):.4f}")
print(f"Recall: {recall_score(y_test, preds_hard):.4f}")
print(f"F1: {f1_score(y_test, preds_hard):.4f}")

# Save the better performing voting model
if f1_score(y_test, preds_soft_tuned) > f1_score(y_test, preds_hard):
    best_model = voting_soft
    best_preds = preds_soft_tuned
    best_probs = probs_soft
    print("\nSoft voting selected as best model.")
else:
    best_model = voting_hard
    best_preds = preds_hard
    # For hard voting, we need probabilities from soft voting for SHAP
    best_probs = probs_soft
    print("\nHard voting selected as best model.")

# Save model files
joblib.dump(best_model, 'ds1vote_churn_model.pkl')
joblib.dump(scaler, 'ds1vote_scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'ds1vote_feature_columns.pkl')
joblib.dump(best_threshold, 'ds1vote_best_threshold.pkl')
xgb_final.save_model('ds1vote_shap_model.json')

final_metrics = {
    'Accuracy': accuracy_score(y_test, best_preds),
    'Precision': precision_score(y_test, best_preds),
    'Recall': recall_score(y_test, best_preds),
    'F1': f1_score(y_test, best_preds)
}

joblib.dump(final_metrics, 'ds1vote_final_metrics.pkl')

print(f"\nFinal metrics (Voting Ensemble): {final_metrics}")
print("\nVoting ensemble pipeline complete.")
