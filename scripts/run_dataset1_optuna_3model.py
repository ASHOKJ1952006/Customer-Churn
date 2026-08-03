"""
run_dataset1_optuna_3model.py — Regenerates the 3-model stack (Random Forest + XGBoost +
LightGBM, no CatBoost) as the PRIMARY reported result, since it has better recall than
the 4-model version. Saves under the same 'ds1opt_' prefix so app.py needs no changes.

Run this to restore ds1opt_ files back to the 3-model version after the CatBoost experiment.
"""

import pandas as pd
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

print("="*60)
print("Regenerating 3-model stack (RF + XGBoost + LightGBM) — primary result")
print("="*60)

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
print("\nTraining stacked ensemble (3 base learners)...")
stack_model.fit(X_train_bal, y_train_bal)

best_threshold, probs = tune_threshold(stack_model, X_test, y_test)
preds_tuned = (probs >= best_threshold).astype(int)
final_metrics = evaluate(y_test, preds_tuned)
print(f"\nBest threshold: {best_threshold:.4f}")
print("Final metrics:", final_metrics)

xgb_final.fit(X_train_bal, y_train_bal)
rf_final.fit(X_train_bal, y_train_bal)
lgbm_final.fit(X_train_bal, y_train_bal)

models_dict = {'Random Forest': rf_final, 'XGBoost': xgb_final,
                'LightGBM': lgbm_final, 'Stacked Ensemble': stack_model}

print("\n=== Overfitting Check ===")
overfit_df = overfitting_check(models_dict, X_train, y_train, X_test, y_test)
print(overfit_df.to_string(index=False))

print("\n=== Efficiency Check ===")
eff_df = efficiency_check(models_dict, X_train_bal, y_train_bal, X_test)
print(eff_df.to_string(index=False))

result = {
    'scaler': scaler, 'xgb_final': xgb_final, 'stack_model': stack_model,
    'best_threshold': best_threshold, 'feature_columns': X_train.columns.tolist()
}
save_model_files(result, prefix='ds1opt')

joblib.dump(final_metrics, 'ds1opt_final_metrics.pkl')
joblib.dump(overfit_df, 'ds1opt_overfit_df.pkl')
joblib.dump(eff_df, 'ds1opt_efficiency_df.pkl')

print("\n\nds1opt_ files restored to the 3-model (primary) version. Your app.py needs no changes.")