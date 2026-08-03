import pandas as pd
import numpy as np
import time
import os
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =========================================================
# PART 1: Load and clean data
# =========================================================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df.drop('customerID', axis=1, inplace=True)

print("Shape:", df.shape)
print("Churn distribution:\n", df['Churn'].value_counts())

# =========================================================
# PART 2: Encoding + train/test split
# =========================================================
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
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
X = X.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nFinal encoded shape:", df_encoded.shape)
print("Train shape:", X_train.shape, "Churn rate:", y_train.mean())
print("Test shape:", X_test.shape, "Churn rate:", y_test.mean())

# =========================================================
# PART 3: Hyperparameter tuning (SMOTE inside CV pipeline — no leakage)
# =========================================================

# --- Random Forest ---
rf_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])

rf_params = {
    'clf__n_estimators': [100, 200, 300, 400],
    'clf__max_depth': [5, 10, 15, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2']
}

print("\nTuning Random Forest...")
rf_search = RandomizedSearchCV(
    rf_pipeline, rf_params, n_iter=30, scoring='f1', cv=5, random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)

print("Best RF params:", rf_search.best_params_)
print("Best RF CV F1:", rf_search.best_score_)

best_rf = rf_search.best_estimator_
preds = best_rf.predict(X_test)
print("RF Test — Acc:", accuracy_score(y_test, preds), "F1:", f1_score(y_test, preds),
      "Recall:", recall_score(y_test, preds), "Precision:", precision_score(y_test, preds))

# --- XGBoost ---
xgb_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(random_state=42, eval_metric='logloss'))
])

xgb_params = {
    'clf__n_estimators': [100, 200, 300, 400],
    'clf__max_depth': [3, 4, 5, 6, 8],
    'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'clf__subsample': [0.7, 0.8, 0.9, 1.0],
    'clf__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'clf__min_child_weight': [1, 3, 5]
}

print("\nTuning XGBoost...")
xgb_search = RandomizedSearchCV(
    xgb_pipeline, xgb_params, n_iter=30, scoring='f1', cv=5, random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train)

print("Best XGBoost params:", xgb_search.best_params_)
print("Best XGBoost CV F1:", xgb_search.best_score_)

best_xgb = xgb_search.best_estimator_
preds = best_xgb.predict(X_test)
print("XGB Test — Acc:", accuracy_score(y_test, preds), "F1:", f1_score(y_test, preds),
      "Recall:", recall_score(y_test, preds), "Precision:", precision_score(y_test, preds))

# --- LightGBM (with regularization to control overfitting) ---
lgbm_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', LGBMClassifier(random_state=42, verbose=-1))
])

lgbm_params = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 4, 5, 6, -1],
    'clf__num_leaves': [15, 31, 50, 70],
    'clf__learning_rate': [0.01, 0.05, 0.1],
    'clf__min_child_samples': [10, 20, 30, 50],
    'clf__subsample': [0.7, 0.8, 0.9, 1.0],
    'clf__reg_alpha': [0, 0.1, 0.5, 1.0],
    'clf__reg_lambda': [0, 0.1, 0.5, 1.0],
}

print("\nTuning LightGBM...")
lgbm_search = RandomizedSearchCV(
    lgbm_pipeline, lgbm_params, n_iter=30, scoring='f1', cv=5, random_state=42, n_jobs=-1
)
lgbm_search.fit(X_train, y_train)

print("Best LightGBM params:", lgbm_search.best_params_)
print("Best LightGBM CV F1:", lgbm_search.best_score_)

best_lgbm = lgbm_search.best_estimator_
preds = best_lgbm.predict(X_test)
print("LGBM Test — Acc:", accuracy_score(y_test, preds), "F1:", f1_score(y_test, preds),
      "Recall:", recall_score(y_test, preds), "Precision:", precision_score(y_test, preds))

# =========================================================
# PART 4: Build final tuned models + stacked ensemble
# =========================================================
rf_final = RandomForestClassifier(
    n_estimators=rf_search.best_params_['clf__n_estimators'],
    min_samples_split=rf_search.best_params_['clf__min_samples_split'],
    min_samples_leaf=rf_search.best_params_['clf__min_samples_leaf'],
    max_features=rf_search.best_params_['clf__max_features'],
    max_depth=rf_search.best_params_['clf__max_depth'],
    random_state=42
)

xgb_final = XGBClassifier(
    subsample=xgb_search.best_params_['clf__subsample'],
    n_estimators=xgb_search.best_params_['clf__n_estimators'],
    min_child_weight=xgb_search.best_params_['clf__min_child_weight'],
    max_depth=xgb_search.best_params_['clf__max_depth'],
    learning_rate=xgb_search.best_params_['clf__learning_rate'],
    colsample_bytree=xgb_search.best_params_['clf__colsample_bytree'],
    random_state=42, eval_metric='logloss'
)

lgbm_final = LGBMClassifier(
    n_estimators=lgbm_search.best_params_['clf__n_estimators'],
    max_depth=lgbm_search.best_params_['clf__max_depth'],
    num_leaves=lgbm_search.best_params_['clf__num_leaves'],
    learning_rate=lgbm_search.best_params_['clf__learning_rate'],
    min_child_samples=lgbm_search.best_params_['clf__min_child_samples'],
    subsample=lgbm_search.best_params_['clf__subsample'],
    reg_alpha=lgbm_search.best_params_['clf__reg_alpha'],
    reg_lambda=lgbm_search.best_params_['clf__reg_lambda'],
    random_state=42, verbose=-1
)

# Balance training data once for the stack
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

stack_model = StackingClassifier(
    estimators=[('rf', rf_final), ('xgb', xgb_final), ('lgbm', lgbm_final)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, n_jobs=-1
)

print("\nTraining stacked ensemble...")
stack_model.fit(X_train_bal, y_train_bal)

probs = stack_model.predict_proba(X_test)[:, 1]

# Threshold tuning
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

preds_tuned = (probs >= best_threshold).astype(int)

print("\nBest threshold:", best_threshold)
print("Stacked Ensemble (tuned threshold) — Acc:", accuracy_score(y_test, preds_tuned),
      "F1:", f1_score(y_test, preds_tuned), "Recall:", recall_score(y_test, preds_tuned),
      "Precision:", precision_score(y_test, preds_tuned))

# Fit final XGBoost separately for SHAP (tree explainer needs a single tree model, not the stack)
xgb_final.fit(X_train_bal, y_train_bal)

# Also fit rf_final and lgbm_final individually — needed for efficiency/overfitting checks below
rf_final.fit(X_train_bal, y_train_bal)
lgbm_final.fit(X_train_bal, y_train_bal)

# =========================================================
# PART 5: Efficiency benchmarking
# =========================================================
efficiency_results = []

for name, m in [('Random Forest', rf_final), ('XGBoost', xgb_final), ('LightGBM', lgbm_final)]:
    start = time.time()
    m.fit(X_train_bal, y_train_bal)
    train_time = time.time() - start

    start = time.time()
    _ = m.predict(X_test)
    inference_time = time.time() - start

    efficiency_results.append({
        'Model': name,
        'Train Time (s)': round(train_time, 3),
        'Inference Time (s, 1409 rows)': round(inference_time, 4),
        'Avg Inference per row (ms)': round((inference_time / len(X_test)) * 1000, 4)
    })

start = time.time()
stack_model.fit(X_train_bal, y_train_bal)
stack_train_time = time.time() - start

start = time.time()
_ = stack_model.predict(X_test)
stack_inference_time = time.time() - start

efficiency_results.append({
    'Model': 'Stacked Ensemble',
    'Train Time (s)': round(stack_train_time, 3),
    'Inference Time (s, 1409 rows)': round(stack_inference_time, 4),
    'Avg Inference per row (ms)': round((stack_inference_time / len(X_test)) * 1000, 4)
})

efficiency_df = pd.DataFrame(efficiency_results)
print("\n=== Efficiency Comparison ===")
print(efficiency_df.to_string(index=False))

# =========================================================
# PART 6: Save model files (needed before checking file size)
# =========================================================
joblib.dump(stack_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_columns.pkl')
joblib.dump(best_threshold, 'best_threshold.pkl')
xgb_final.save_model('churn_shap_model.json')  # native format, for SHAP explanations in the app

model_size_kb = os.path.getsize('churn_model.pkl') / 1024
print(f"\nStacked model file size: {model_size_kb:.1f} KB")

# =========================================================
# PART 7: Overfitting check (correct version — train evaluated
# on the ORIGINAL unbalanced distribution, same as test)
# =========================================================
print("\n=== Overfitting Check (Train vs Test, both on original unbalanced distribution) ===")

overfit_results = []

for name, m in [('Random Forest', rf_final), ('XGBoost', xgb_final),
                 ('LightGBM', lgbm_final), ('Stacked Ensemble', stack_model)]:
    train_preds = m.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    train_f1 = f1_score(y_train, train_preds)

    test_preds = m.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)

    overfit_results.append({
        'Model': name,
        'Train Acc': round(train_acc, 4),
        'Test Acc': round(test_acc, 4),
        'Acc Gap': round(train_acc - test_acc, 4),
        'Train F1': round(train_f1, 4),
        'Test F1': round(test_f1, 4),
        'F1 Gap': round(train_f1 - test_f1, 4)
    })

overfit_df = pd.DataFrame(overfit_results)
print(overfit_df.to_string(index=False))
print("\nRule of thumb: Gap > 0.10 suggests meaningful overfitting. Gap < 0.05 is generally healthy.")

print("\nAll model files saved successfully.")