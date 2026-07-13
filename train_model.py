import pandas as pd
import numpy as np

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df.drop('customerID', axis=1, inplace=True)

print("Shape:", df.shape)
print("Churn distribution:\n", df['Churn'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Binary Yes/No columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Multi-category columns — one-hot encode
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']

df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Scale numeric features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Cast everything to float — avoids XGBoost dtype issues with bool columns
X = X.astype(float)

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nFinal encoded shape:", df_encoded.shape)
print("Train shape:", X_train.shape, "Churn rate:", y_train.mean())
print("Test shape:", X_test.shape, "Churn rate:", y_test.mean())

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Random Forest tuning (SMOTE inside CV pipeline — no leakage) ---
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
probs = best_rf.predict_proba(X_test)[:, 1]
print("RF Test — Acc:", accuracy_score(y_test, preds), "F1:", f1_score(y_test, preds),
      "Recall:", recall_score(y_test, preds), "Precision:", precision_score(y_test, preds))

# --- XGBoost tuning ---
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
probs = best_xgb.predict_proba(X_test)[:, 1]
print("XGB Test — Acc:", accuracy_score(y_test, preds), "F1:", f1_score(y_test, preds),
      "Recall:", recall_score(y_test, preds), "Precision:", precision_score(y_test, preds))


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve
import shap
import joblib

# Rebuild final tuned models with the exact best params found above
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

lgbm_final = LGBMClassifier(random_state=42, verbose=-1)

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

import time
import os

# --- Efficiency benchmarking ---
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

# Time the full stacked ensemble too
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

# Model file size, as a deployability metric
joblib.dump(stack_model, 'churn_model.pkl')  # ensure it's saved before checking size
model_size_kb = os.path.getsize('churn_model.pkl') / 1024
print(f"\nStacked model file size: {model_size_kb:.1f} KB")

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

print("\n=== Overfitting Check: Train vs Test Performance ===")

overfit_results = []

for name, m in [('Random Forest', rf_final), ('XGBoost', xgb_final),
                 ('LightGBM', lgbm_final), ('Stacked Ensemble', stack_model)]:
    # Predictions on TRAINING data (the model has seen this)
    train_preds = m.predict(X_train_bal)
    train_acc = accuracy_score(y_train_bal, train_preds)
    train_f1 = f1_score(y_train_bal, train_preds)

    # Predictions on TEST data (unseen)
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
print("\nRule of thumb: Gap > 0.10 (10 percentage points) suggests meaningful overfitting.")
print("Gap < 0.05 is generally healthy.")

# --- Save everything the app needs ---
joblib.dump(stack_model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_columns.pkl')
joblib.dump(best_threshold, 'best_threshold.pkl')
xgb_final.save_model('churn_shap_model.json')  # native format, for SHAP explanations in the app

print("\nAll model files saved successfully.")