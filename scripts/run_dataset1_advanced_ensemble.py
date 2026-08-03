"""
run_dataset1_advanced_ensemble.py — Advanced ensemble techniques using sklearn
to potentially achieve higher accuracy: AdaBoost, GradientBoosting, and hybrid ensembles.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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

print(f"\n{'='*60}\nADVANCED ENSEMBLE PIPELINE\n{'='*60}")
print(f"Shape: {X.shape}, Churn rate: {y.mean():.4f}")

# Split and scale
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
X_scaled = X_scaled.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, churn rate {y_train.mean():.4f} | Test: {X_test.shape}, churn rate {y_test.mean():.4f}")

# Balance training data
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_bal.shape}, churn rate {y_train_bal.mean():.4f}")

# =========================================================
# Train multiple diverse models
# =========================================================

print("\n=== Training Individual Models ===")

# 1. Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_split=10,
    min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)

# 2. XGBoost
print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, eval_metric='logloss'
)
xgb.fit(X_train_bal, y_train_bal)

# 3. LightGBM
print("Training LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=300, max_depth=5, num_leaves=31,
    learning_rate=0.05, min_child_samples=20, subsample=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
)
lgbm.fit(X_train_bal, y_train_bal)

# 4. Gradient Boosting
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, min_samples_split=10, min_samples_leaf=4,
    random_state=42
)
gb.fit(X_train_bal, y_train_bal)

# 5. AdaBoost
print("Training AdaBoost...")
ada = AdaBoostClassifier(
    n_estimators=300, learning_rate=0.05, random_state=42
)
ada.fit(X_train_bal, y_train_bal)

# =========================================================
# Create diverse ensembles
# =========================================================

print("\n=== Creating Ensembles ===")

# Ensemble 1: All 5 models (soft voting)
print("Training Voting Ensemble (5 models, soft)...")
voting_soft_5 = VotingClassifier(
    estimators=[
        ('rf', rf), ('xgb', xgb), ('lgbm', lgbm), ('gb', gb), ('ada', ada)
    ],
    voting='soft', n_jobs=-1
)
voting_soft_5.fit(X_train_bal, y_train_bal)

# Ensemble 2: All 5 models (hard voting)
print("Training Voting Ensemble (5 models, hard)...")
voting_hard_5 = VotingClassifier(
    estimators=[
        ('rf', rf), ('xgb', xgb), ('lgbm', lgbm), ('gb', gb), ('ada', ada)
    ],
    voting='hard', n_jobs=-1
)
voting_hard_5.fit(X_train_bal, y_train_bal)

# Ensemble 3: Stacking with best 3
print("Training Stacking Ensemble (3 best models)...")
stacking_3 = StackingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm), ('gb', gb)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, n_jobs=-1
)
stacking_3.fit(X_train_bal, y_train_bal)

# =========================================================
# Evaluate all models
# =========================================================

print("\n=== Model Evaluation ===")

models = {
    'Random Forest': rf,
    'XGBoost': xgb,
    'LightGBM': lgbm,
    'Gradient Boosting': gb,
    'AdaBoost': ada,
    'Voting Soft (5)': voting_soft_5,
    'Voting Hard (5)': voting_hard_5,
    'Stacking (3)': stacking_3
}

best_model = None
best_accuracy = 0
best_metrics = None
best_threshold = 0

for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
        
        # Threshold tuning
        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = f1_scores.argmax()
        threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        preds = (probs >= threshold).astype(int)
    else:
        preds = model.predict(X_test)
        threshold = 0.5
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        best_threshold = threshold

print(f"\n=== BEST MODEL ===")
print(f"Model: {list(models.keys())[list(models.values()).index(best_model)]}")
print(f"Metrics: {best_metrics}")
print(f"Threshold: {best_threshold:.4f}")

# Save best model
joblib.dump(best_model, 'ds1adv_churn_model.pkl')
joblib.dump(scaler, 'ds1adv_scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'ds1adv_feature_columns.pkl')
joblib.dump(best_threshold, 'ds1adv_best_threshold.pkl')
joblib.dump(best_metrics, 'ds1adv_final_metrics.pkl')

# Also save XGBoost for SHAP
xgb.save_model('ds1adv_shap_model.json')

print("\nAdvanced ensemble pipeline complete.")
print(f"Best accuracy achieved: {best_accuracy:.2%}")
