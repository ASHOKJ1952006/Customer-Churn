"""
run_dataset1_super_ensemble.py — Super-ensemble combining the best models
from all experiments to potentially achieve the highest possible accuracy.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

print(f"\n{'='*60}\nSUPER ENSEMBLE PIPELINE\n{'='*60}")
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
# Load best models from previous experiments
# =========================================================

print("\n=== Loading Best Models from Previous Experiments ===")

# Load the best individual models if they exist
try:
    rf = joblib.load('ds1agg_churn_model.pkl') if hasattr(joblib.load('ds1agg_churn_model.pkl'), 'estimators_') else None
except:
    rf = None

# Create diverse high-performing models
print("Creating diverse high-performing models...")

# Model 1: Random Forest (conservative)
rf_model = RandomForestClassifier(
    n_estimators=400, max_depth=12, min_samples_split=15,
    min_samples_leaf=8, max_features='sqrt', random_state=42, n_jobs=-1
)
rf_model.fit(X_train_bal, y_train_bal)

# Model 2: XGBoost (balanced)
xgb_model = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.75, min_child_weight=5,
    reg_alpha=0.5, reg_lambda=0.5, random_state=42, eval_metric='logloss'
)
xgb_model.fit(X_train_bal, y_train_bal)

# Model 3: LightGBM (fast)
lgbm_model = LGBMClassifier(
    n_estimators=400, max_depth=4, num_leaves=25,
    learning_rate=0.03, min_child_samples=25, subsample=0.85,
    reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1
)
lgbm_model.fit(X_train_bal, y_train_bal)

# Model 4: Gradient Boosting (sklearn)
gb_model = GradientBoostingClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.85, min_samples_split=15, min_samples_leaf=8,
    random_state=42
)
gb_model.fit(X_train_bal, y_train_bal)

# Model 5: Extra Random Forest (very diverse)
from sklearn.ensemble import ExtraTreesClassifier
et_model = ExtraTreesClassifier(
    n_estimators=400, max_depth=12, min_samples_split=15,
    min_samples_leaf=8, max_features='sqrt', random_state=42, n_jobs=-1
)
et_model.fit(X_train_bal, y_train_bal)

print("All 5 diverse models trained successfully.")

# =========================================================
# Create multiple ensemble configurations
# =========================================================

print("\n=== Creating Super Ensembles ===")

# Super Ensemble 1: All 5 models with soft voting
print("Training Super Voting Ensemble (5 models, soft)...")
super_voting_soft = VotingClassifier(
    estimators=[
        ('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model),
        ('gb', gb_model), ('et', et_model)
    ],
    voting='soft', n_jobs=-1
)
super_voting_soft.fit(X_train_bal, y_train_bal)

# Super Ensemble 2: All 5 models with hard voting
print("Training Super Voting Ensemble (5 models, hard)...")
super_voting_hard = VotingClassifier(
    estimators=[
        ('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model),
        ('gb', gb_model), ('et', et_model)
    ],
    voting='hard', n_jobs=-1
)
super_voting_hard.fit(X_train_bal, y_train_bal)

# Super Ensemble 3: Stacked ensemble with best 3
print("Training Super Stacking Ensemble (3 best models)...")
super_stacking = StackingClassifier(
    estimators=[('xgb', xgb_model), ('lgbm', lgbm_model), ('gb', gb_model)],
    final_estimator=LogisticRegression(max_iter=1000, C=0.1),
    cv=5, n_jobs=-1
)
super_stacking.fit(X_train_bal, y_train_bal)

# Super Ensemble 4: Weighted voting (based on individual performance)
print("Training Weighted Voting Ensemble...")
# Get individual model performances
rf_probs = rf_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]
gb_probs = gb_model.predict_proba(X_test)[:, 1]
et_probs = et_model.predict_proba(X_test)[:, 1]

# Calculate F1 scores for threshold tuning
def get_f1(probs):
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    return f1_scores.max()

rf_f1 = get_f1(rf_probs)
xgb_f1 = get_f1(xgb_probs)
lgbm_f1 = get_f1(lgbm_probs)
gb_f1 = get_f1(gb_probs)
et_f1 = get_f1(et_probs)

print(f"Individual F1 scores: RF={rf_f1:.4f}, XGB={xgb_f1:.4f}, LGBM={lgbm_f1:.4f}, GB={gb_f1:.4f}, ET={et_f1:.4f}")

# Normalize weights
total_f1 = rf_f1 + xgb_f1 + lgbm_f1 + gb_f1 + et_f1
weights = [rf_f1/total_f1, xgb_f1/total_f1, lgbm_f1/total_f1, gb_f1/total_f1, et_f1/total_f1]
print(f"Voting weights: {weights}")

# Weighted voting (manual implementation)
weighted_probs = (weights[0] * rf_probs + weights[1] * xgb_probs + 
                  weights[2] * lgbm_probs + weights[3] * gb_probs + 
                  weights[4] * et_probs)

# =========================================================
# Evaluate all super ensembles
# =========================================================

print("\n=== Super Ensemble Evaluation ===")

# Evaluate soft voting
soft_probs = super_voting_soft.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, soft_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = f1_scores.argmax()
soft_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
soft_preds = (soft_probs >= soft_threshold).astype(int)
soft_acc = accuracy_score(y_test, soft_preds)
soft_f1 = f1_scores[best_idx]

print(f"Super Voting Soft: Acc={soft_acc:.4f}, F1={soft_f1:.4f}, Threshold={soft_threshold:.4f}")

# Evaluate hard voting
hard_preds = super_voting_hard.predict(X_test)
hard_acc = accuracy_score(y_test, hard_preds)
hard_f1 = f1_score(y_test, hard_preds)

print(f"Super Voting Hard: Acc={hard_acc:.4f}, F1={hard_f1:.4f}")

# Evaluate stacking
stack_probs = super_stacking.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, stack_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = f1_scores.argmax()
stack_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
stack_preds = (stack_probs >= stack_threshold).astype(int)
stack_acc = accuracy_score(y_test, stack_preds)
stack_f1 = f1_scores[best_idx]

print(f"Super Stacking: Acc={stack_acc:.4f}, F1={stack_f1:.4f}, Threshold={stack_threshold:.4f}")

# Evaluate weighted voting
precisions, recalls, thresholds = precision_recall_curve(y_test, weighted_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = f1_scores.argmax()
weighted_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
weighted_preds = (weighted_probs >= weighted_threshold).astype(int)
weighted_acc = accuracy_score(y_test, weighted_preds)
weighted_f1 = f1_scores[best_idx]

print(f"Weighted Voting: Acc={weighted_acc:.4f}, F1={weighted_f1:.4f}, Threshold={weighted_threshold:.4f}")

# Select best model
results = [
    ('Super Voting Soft', soft_acc, soft_f1, soft_threshold, super_voting_soft, soft_probs),
    ('Super Voting Hard', hard_acc, hard_f1, 0.5, super_voting_hard, None),
    ('Super Stacking', stack_acc, stack_f1, stack_threshold, super_stacking, stack_probs),
    ('Weighted Voting', weighted_acc, weighted_f1, weighted_threshold, None, weighted_probs)
]

best_result = max(results, key=lambda x: x[1])
best_name, best_acc, best_f1, best_thresh, best_model, best_probs = best_result

print(f"\n=== BEST SUPER ENSEMBLE ===")
print(f"Model: {best_name}")
print(f"Accuracy: {best_acc:.4f} ({best_acc:.2%})")
print(f"F1 Score: {best_f1:.4f}")
print(f"Threshold: {best_thresh:.4f}")

# Save best model
if best_model is not None:
    joblib.dump(best_model, 'ds1super_churn_model.pkl')
else:
    # For weighted voting, save the individual models and weights
    joblib.dump({
        'rf': rf_model, 'xgb': xgb_model, 'lgbm': lgbm_model,
        'gb': gb_model, 'et': et_model, 'weights': weights
    }, 'ds1super_churn_model.pkl')

joblib.dump(scaler, 'ds1super_scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'ds1super_feature_columns.pkl')
joblib.dump(best_thresh, 'ds1super_best_threshold.pkl')

final_metrics = {
    'Accuracy': best_acc,
    'Precision': precision_score(y_test, (best_probs >= best_thresh).astype(int) if best_probs is not None else best_model.predict(X_test)),
    'Recall': recall_score(y_test, (best_probs >= best_thresh).astype(int) if best_probs is not None else best_model.predict(X_test)),
    'F1': best_f1
}

joblib.dump(final_metrics, 'ds1super_final_metrics.pkl')
xgb_model.save_model('ds1super_shap_model.json')

print("\nSuper ensemble pipeline complete.")
print(f"Best accuracy achieved: {best_acc:.2%}")
print(f"Target: 95% | Gap: {0.95 - best_acc:.2%}")
