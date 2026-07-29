"""
pipeline.py — Shared, reusable churn-prediction pipeline.
Used by run_dataset1.py (Telco) and run_dataset2.py (Indian high-value churn),
so both datasets go through an identical, documented methodology.

Includes both RandomizedSearchCV-based tuning (tune_random_forest, tune_xgboost,
tune_lightgbm / run_full_pipeline) and Optuna TPE-based tuning
(tune_*_optuna / run_full_pipeline_optuna) as a smarter alternative.
"""

import time
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =========================================================
# Shared utilities
# =========================================================

def split_and_scale(X, y, numeric_cols, test_size=0.2, random_state=42):
    """Stratified split + scale numeric columns. Returns X_train, X_test, y_train, y_test, scaler."""
    scaler = StandardScaler()
    X = X.copy()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def train_stacked_ensemble(rf_final, xgb_final, lgbm_final, X_train, y_train, random_state=42):
    """Balances training data, trains the stack, tunes the decision threshold on held-out test data."""
    X_train_bal, y_train_bal = SMOTE(random_state=random_state).fit_resample(X_train, y_train)

    stack_model = StackingClassifier(
        estimators=[('rf', rf_final), ('xgb', xgb_final), ('lgbm', lgbm_final)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, n_jobs=-1
    )
    stack_model.fit(X_train_bal, y_train_bal)
    return stack_model, X_train_bal, y_train_bal


def tune_threshold(stack_model, X_test, y_test):
    probs = stack_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    return best_threshold, probs


def evaluate(y_true, preds):
    return {
        'Accuracy': accuracy_score(y_true, preds),
        'Precision': precision_score(y_true, preds),
        'Recall': recall_score(y_true, preds),
        'F1': f1_score(y_true, preds)
    }


def overfitting_check(models_dict, X_train, y_train, X_test, y_test):
    """models_dict: {'Model Name': fitted_model, ...}. Evaluates on ORIGINAL (unbalanced) train vs test."""
    results = []
    for name, m in models_dict.items():
        train_preds = m.predict(X_train)
        test_preds = m.predict(X_test)
        train_acc, test_acc = accuracy_score(y_train, train_preds), accuracy_score(y_test, test_preds)
        train_f1, test_f1 = f1_score(y_train, train_preds), f1_score(y_test, test_preds)
        results.append({
            'Model': name,
            'Train Acc': round(train_acc, 4), 'Test Acc': round(test_acc, 4),
            'Acc Gap': round(train_acc - test_acc, 4),
            'Train F1': round(train_f1, 4), 'Test F1': round(test_f1, 4),
            'F1 Gap': round(train_f1 - test_f1, 4)
        })
    return pd.DataFrame(results)


def efficiency_check(models_dict, X_train_bal, y_train_bal, X_test):
    """models_dict: {'Model Name': unfitted_or_fitted_model, ...}"""
    results = []
    for name, m in models_dict.items():
        start = time.time()
        m.fit(X_train_bal, y_train_bal)
        train_time = time.time() - start

        start = time.time()
        _ = m.predict(X_test)
        inference_time = time.time() - start

        results.append({
            'Model': name,
            'Train Time (s)': round(train_time, 3),
            'Inference Time (s)': round(inference_time, 4),
            'Avg Inference per row (ms)': round((inference_time / len(X_test)) * 1000, 4)
        })
    return pd.DataFrame(results)


def save_model_files(result, prefix):
    """Saves all deployable artifacts for a pipeline run, prefixed so dataset1/dataset2 files don't collide."""
    joblib.dump(result['stack_model'], f'{prefix}_churn_model.pkl')
    joblib.dump(result['scaler'], f'{prefix}_scaler.pkl')
    joblib.dump(result['feature_columns'], f'{prefix}_feature_columns.pkl')
    joblib.dump(result['best_threshold'], f'{prefix}_best_threshold.pkl')
    result['xgb_final'].save_model(f'{prefix}_shap_model.json')
    print(f"\nSaved all model files with prefix '{prefix}_'.")


# =========================================================
# RandomizedSearchCV-based tuning (original approach)
# =========================================================

def tune_random_forest(X_train, y_train, n_iter=30, cv=5, random_state=42, custom_params=None):
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('clf', RandomForestClassifier(random_state=random_state, n_jobs=-1))
    ])
    params = custom_params if custom_params is not None else {
        'clf__n_estimators': [100, 200, 300, 400],
        'clf__max_depth': [5, 10, 15, 20, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4],
        'clf__max_features': ['sqrt', 'log2']
    }
    search = RandomizedSearchCV(pipeline, params, n_iter=n_iter, scoring='f1',
                                 cv=cv, random_state=random_state, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def tune_xgboost(X_train, y_train, n_iter=30, cv=5, random_state=42, custom_params=None):
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('clf', XGBClassifier(random_state=random_state, eval_metric='logloss',
                               tree_method='hist', device='cuda'))
    ])
    params = custom_params if custom_params is not None else {
        'clf__n_estimators': [100, 200, 300, 400],
        'clf__max_depth': [3, 4, 5, 6, 8],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__subsample': [0.7, 0.8, 0.9, 1.0],
        'clf__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'clf__min_child_weight': [1, 3, 5]
    }
    search = RandomizedSearchCV(pipeline, params, n_iter=n_iter, scoring='f1',
                                 cv=cv, random_state=random_state, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def tune_lightgbm(X_train, y_train, n_iter=30, cv=5, random_state=42, custom_params=None):
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('clf', LGBMClassifier(random_state=random_state, verbose=-1))
    ])
    params = custom_params if custom_params is not None else {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [3, 4, 5, 6, -1],
        'clf__num_leaves': [15, 31, 50, 70],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        'clf__min_child_samples': [10, 20, 30, 50],
        'clf__subsample': [0.7, 0.8, 0.9, 1.0],
        'clf__reg_alpha': [0, 0.1, 0.5, 1.0],
        'clf__reg_lambda': [0, 0.1, 0.5, 1.0],
    }
    search = RandomizedSearchCV(pipeline, params, n_iter=n_iter, scoring='f1',
                                 cv=cv, random_state=random_state, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def build_final_models(rf_search, xgb_search, lgbm_search, random_state=42):
    rf_final = RandomForestClassifier(
        n_estimators=rf_search.best_params_['clf__n_estimators'],
        min_samples_split=rf_search.best_params_['clf__min_samples_split'],
        min_samples_leaf=rf_search.best_params_['clf__min_samples_leaf'],
        max_features=rf_search.best_params_['clf__max_features'],
        max_depth=rf_search.best_params_['clf__max_depth'],
        random_state=random_state, n_jobs=-1
    )
    xgb_kwargs = dict(
        subsample=xgb_search.best_params_['clf__subsample'],
        n_estimators=xgb_search.best_params_['clf__n_estimators'],
        min_child_weight=xgb_search.best_params_['clf__min_child_weight'],
        max_depth=xgb_search.best_params_['clf__max_depth'],
        learning_rate=xgb_search.best_params_['clf__learning_rate'],
        colsample_bytree=xgb_search.best_params_['clf__colsample_bytree'],
        random_state=random_state, eval_metric='logloss',
        tree_method='hist', device='cuda'
    )
    if 'clf__reg_alpha' in xgb_search.best_params_:
        xgb_kwargs['reg_alpha'] = xgb_search.best_params_['clf__reg_alpha']
    if 'clf__reg_lambda' in xgb_search.best_params_:
        xgb_kwargs['reg_lambda'] = xgb_search.best_params_['clf__reg_lambda']
    xgb_final = XGBClassifier(**xgb_kwargs)

    lgbm_final = LGBMClassifier(
        n_estimators=lgbm_search.best_params_['clf__n_estimators'],
        max_depth=lgbm_search.best_params_['clf__max_depth'],
        num_leaves=lgbm_search.best_params_['clf__num_leaves'],
        learning_rate=lgbm_search.best_params_['clf__learning_rate'],
        min_child_samples=lgbm_search.best_params_['clf__min_child_samples'],
        subsample=lgbm_search.best_params_['clf__subsample'],
        reg_alpha=lgbm_search.best_params_['clf__reg_alpha'],
        reg_lambda=lgbm_search.best_params_['clf__reg_lambda'],
        random_state=random_state, verbose=-1
    )
    return rf_final, xgb_final, lgbm_final


def run_full_pipeline(X, y, numeric_cols, dataset_name="Dataset", n_iter=30,
                       rf_custom_params=None, xgb_custom_params=None, lgbm_custom_params=None):
    """
    Runs the ENTIRE pipeline end-to-end on a given (X, y) using RandomizedSearchCV tuning.
    Pass rf_custom_params / xgb_custom_params / lgbm_custom_params to override the
    default hyperparameter search grids (e.g., stricter regularization for
    high-dimensional or heavily imbalanced datasets).
    """
    print(f"\n{'='*60}\nRUNNING FULL PIPELINE ON: {dataset_name}\n{'='*60}")
    print(f"Shape: {X.shape}, Churn rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, numeric_cols)
    print(f"Train: {X_train.shape}, churn rate {y_train.mean():.4f} | Test: {X_test.shape}, churn rate {y_test.mean():.4f}")

    print("\nTuning Random Forest...")
    rf_search = tune_random_forest(X_train, y_train, n_iter=n_iter, custom_params=rf_custom_params)
    print("Best RF CV F1:", rf_search.best_score_)

    print("\nTuning XGBoost...")
    xgb_search = tune_xgboost(X_train, y_train, n_iter=n_iter, custom_params=xgb_custom_params)
    print("Best XGBoost CV F1:", xgb_search.best_score_)

    print("\nTuning LightGBM...")
    lgbm_search = tune_lightgbm(X_train, y_train, n_iter=n_iter, custom_params=lgbm_custom_params)
    print("Best LightGBM CV F1:", lgbm_search.best_score_)

    rf_final, xgb_final, lgbm_final = build_final_models(rf_search, xgb_search, lgbm_search)

    print("\nTraining stacked ensemble...")
    stack_model, X_train_bal, y_train_bal = train_stacked_ensemble(rf_final, xgb_final, lgbm_final, X_train, y_train)

    best_threshold, probs = tune_threshold(stack_model, X_test, y_test)
    preds_tuned = (probs >= best_threshold).astype(int)
    final_metrics = evaluate(y_test, preds_tuned)
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"Final metrics ({dataset_name}):", final_metrics)

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

    return {
        'dataset_name': dataset_name,
        'scaler': scaler, 'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'rf_final': rf_final, 'xgb_final': xgb_final, 'lgbm_final': lgbm_final,
        'stack_model': stack_model, 'best_threshold': best_threshold,
        'final_metrics': final_metrics, 'overfit_df': overfit_df, 'efficiency_df': eff_df,
        'feature_columns': X_train.columns.tolist()
    }


# =========================================================
# Optuna (TPE) based tuning — smarter search than RandomizedSearchCV
# =========================================================

def tune_random_forest_optuna(X_train, y_train, n_trials=30, cv=5, random_state=42, search_bounds=None):
    b = search_bounds or {}
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *b.get('n_estimators', (100, 500))),
            'max_depth': trial.suggest_int('max_depth', *b.get('max_depth', (3, 25))),
            'min_samples_split': trial.suggest_int('min_samples_split', *b.get('min_samples_split', (2, 20))),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *b.get('min_samples_leaf', (1, 15))),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('clf', RandomForestClassifier(random_state=random_state, n_jobs=-1, **params))
        ])
        return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def tune_xgboost_optuna(X_train, y_train, n_trials=30, cv=5, random_state=42, search_bounds=None):
    b = search_bounds or {}
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *b.get('n_estimators', (100, 500))),
            'max_depth': trial.suggest_int('max_depth', *b.get('max_depth', (2, 10))),
            'learning_rate': trial.suggest_float('learning_rate', *b.get('learning_rate', (0.005, 0.3)), log=True),
            'subsample': trial.suggest_float('subsample', *b.get('subsample', (0.5, 1.0))),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *b.get('colsample_bytree', (0.5, 1.0))),
            'min_child_weight': trial.suggest_int('min_child_weight', *b.get('min_child_weight', (1, 15))),
            'reg_alpha': trial.suggest_float('reg_alpha', *b.get('reg_alpha', (1e-3, 5.0)), log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', *b.get('reg_lambda', (1e-3, 5.0)), log=True),
        }
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('clf', XGBClassifier(random_state=random_state, eval_metric='logloss',
                                   tree_method='hist', device='cuda', **params))
        ])
        # n_jobs=1 here — GPU training doesn't parallelize safely across CV folds the way CPU does
        return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=1).mean()

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def tune_lightgbm_optuna(X_train, y_train, n_trials=30, cv=5, random_state=42, search_bounds=None):
    b = search_bounds or {}
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *b.get('n_estimators', (100, 400))),
            'max_depth': trial.suggest_int('max_depth', *b.get('max_depth', (3, 10))),
            'num_leaves': trial.suggest_int('num_leaves', *b.get('num_leaves', (7, 80))),
            'learning_rate': trial.suggest_float('learning_rate', *b.get('learning_rate', (0.005, 0.3)), log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', *b.get('min_child_samples', (10, 70))),
            'subsample': trial.suggest_float('subsample', *b.get('subsample', (0.5, 1.0))),
            'reg_alpha': trial.suggest_float('reg_alpha', *b.get('reg_alpha', (1e-3, 5.0)), log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', *b.get('reg_lambda', (1e-3, 5.0)), log=True),
        }
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('clf', LGBMClassifier(random_state=random_state, verbose=-1, **params))
        ])
        return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1).mean()

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def build_final_models_from_optuna(rf_study, xgb_study, lgbm_study, random_state=42):
    rf_final = RandomForestClassifier(**rf_study.best_params, random_state=random_state, n_jobs=-1)
    xgb_final = XGBClassifier(**xgb_study.best_params, random_state=random_state,
                               eval_metric='logloss', tree_method='hist', device='cuda')
    lgbm_final = LGBMClassifier(**lgbm_study.best_params, random_state=random_state, verbose=-1)
    return rf_final, xgb_final, lgbm_final


def run_full_pipeline_optuna(X, y, numeric_cols, dataset_name="Dataset", n_trials=30,
                              rf_bounds=None, xgb_bounds=None, lgbm_bounds=None):
    """
    Same end-to-end pipeline as run_full_pipeline, but using Optuna (TPE search)
    instead of RandomizedSearchCV for all three base learners.
    Pass rf_bounds / xgb_bounds / lgbm_bounds (dicts of param_name -> (low, high))
    to tighten the search space for high-dimensional or heavily imbalanced datasets.
    """
    print(f"\n{'='*60}\nRUNNING FULL PIPELINE (OPTUNA) ON: {dataset_name}\n{'='*60}")
    print(f"Shape: {X.shape}, Churn rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, numeric_cols)
    print(f"Train: {X_train.shape}, churn rate {y_train.mean():.4f} | Test: {X_test.shape}, churn rate {y_test.mean():.4f}")

    print("\nTuning Random Forest (Optuna)...")
    rf_study = tune_random_forest_optuna(X_train, y_train, n_trials=n_trials, search_bounds=rf_bounds)
    print("Best RF CV F1:", rf_study.best_value, "| params:", rf_study.best_params)

    print("\nTuning XGBoost (Optuna)...")
    xgb_study = tune_xgboost_optuna(X_train, y_train, n_trials=n_trials, search_bounds=xgb_bounds)
    print("Best XGBoost CV F1:", xgb_study.best_value, "| params:", xgb_study.best_params)

    print("\nTuning LightGBM (Optuna)...")
    lgbm_study = tune_lightgbm_optuna(X_train, y_train, n_trials=n_trials, search_bounds=lgbm_bounds)
    print("Best LightGBM CV F1:", lgbm_study.best_value, "| params:", lgbm_study.best_params)

    rf_final, xgb_final, lgbm_final = build_final_models_from_optuna(rf_study, xgb_study, lgbm_study)

    print("\nTraining stacked ensemble...")
    stack_model, X_train_bal, y_train_bal = train_stacked_ensemble(rf_final, xgb_final, lgbm_final, X_train, y_train)

    best_threshold, probs = tune_threshold(stack_model, X_test, y_test)
    preds_tuned = (probs >= best_threshold).astype(int)
    final_metrics = evaluate(y_test, preds_tuned)
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"Final metrics ({dataset_name}):", final_metrics)

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

    return {
        'dataset_name': dataset_name,
        'scaler': scaler, 'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'rf_final': rf_final, 'xgb_final': xgb_final, 'lgbm_final': lgbm_final,
        'stack_model': stack_model, 'best_threshold': best_threshold,
        'final_metrics': final_metrics, 'overfit_df': overfit_df, 'efficiency_df': eff_df,
        'feature_columns': X_train.columns.tolist()
    }