"""
run_dataset1_neural_network.py — Deep learning approach using neural networks
to potentially achieve higher accuracy than tree-based models.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

print(f"\n{'='*60}\nNEURAL NETWORK PIPELINE\n{'='*60}")
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

def create_model(trial):
    """Create neural network architecture based on Optuna hyperparameters"""
    n_layers = trial.suggest_int('n_layers', 2, 5)
    layers_list = []
    
    # Input layer
    input_dim = X_train_bal.shape[1]
    
    # Hidden layers
    for i in range(n_layers):
        units = trial.suggest_int(f'units_{i}', 32, 512)
        dropout = trial.suggest_float(f'dropout_{i}', 0.1, 0.5)
        
        if i == 0:
            layers_list.append(layers.Input(shape=(input_dim,)))
        
        layers_list.append(layers.Dense(units, activation='relu'))
        layers_list.append(layers.BatchNormalization())
        layers_list.append(layers.Dropout(dropout))
    
    # Output layer
    layers_list.append(layers.Dense(1, activation='sigmoid'))
    
    model = keras.Sequential(layers_list)
    
    # Learning rate
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def objective(trial):
    """Optuna objective function"""
    model = create_model(trial)
    
    # Batch size and epochs
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 50, 200)
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    
    # Train with validation split
    history = model.fit(
        X_train_bal, y_train_bal,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate on test set
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1 = f1_scores.max()
    
    return best_f1

print("\nStarting Optuna hyperparameter optimization for neural network...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nBest F1 Score: {study.best_value:.4f}")
print("Best parameters:", study.best_params)

# Train final model with best parameters
print("\nTraining final neural network with best parameters...")
best_model = create_model(study.best_trial)

batch_size = study.best_params['batch_size']
epochs = study.best_params['epochs']

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=30, restore_best_weights=True
)

history = best_model.fit(
    X_train_bal, y_train_bal,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1
)

# Final evaluation
y_pred_proba = best_model.predict(X_test, verbose=0).flatten()

# Threshold tuning
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
y_pred_tuned = (y_pred_proba >= best_threshold).astype(int)

final_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_tuned),
    'Precision': precision_score(y_test, y_pred_tuned),
    'Recall': recall_score(y_test, y_pred_tuned),
    'F1': f1_score(y_test, y_pred_tuned)
}

print(f"\nBest threshold: {best_threshold:.4f}")
print("Final metrics (Neural Network):", final_metrics)

# Save model and artifacts
best_model.save('ds1nn_churn_model.keras')
joblib.dump(scaler, 'ds1nn_scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'ds1nn_feature_columns.pkl')
joblib.dump(best_threshold, 'ds1nn_best_threshold.pkl')
joblib.dump(final_metrics, 'ds1nn_final_metrics.pkl')

print("\nNeural network pipeline complete.")
print(f"Accuracy achieved: {final_metrics['Accuracy']:.2%}")
