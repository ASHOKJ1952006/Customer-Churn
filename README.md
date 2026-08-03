# Telecom Customer Churn Prediction

A churn prediction system for Indian telecom providers with focus on recall (catching real churners) rather than just accuracy. Includes explainability (SHAP + natural language), persistent storage, and a working demo application.

## Project Structure

```
churn-prediction-app/
├── app.py                      # Streamlit application (main entry point)
├── database.py                 # SQLite database for persistent storage
├── shap_explainer.py           # Natural language SHAP explanations
├── PAPER_STRUCTURE.md          # Complete paper outline
├── README.md                   # This file
│
├── models/                     # Trained model artifacts
│   ├── ds1calcv_*.pkl         # Dataset 1 calibrated model (primary for app)
│   ├── ds1fe_shap_model.json   # Dataset 1 SHAP model
│   ├── ds2fs_selected_*.pkl    # Dataset 2 feature-selected model
│   └── ds2fs_selected_shap_model.json
│
├── data/                       # Datasets and database
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset 1 (Telco)
│   ├── telecom_churn_data.csv  # Dataset 2 (Indian telecom)
│   ├── churn_predictions.db    # SQLite database
│   ├── test_batch_upload.csv  # Test file for batch upload
│   └── final_cross_dataset_comparison.csv
│
├── scripts/                    # Training and utility scripts
│   ├── run_dataset1_*.py       # Dataset 1 training scripts
│   ├── run_dataset2_*.py       # Dataset 2 training scripts
│   ├── train_model.py          # General training script
│   ├── pipeline.py             # ML pipeline utilities
│   ├── inspect_dataset2.py     # Dataset 2 inspection
│   ├── compare_datasets.py     # Cross-dataset comparison
│   └── list_columns.py         # Column listing utility
│
├── logs/                       # Training logs
│   ├── log_dataset1.txt
│   ├── log_dataset2.txt
│   ├── log_calibration.txt
│   └── ... (other training logs)
│
└── tests/                      # Test scripts
    ├── test_gpu.py
    ├── test_gpu_comprehensive.py
    └── test_shap_explainer.py
```

## Model Performance

### Dataset 1 (Telco - Primary Model for App)
- **Accuracy**: 75.0%
- **Recall**: 81.0% (primary metric)
- **F1**: 63.3%
- **Brier Score**: 0.1366
- **Features**: 39 (including 6 engineered features)

### Dataset 2 (Indian Telecom - Cross-Dataset Validation)
- **Accuracy**: 93.3%
- **Recall**: 67.6%
- **F1**: 62.8%
- **Brier Score**: 0.0446
- **Features**: 100 (top features selected from 315)

## Installation

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn shap joblib
```

## Usage

### Run the App
```bash
streamlit run app.py
```

### Train Models
```bash
# Dataset 1 (primary)
python scripts/run_dataset1_calibrated_cv.py

# Dataset 2 (with feature selection)
python scripts/run_dataset2_feature_selected.py
```

### Test GPU Setup
```bash
python tests/test_gpu_comprehensive.py
```

## App Features

1. **Single Customer Prediction**: Manual input with SHAP explanations and natural language output
2. **Batch Upload**: CSV upload with risk-ranked downloadable report
3. **What-If Simulator**: Explore how changing factors affects churn risk
4. **Dashboard**: Persistent SQLite storage with:
   - Prediction history
   - Risk queue management (pending → contacted → resolved)
   - Batch upload history
   - Dashboard statistics

## Key Methodology

- **Class Imbalance**: SMOTE applied INSIDE CV folds (leakage-safe)
- **Feature Engineering**: Behavioral change features for time-series data
- **Model Architecture**: Stacked ensemble (XGBoost + LightGBM + Random Forest)
- **Probability Calibration**: CV-based calibration for reliable probabilities
- **Explainability**: SHAP + natural language explanations
- **GPU Acceleration**: XGBoost on NVIDIA RTX 5050 (1.90x speedup)

## Paper Structure

See `PAPER_STRUCTURE.md` for complete paper outline including:
- Abstract
- Methodology
- Results
- Discussion
- Demo application description

## Notes

- **Primary metric**: Recall on churn class (catching real churners)
- **Dataset 2 accuracy**: 93.3% is misleading due to 91.9% trivial baseline
- **Honest evaluation**: No leakage, proper train/test split, SMOTE inside CV
- **Overfitting check**: Train vs test F1 gap < 0.10 threshold
