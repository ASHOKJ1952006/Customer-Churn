"""
compare_datasets.py — Run AFTER both run_dataset1.py and run_dataset2.py have completed.
Produces the side-by-side comparison table for your paper's generalizability section.
"""

import joblib
import pandas as pd

m1 = joblib.load('ds1_final_metrics.pkl')
m2 = joblib.load('ds2_final_metrics.pkl')

comparison = pd.DataFrame([
    {'Dataset': 'Telco Customer Churn (Dataset 1)', **m1},
    {'Dataset': 'High-Value Customer Churn (Dataset 2)', **m2},
])

print("=== Final Model Performance: Cross-Dataset Comparison ===")
print(comparison.to_string(index=False))

overfit1 = joblib.load('ds1_overfit_df.pkl')
overfit2 = joblib.load('ds2_overfit_df.pkl')

print("\n=== Dataset 1 — Overfitting Check ===")
print(overfit1.to_string(index=False))

print("\n=== Dataset 2 — Overfitting Check ===")
print(overfit2.to_string(index=False))

eff1 = joblib.load('ds1_efficiency_df.pkl')
eff2 = joblib.load('ds2_efficiency_df.pkl')

print("\n=== Dataset 1 — Efficiency ===")
print(eff1.to_string(index=False))

print("\n=== Dataset 2 — Efficiency ===")
print(eff2.to_string(index=False))

comparison.to_csv('final_cross_dataset_comparison.csv', index=False)
print("\nSaved: final_cross_dataset_comparison.csv")