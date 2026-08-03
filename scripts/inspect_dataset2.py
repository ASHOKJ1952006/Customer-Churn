"""
inspect_dataset2.py — Run this FIRST after downloading the second dataset
(https://www.kaggle.com/datasets/shivam131019/telecom-churn-dataset).

We don't yet know this dataset's exact column names, types, or target column,
so this script just inspects it. Once you run this and share the output,
the actual cleaning/encoding script (run_dataset2.py) can be written correctly
instead of guessing at column names that might not exist.

IMPORTANT: update CSV_FILENAME below to match whatever file(s) you actually
downloaded and unzipped — Kaggle datasets sometimes come as a single CSV,
sometimes as multiple files (e.g., separate train/test, or separate files
per month for this specific dataset).
"""

import pandas as pd
import os

CSV_FILENAME = "telecom_churn_data.csv"  # <-- change this

print("Files in current folder:")
for f in os.listdir('.'):
    if f.endswith('.csv'):
        print(" -", f)

print(f"\nAttempting to load: {CSV_FILENAME}")
df2 = pd.read_csv(CSV_FILENAME)

print("\nShape:", df2.shape)
print("\nColumn names and dtypes:")
print(df2.dtypes)

print("\nFirst 5 rows:")
print(df2.head())

print("\nMissing values per column:")
print(df2.isnull().sum())

print("\nLikely target column candidates (look for churn/label/target in the name):")
for col in df2.columns:
    if any(kw in col.lower() for kw in ['churn', 'target', 'label']):
        print(" -", col, "| unique values:", df2[col].unique()[:10])