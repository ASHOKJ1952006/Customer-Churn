import pandas as pd
pd.set_option('display.max_rows', None)

df2 = pd.read_csv("telecom_churn_data.csv")
print("Full column list:")
for i, col in enumerate(df2.columns):
    print(i, col)