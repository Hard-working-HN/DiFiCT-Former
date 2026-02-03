import pandas as pd

df = pd.read_csv("Data_Prefecture_Total.csv")

df_subset = df.iloc[:, 4:]

missing_counts = df_subset.isna().sum()

print("Missing value counts per column:")
print(missing_counts)
