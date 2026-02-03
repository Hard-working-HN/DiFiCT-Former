# -*- coding: utf-8 -*-

import pandas as pd

INPUT_FILE = r"Data_Prefecture_Total_Missing.xlsx"
OUTPUT_FILE = r"Data_Prefecture_Total_Missing_cleaned.xlsx"

df = pd.read_excel(INPUT_FILE)

code_col = df.columns[0]
value_col = df.columns[1]

print("[Info] Code column:", code_col)
print("[Info] Numeric column to clean:", value_col)

def handle_outliers(group: pd.DataFrame) -> pd.DataFrame:
    vals = pd.to_numeric(group[value_col], errors="coerce")

    non_na = vals.dropna()
    if len(non_na) < 3:
        return group

    q1 = non_na.quantile(0.25)
    q3 = non_na.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr

    is_outlier = vals > upper

    normal_vals = vals[~is_outlier]
    if normal_vals.dropna().empty:
        mean_val = non_na.mean()
    else:
        mean_val = normal_vals.mean()

    group.loc[is_outlier, value_col] = mean_val

    n_out = int(is_outlier.sum())
    if n_out > 0:
        code_value = group[code_col].iloc[0]
        print(f"[Handle] Code={code_value} found {n_out} outlier(s), replaced with group mean {mean_val:.4f}")

    return group

df_clean = df.groupby(code_col, group_keys=False).apply(handle_outliers)

df_clean.to_excel(OUTPUT_FILE, index=False)
print(f"[Done] Outlier handling completed. Saved to: {OUTPUT_FILE}")
