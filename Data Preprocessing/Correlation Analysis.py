# -*- coding: utf-8 -*-

import pandas as pd
from scipy.stats import spearmanr, kendalltau

input_file = r"Corr_Analysis.xlsx"
output_file = r"Corr_Result.xlsx"

df = pd.read_excel(input_file)

df.columns = [str(c).strip() for c in df.columns]
print("Columns read:", df.columns.tolist())

required = {"Year", "Light_Data", "Energy"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Light_Data"] = pd.to_numeric(df["Light_Data"], errors="coerce")
df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")

df = df.dropna(subset=["Light_Data", "Energy"])

def calc_corr(x, y):
    sp_r, sp_p = spearmanr(x, y, nan_policy="omit")
    ke_r, ke_p = kendalltau(x, y, nan_policy="omit")
    return sp_r, sp_p, ke_r, ke_p

sp_r, sp_p, ke_r, ke_p = calc_corr(df["Light_Data"], df["Energy"])

results = [{
    "Year": "total",
    "Spearman_r": sp_r,
    "Spearman_p": sp_p,
    "Kendall_tau": ke_r,
    "Kendall_p": ke_p,
}]

for year, group in df.groupby("Year", dropna=True):
    sp_r, sp_p, ke_r, ke_p = calc_corr(group["Light_Data"], group["Energy"])
    results.append({
        "Year": int(year) if pd.notna(year) else year,
        "Spearman_r": sp_r,
        "Spearman_p": sp_p,
        "Kendall_tau": ke_r,
        "Kendall_p": ke_p,
    })

out_df = pd.DataFrame(results)
out_df.to_excel(output_file, index=False)

print("Correlation analysis finished. Results saved to:", output_file)
