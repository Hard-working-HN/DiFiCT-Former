import pandas as pd
import numpy as np

IN_PATH = r"Subdivision_Energy_Standard_Coal.csv"
OUT_BASELINE_WIDE = r"Baseline_SectorShare_by_Code_expWeight.csv"
OUT_DETAIL = r"SectorShare_by_CodeYear_expWeight_detail.csv"

START_YEAR = 2013
END_YEAR = 2022
BASE_YEAR = 2022
ALPHA = 0.9

df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

sector_col = None
for c in ["Sector", "Sectore", "SECTOR", "sectore", "sector"]:
    if c in df.columns:
        sector_col = c
        break
if sector_col is None:
    raise ValueError("Sector column not found. Please confirm the column name is 'Sector' or 'Sectore'.")

need_cols = {"Code", "Year", "Total_Energy", sector_col}
missing = need_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Code"] = df["Code"].astype(str)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Total_Energy"] = pd.to_numeric(df["Total_Energy"], errors="coerce").fillna(0)

df = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)].copy()

g = (
    df.groupby(["Code", "Year", sector_col], as_index=False)["Total_Energy"]
      .sum()
)

tot = (
    g.groupby(["Code", "Year"], as_index=False)["Total_Energy"]
     .sum()
     .rename(columns={"Total_Energy": "Total_Energy_All"})
)

g = g.merge(tot, on=["Code", "Year"], how="left")

g["share_year"] = np.where(
    g["Total_Energy_All"] > 0,
    g["Total_Energy"] / g["Total_Energy_All"],
    np.nan
)

w = tot.copy()
w["w_raw"] = ALPHA ** (BASE_YEAR - w["Year"])

w_sum = w.groupby("Code")["w_raw"].transform("sum")
w["w"] = np.where(w_sum > 0, w["w_raw"] / w_sum, np.nan)

g = g.merge(w[["Code", "Year", "w"]], on=["Code", "Year"], how="left")

g["share_weighted"] = g["share_year"] * g["w"]

baseline_long = (
    g.groupby(["Code", sector_col], as_index=False)["share_weighted"]
     .sum()
     .rename(columns={"share_weighted": "Baseline_SectorShare"})
)

baseline_wide = (
    baseline_long.pivot(index="Code", columns=sector_col, values="Baseline_SectorShare")
                 .reset_index()
)

sector_cols = [c for c in baseline_wide.columns if c != "Code"]
baseline_wide["share_sum_check"] = baseline_wide[sector_cols].sum(axis=1)

baseline_wide.to_csv(OUT_BASELINE_WIDE, index=False, encoding="utf-8-sig")

detail_cols = ["Code", "Year", sector_col, "Total_Energy", "Total_Energy_All", "share_year", "w", "share_weighted"]
g[detail_cols].to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

print("Done!")
print("Baseline sector structure (wide):", OUT_BASELINE_WIDE)
print("Detail (Code-Year-Sector):", OUT_DETAIL)
print(baseline_wide.head())
print("share_sum_check statistics:")
print(baseline_wide["share_sum_check"].describe())

example_years = pd.DataFrame({"Year": list(range(END_YEAR, START_YEAR - 1, -1))})
example_years["w_raw"] = ALPHA ** (BASE_YEAR - example_years["Year"])
print("\nRaw weights example:")
print(example_years)
