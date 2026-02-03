import pandas as pd
import numpy as np

IN_PATH = r"Subdivision_Energy_Standard_Coal.csv"
OUT_BASELINE = r"Baseline_EnergyMix_by_CodeSector_expWeight.csv"
OUT_DETAIL = r"EnergyMix_by_CodeYearSector_expWeight_detail.csv"

START_YEAR = 2013
END_YEAR = 2022
BASE_YEAR = 2022
ALPHA = 0.9

ENERGY_COLS_CANON = [
    "Coal and coal products",
    "Petroleum and petroleum products",
    "Natural gas",
    "Heat",
    "Electricity",
]

df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

sector_col = None
for c in ["Sectore", "Sector", "SECTOR", "sectore", "sector"]:
    if c in df.columns:
        sector_col = c
        break
if sector_col is None:
    raise ValueError("Sector column not found. Please confirm the column name is 'Sector' or 'Sectore'.")

def norm_col(s: str) -> str:
    return "".join(str(s).strip().lower().split())

col_map = {norm_col(c): c for c in df.columns}

energy_cols = []
missing_energy = []
for canon in ENERGY_COLS_CANON:
    key = norm_col(canon)
    if key in col_map:
        energy_cols.append(col_map[key])
    else:
        missing_energy.append(canon)

if missing_energy:
    raise ValueError(
        f"Energy columns not found: {missing_energy}\n"
        "Please check whether the CSV column names match the expected names."
    )

total_col = None
for c in ["Total_Energy", "Total energy", "TotalEnergy", "TOTAL_ENERGY"]:
    if c in df.columns:
        total_col = c
        break
if total_col is None:
    total_col = df.columns[-1]
    print(f"[Warn] 'Total_Energy' column not found. Using the last column as total energy: {total_col}")

need_cols = {"Code", "Year", sector_col, total_col, *energy_cols}
missing = need_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Code"] = df["Code"].astype(str)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

for c in energy_cols + [total_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)].copy()

agg_cols = {c: "sum" for c in energy_cols + [total_col]}
g = (
    df.groupby(["Code", "Year", sector_col], as_index=False)
      .agg(agg_cols)
)

den = g[total_col].replace(0, np.nan)
for c in energy_cols:
    g[c + "_share_year"] = g[c] / den

share_cols = [c + "_share_year" for c in energy_cols]

w = g[["Code", "Year"]].drop_duplicates().copy()
w["w_raw"] = ALPHA ** (BASE_YEAR - w["Year"])

w_sum = w.groupby("Code")["w_raw"].transform("sum")
w["w"] = np.where(w_sum > 0, w["w_raw"] / w_sum, np.nan)

g = g.merge(w[["Code", "Year", "w"]], on=["Code", "Year"], how="left")

def weighted_avg(group: pd.DataFrame) -> pd.Series:
    out = {"Code": group["Code"].iloc[0], sector_col: group[sector_col].iloc[0]}
    weights = group["w"].to_numpy(dtype=float)

    for sc in share_cols:
        x = group[sc].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
        out_name = sc.replace("_share_year", "_share_baseline")
        out[out_name] = float(np.average(x[m], weights=weights[m])) if m.any() else np.nan

    return pd.Series(out)

baseline = (
    g.groupby(["Code", sector_col], as_index=False)
     .apply(weighted_avg)
)

baseline_cols = [c.replace("_share_year", "_share_baseline") for c in share_cols]
baseline["share_sum_check"] = baseline[baseline_cols].sum(axis=1)

baseline.to_csv(OUT_BASELINE, index=False, encoding="utf-8-sig")

detail_cols = ["Code", "Year", sector_col, total_col, "w"] + share_cols
g[detail_cols].to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

print("Done!")
print("Baseline output:", OUT_BASELINE)
print("Detail output:", OUT_DETAIL)
print(baseline.head())
print("share_sum_check statistics:")
print(baseline["share_sum_check"].describe())

example_years = pd.DataFrame({"Year": list(range(END_YEAR, START_YEAR - 1, -1))})
example_years["w_raw"] = ALPHA ** (BASE_YEAR - example_years["Year"])
print("\nRaw weights example:")
print(example_years)
