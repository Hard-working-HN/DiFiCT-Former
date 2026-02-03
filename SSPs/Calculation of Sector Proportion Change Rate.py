import os
import numpy as np
import pandas as pd

IN_PATH = r"Subdivision_Energy_Standard_Coal.csv"
OUT_DIR = r"./Sector_Energy_Allocation_by_SSP"
os.makedirs(OUT_DIR, exist_ok=True)

FIT_START = 2013
FIT_END = 2022

EPS = 1e-12
MIN_POINTS = 10

df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

required_cols = {"Code", "Year", "Sector", "Total_Energy"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"{os.path.basename(IN_PATH)} is missing required columns: {missing}")

df["Code"] = df["Code"].astype(str)
df["Sector"] = df["Sector"].astype(str)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Total_Energy"] = pd.to_numeric(df["Total_Energy"], errors="coerce")

df = df.dropna(subset=["Year", "Total_Energy", "Code", "Sector"]).copy()
df["Year"] = df["Year"].astype(int)

df = (
    df.groupby(["Code", "Year", "Sector"], as_index=False)
      .agg(Total_Energy=("Total_Energy", "sum"))
)

total_by_code_year = (
    df.groupby(["Code", "Year"], as_index=False)
      .agg(Total_Energy_allSectors=("Total_Energy", "sum"))
)

df_share = df.merge(total_by_code_year, on=["Code", "Year"], how="left")
df_share["share"] = df_share["Total_Energy"] / df_share["Total_Energy_allSectors"]

out1 = os.path.join(OUT_DIR, "sector_share_byCodeYear.csv")
df_share.sort_values(["Code", "Year", "Sector"]).to_csv(out1, index=False, encoding="utf-8-sig")
print("[OK] Saved:", out1)

year_sector_sum = (
    df.groupby(["Year", "Sector"], as_index=False)
      .agg(Total_Energy_sum=("Total_Energy", "sum"))
)

year_total_sum = (
    df.groupby(["Year"], as_index=False)
      .agg(Total_Energy_sum_allSectors=("Total_Energy", "sum"))
)

year_share = year_sector_sum.merge(year_total_sum, on="Year", how="left")
year_share["share"] = year_share["Total_Energy_sum"] / year_share["Total_Energy_sum_allSectors"]

out2 = os.path.join(OUT_DIR, "sector_share_byYear.csv")
year_share.sort_values(["Year", "Sector"]).to_csv(out2, index=False, encoding="utf-8-sig")
print("[OK] Saved:", out2)

fit_df = df_share[(df_share["Year"] >= FIT_START) & (df_share["Year"] <= FIT_END)].copy()
fit_df["share_clipped"] = fit_df["share"].clip(lower=EPS)
fit_df["ln_share"] = np.log(fit_df["share_clipped"])

rows = []
for (code, sector), grp in fit_df.groupby(["Code", "Sector"]):
    grp = grp.sort_values("Year")
    grp = grp[np.isfinite(grp["ln_share"])].copy()

    if len(grp) < MIN_POINTS:
        rows.append([
            code,
            sector,
            len(grp),
            int(grp["Year"].min()) if len(grp) else np.nan,
            int(grp["Year"].max()) if len(grp) else np.nan,
            np.nan,
            np.nan,
            np.nan,
        ])
        continue

    x = grp["Year"].to_numpy(dtype=float)
    y = grp["ln_share"].to_numpy(dtype=float)

    b, a = np.polyfit(x, y, deg=1)
    decline_rate_cont = -b
    rate_discrete = float(np.expm1(b))

    rows.append([
        code,
        sector,
        len(grp),
        int(x.min()),
        int(x.max()),
        float(b),
        float(decline_rate_cont),
        rate_discrete,
    ])

slope_df = pd.DataFrame(
    rows,
    columns=[
        "Code",
        "Sector",
        "n_used",
        "year_min",
        "year_max",
        "log_slope",
        "decline_rate_continuous",
        "rate_discrete",
    ],
)

out3 = os.path.join(OUT_DIR, f"sector_share_log_slope_{FIT_START}_{FIT_END}_byCodeSector.csv")
slope_df.sort_values(["Code", "Sector"]).to_csv(out3, index=False, encoding="utf-8-sig")
print("[OK] Saved:", out3)

fit_year = year_share[(year_share["Year"] >= FIT_START) & (year_share["Year"] <= FIT_END)].copy()
fit_year["share_clipped"] = fit_year["share"].clip(lower=EPS)
fit_year["ln_share"] = np.log(fit_year["share_clipped"])

rows2 = []
for sector, grp in fit_year.groupby("Sector"):
    grp = grp.sort_values("Year")
    grp = grp[np.isfinite(grp["ln_share"])].copy()

    if len(grp) < MIN_POINTS:
        rows2.append([
            sector,
            len(grp),
            int(grp["Year"].min()) if len(grp) else np.nan,
            int(grp["Year"].max()) if len(grp) else np.nan,
            np.nan,
            np.nan,
            np.nan,
        ])
        continue

    x = grp["Year"].to_numpy(dtype=float)
    y = grp["ln_share"].to_numpy(dtype=float)

    b, a = np.polyfit(x, y, deg=1)
    decline_rate_cont = -b
    rate_discrete = float(np.expm1(b))

    rows2.append([
        sector,
        len(grp),
        int(x.min()),
        int(x.max()),
        float(b),
        float(decline_rate_cont),
        rate_discrete,
    ])

sector_total_slope = (
    pd.DataFrame(
        rows2,
        columns=[
            "Sector",
            "n_used",
            "year_min",
            "year_max",
            "log_slope",
            "decline_rate_continuous",
            "rate_discrete",
        ],
    )
    .sort_values("Sector")
    .reset_index(drop=True)
)

out4 = os.path.join(OUT_DIR, f"sector_total_share_log_slope_{FIT_START}_{FIT_END}.csv")
sector_total_slope.to_csv(out4, index=False, encoding="utf-8-sig")
print("[OK] Saved:", out4)

print("\nAll done! Outputs in:", OUT_DIR)
