import os
import numpy as np
import pandas as pd

IN_PATH = r"Subdivision_Energy_Standard_Coal.csv"
OUT_DIR = r"./Sector_Energy_Structure"
os.makedirs(OUT_DIR, exist_ok=True)

FIT_START = 2013
FIT_END = 2022

ZERO_YEARS_THRESHOLD = 7
MIN_NONZERO_POINTS = 2
EPS = 1e-12
ZERO_TOL = 0.0

ENERGY_COLS = [
    "Coal and coal products",
    "Petroleum and petroleum products",
    "Natural gas",
    "Heat",
    "Electricity",
]

df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

required = {"Code", "Year", "Sector", "Total_Energy"} | set(ENERGY_COLS)
missing = required - set(df.columns)
if missing:
    raise ValueError(f"{os.path.basename(IN_PATH)} is missing columns: {missing}")

df["Code"] = df["Code"].astype(str)
df["Sector"] = df["Sector"].astype(str)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

for c in ENERGY_COLS + ["Total_Energy"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Code", "Sector", "Year"]).copy()
df["Year"] = df["Year"].astype(int)

agg_dict = {c: "sum" for c in ENERGY_COLS + ["Total_Energy"]}
df = df.groupby(["Code", "Year", "Sector"], as_index=False).agg(agg_dict)

df["Total_Energy"] = df[ENERGY_COLS].sum(axis=1)

df_fit_base = df[(df["Year"] >= FIT_START) & (df["Year"] <= FIT_END)].copy()

long = df_fit_base.melt(
    id_vars=["Code", "Year", "Sector", "Total_Energy"],
    value_vars=ENERGY_COLS,
    var_name="EnergyType",
    value_name="Energy",
)

long["share"] = np.where(
    long["Total_Energy"] > 0,
    long["Energy"] / long["Total_Energy"],
    np.nan,
)

out1 = os.path.join(OUT_DIR, "energy_share_long_byCodeYearSector.csv")
long.sort_values(["Code", "Sector", "EnergyType", "Year"]).to_csv(out1, index=False, encoding="utf-8-sig")
print("[OK] Saved:", out1)

year_energy = (
    long.groupby(["Year", "EnergyType"], as_index=False)
        .agg(Energy_sum=("Energy", "sum"),
             Total_Energy_sum=("Total_Energy", "sum"))
)

year_energy["share"] = np.where(
    year_energy["Total_Energy_sum"] > 0,
    year_energy["Energy_sum"] / year_energy["Total_Energy_sum"],
    np.nan,
)

out2 = os.path.join(OUT_DIR, "energy_share_byYear.csv")
year_energy.sort_values(["Year", "EnergyType"]).to_csv(out2, index=False, encoding="utf-8-sig")
print("[OK] Saved:", out2)

rows = []
for (code, sector, etype), g in long.groupby(["Code", "Sector", "EnergyType"]):
    g = g.sort_values("Year").copy()

    share_series = g["share"]
    n_years_total = int(g["Year"].nunique())

    zero_mask = share_series.notna() & (share_series <= ZERO_TOL)
    n_zero_years = int(zero_mask.sum())

    if n_zero_years >= ZERO_YEARS_THRESHOLD:
        rows.append([
            code, sector, etype,
            n_years_total, n_zero_years, 0,
            np.nan, np.nan,
            0.0, 0.0, 0.0,
            f"mostly_zero>= {ZERO_YEARS_THRESHOLD} -> slope=0",
        ])
        continue

    g2 = g[(g["share"].notna()) & (g["share"] > ZERO_TOL)].copy()
    n_used = int(len(g2))

    if n_used < MIN_NONZERO_POINTS:
        rows.append([
            code, sector, etype,
            n_years_total, n_zero_years, n_used,
            int(g2["Year"].min()) if n_used else np.nan,
            int(g2["Year"].max()) if n_used else np.nan,
            np.nan, np.nan, np.nan,
            f"nonzero_points<{MIN_NONZERO_POINTS} -> no_fit",
        ])
        continue

    g2["share_clipped"] = g2["share"].clip(lower=EPS)
    g2["ln_share"] = np.log(g2["share_clipped"])

    x = g2["Year"].to_numpy(dtype=float)
    y = g2["ln_share"].to_numpy(dtype=float)

    b, a = np.polyfit(x, y, deg=1)
    decline_rate_cont = -float(b)
    rate_discrete = float(np.expm1(b))

    rows.append([
        code, sector, etype,
        n_years_total, n_zero_years, n_used,
        int(x.min()), int(x.max()),
        float(b), decline_rate_cont, rate_discrete,
        "drop_zero_then_fit",
    ])

slope_df = pd.DataFrame(
    rows,
    columns=[
        "Code", "Sector", "EnergyType",
        "n_years_total", "n_zero_years", "n_used",
        "year_min", "year_max",
        "log_slope", "decline_rate_continuous", "rate_discrete",
        "rule",
    ],
)

out3 = os.path.join(OUT_DIR, f"energy_share_log_slope_{FIT_START}_{FIT_END}_byCodeSectorEnergy.csv")
slope_df.sort_values(["Code", "Sector", "EnergyType"]).to_csv(out3, index=False, encoding="utf-8-sig")
print("[OK] Saved:", out3)

stats = []
for (sector, etype), g in slope_df.groupby(["Sector", "EnergyType"], dropna=False):
    n_total = int(len(g))
    g_valid = g[np.isfinite(g["log_slope"])].copy()
    n_valid = int(len(g_valid))

    stats.append([
        sector,
        etype,
        n_total,
        n_valid,
        float(g_valid["log_slope"].mean()) if n_valid else np.nan,
        float(g_valid["decline_rate_continuous"].mean()) if n_valid else np.nan,
        float(g_valid["rate_discrete"].mean()) if n_valid else np.nan,
        float(g_valid["log_slope"].abs().mean()) if n_valid else np.nan,
        float(g_valid["log_slope"].median()) if n_valid else np.nan,
    ])

mean_df = (
    pd.DataFrame(
        stats,
        columns=[
            "Sector",
            "EnergyType",
            "n_total",
            "n_valid_fit",
            "mean_log_slope",
            "mean_decline_rate_continuous",
            "mean_rate_discrete",
            "mean_abs_log_slope",
            "median_log_slope",
        ],
    )
    .sort_values(["Sector", "EnergyType"])
    .reset_index(drop=True)
)

out4 = os.path.join(OUT_DIR, f"sector_energy_share_slope_mean_{FIT_START}_{FIT_END}.csv")
mean_df.to_csv(out4, index=False, encoding="utf-8-sig")
print("[OK] Saved:", out4)

print("\nAll done! Outputs in:", OUT_DIR)
