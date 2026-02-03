import pandas as pd
import numpy as np

IN_PATH = r"Total_Energy_GDP_Pop_with_Total_Energy.csv"

OUT_RATES = r"EI_change_rates_2013_2022_byCode_and_TOTAL.csv"
OUT_TOTAL_SERIES = r"EI_series_total_2013_2022.csv"

START_YEAR = 2013
END_YEAR = 2022

df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

required_cols = {"Code", "Year", "GDP", "Total_Energy"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Code"] = df["Code"].astype(str)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["GDP"] = pd.to_numeric(df["GDP"], errors="coerce")
df["Total_Energy"] = pd.to_numeric(df["Total_Energy"], errors="coerce")

code_order = df["Code"].drop_duplicates().tolist()

base = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)].copy()
base = base.dropna(subset=["Code", "Year", "GDP", "Total_Energy"])

cy = (
    base.groupby(["Code", "Year"], as_index=False)
        .agg(GDP_sum=("GDP", "sum"),
             E_sum=("Total_Energy", "sum"))
)

cy["EI"] = np.where(cy["GDP_sum"] > 0, cy["E_sum"] / cy["GDP_sum"], np.nan)

def calc_rates_one_code(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("Year").copy()
    g = g.dropna(subset=["EI"])
    out = {"Code": g["Code"].iloc[0] if len(g) else np.nan}

    if len(g) < 2:
        out.update({
            "n_years_used": len(g),
            "first_year": int(g["Year"].min()) if len(g) else np.nan,
            "last_year": int(g["Year"].max()) if len(g) else np.nan,
            "EI_first": float(g["EI"].iloc[0]) if len(g) else np.nan,
            "EI_last": float(g["EI"].iloc[-1]) if len(g) else np.nan,
            "YoY_mean": np.nan,
            "YoY_median": np.nan,
            "CAGR": np.nan,
            "log_slope": np.nan
        })
        return pd.Series(out)

    years = g["Year"].to_numpy(dtype=float)
    ei = g["EI"].to_numpy(dtype=float)

    yoy = pd.Series(ei).pct_change().to_numpy()
    out["YoY_mean"] = float(np.nanmean(yoy))
    out["YoY_median"] = float(np.nanmedian(yoy))

    first_year = int(years[0])
    last_year = int(years[-1])
    dt = last_year - first_year

    out["first_year"] = first_year
    out["last_year"] = last_year
    out["EI_first"] = float(ei[0])
    out["EI_last"] = float(ei[-1])
    out["n_years_used"] = int(len(g))

    if dt > 0 and ei[0] > 0 and ei[-1] > 0:
        out["CAGR"] = float((ei[-1] / ei[0]) ** (1.0 / dt) - 1.0)
    else:
        out["CAGR"] = np.nan

    mask = (ei > 0) & np.isfinite(ei) & np.isfinite(years)
    if int(mask.sum()) >= 2:
        b = float(np.polyfit(years[mask], np.log(ei[mask]), 1)[0])
        out["log_slope"] = b
    else:
        out["log_slope"] = np.nan

    return pd.Series(out)

rates_by_code = cy.groupby("Code", as_index=False).apply(calc_rates_one_code)

cat = pd.CategoricalDtype(categories=code_order, ordered=True)
rates_by_code["Code"] = rates_by_code["Code"].astype(cat)
rates_by_code = rates_by_code.sort_values("Code").reset_index(drop=True)
rates_by_code["Code"] = rates_by_code["Code"].astype(str)

total_year = (
    cy.groupby("Year", as_index=False)
      .agg(GDP_sum=("GDP_sum", "sum"),
           E_sum=("E_sum", "sum"))
)
total_year["Code"] = "TOTAL"
total_year["EI"] = np.where(total_year["GDP_sum"] > 0, total_year["E_sum"] / total_year["GDP_sum"], np.nan)

total_rates = calc_rates_one_code(total_year[["Code", "Year", "EI"]].copy()).to_frame().T

total_year[["Year", "GDP_sum", "E_sum", "EI"]].to_csv(OUT_TOTAL_SERIES, index=False, encoding="utf-8-sig")

out = pd.concat([rates_by_code, total_rates], ignore_index=True)

out["__is_total"] = (out["Code"] == "TOTAL").astype(int)
out = out.sort_values(["__is_total", "Code"]).drop(columns="__is_total").reset_index(drop=True)

out.to_csv(OUT_RATES, index=False, encoding="utf-8-sig")

print("Done!")
print("Change-rate results saved to:", OUT_RATES)
print("TOTAL annual series saved to:", OUT_TOTAL_SERIES)
print(out.head())
print("\nTOTAL row:")
print(out[out["Code"] == "TOTAL"])
