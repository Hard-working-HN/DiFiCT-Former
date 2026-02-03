import pandas as pd
import numpy as np

IN_PATH = r"Total_Energy_GDP_Pop_with_Total_Energy.csv"
OUT_PATH = r"EI_4types_by_Code.csv"

START_YEAR = 2013
END_YEAR = 2022
BASE_YEAR = 2022

ALPHA = 0.9

df = pd.read_csv(IN_PATH, encoding="utf-8-sig")

df["Code"] = df["Code"].astype(str)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["GDP"] = pd.to_numeric(df["GDP"], errors="coerce")
df["Total_Energy"] = pd.to_numeric(df["Total_Energy"], errors="coerce")

base = df[(df["Year"] >= START_YEAR) & (df["Year"] <= END_YEAR)].copy()

cy = (
    base.groupby(["Code", "Year"], as_index=False)
        .agg(GDP_sum=("GDP", "sum"),
             E_sum=("Total_Energy", "sum"))
)

cy["EI"] = np.where(cy["GDP_sum"] > 0, cy["E_sum"] / cy["GDP_sum"], np.nan)

ei_mean = (
    cy.groupby("Code", as_index=False)["EI"]
      .mean()
      .rename(columns={"EI": f"EI_mean_{START_YEAR}_{END_YEAR}"})
)

tmp = cy.groupby("Code", as_index=False).agg(
    E_total=("E_sum", "sum"),
    GDP_total=("GDP_sum", "sum")
)
tmp[f"EI_wGDP_{START_YEAR}_{END_YEAR}"] = np.where(
    tmp["GDP_total"] > 0, tmp["E_total"] / tmp["GDP_total"], np.nan
)
ei_wgdp = tmp[["Code", f"EI_wGDP_{START_YEAR}_{END_YEAR}"]]

cy_end = cy[cy["Year"] == END_YEAR].copy()
ei_end = (
    cy_end.groupby("Code", as_index=False)["EI"]
          .mean()
          .rename(columns={"EI": f"EI_{END_YEAR}"})
)

cy["w_raw"] = ALPHA ** (BASE_YEAR - cy["Year"])
w_sum = cy.groupby("Code")["w_raw"].transform("sum")
cy["w"] = np.where(w_sum > 0, cy["w_raw"] / w_sum, np.nan)

cy["EI_w"] = cy["EI"] * cy["w"]

ei_exp = (
    cy.groupby("Code", as_index=False)["EI_w"]
      .sum()
      .rename(columns={"EI_w": f"EI_exp_{BASE_YEAR}_alpha{ALPHA}"})
)

out = (
    ei_mean.merge(ei_wgdp, on="Code", how="outer")
           .merge(ei_end, on="Code", how="outer")
           .merge(ei_exp, on="Code", how="outer")
)

out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
print("Done:", OUT_PATH)
print(out.head())

example_years = pd.DataFrame({"Year": list(range(END_YEAR, START_YEAR - 1, -1))})
example_years["w_raw"] = ALPHA ** (BASE_YEAR - example_years["Year"])
print("\nRaw weights example:")
print(example_years)
