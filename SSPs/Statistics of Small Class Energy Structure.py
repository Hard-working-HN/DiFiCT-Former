from pathlib import Path
import pandas as pd

def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\t")
    return df

def calc_within_big_shares_2022(
    in_csv: str = "Origin_Total_Energy.csv",
    out_csv: str = "Origin_Total_Energy_2022_within_big_share.csv",
):
    in_path = Path(in_csv)
    df = read_csv_auto(in_path)

    for c in ["Code", "Year", "Sector"]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    if df.shape[1] < 36:
        raise ValueError(f"Not enough columns: current={df.shape[1]}, but need at least 36 columns.")

    coal_cols = list(df.columns[7:18])
    petro_cols = list(df.columns[18:32])
    gas_cols = list(df.columns[32:34])
    heat_cols = list(df.columns[34:35])
    elec_cols = list(df.columns[35:36])

    big_map = {
        "Coal and coal products_Energy": coal_cols,
        "Petroleum and petroleum products_Energy": petro_cols,
        "Natural gas_Energy": gas_cols,
        "Heat_Energy": heat_cols,
        "Electricity_Energy": elec_cols,
    }

    all_energy_cols = coal_cols + petro_cols + gas_cols + heat_cols + elec_cols

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    d2022 = df[df["Year"] == 2022].copy()
    if d2022.empty:
        raise ValueError("No data found after filtering Year == 2022.")

    d2022[all_energy_cols] = d2022[all_energy_cols].apply(pd.to_numeric, errors="coerce")

    g = (
        d2022.groupby(["Code", "Sector"], as_index=False)[all_energy_cols]
            .sum(min_count=1)
    )
    g.insert(2, "Year", 2022)

    out = g[["Code", "Sector", "Year"]].copy()

    for big_name, cols in big_map.items():
        big_sum = g[cols].sum(axis=1)
        denom = big_sum.replace({0: pd.NA})

        shares = g[cols].div(denom, axis=0)

        rename = {c: f"{c}_share_in_{big_name}" for c in cols}
        shares = shares.rename(columns=rename)

        out = pd.concat([out, shares], axis=1)

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] Output saved: {Path(out_csv).resolve()} | rows={len(out)}")

if __name__ == "__main__":
    calc_within_big_shares_2022(
        in_csv="Origin_Total_Energy.csv",
        out_csv="Origin_Total_Energy_2022_within_big_share.csv",
    )
