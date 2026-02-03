# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path

in_total = Path("Total_Energy.csv")
in_factors = Path("EF_NCV.csv")
out_path = Path("Total_Energy_CO2_Except_ELEC.csv")

MAKE_SUM_CO2 = True
ON_MISSING_ENERGY = "skip"

TRY_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb2312")
DEFAULT_OF = 1.0


def read_csv_smart(p: Path):
    last_err = None
    for enc in TRY_ENCODINGS:
        try:
            df = pd.read_csv(p, encoding=enc)
            print(f"[READ] {p} OK | encoding={enc} | shape={df.shape}")
            return df, enc
        except Exception as e:
            last_err = e
    raise RuntimeError(f"[READ-FAIL] Cannot read {p}. Last error: {last_err}")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_numeric_safe(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")


def compute_co2_series(activity: pd.Series, unit: int, ef: float, ncv: float | None, of: float = DEFAULT_OF):
    a = to_numeric_safe(activity)
    if unit == 1:
        if ncv is None or np.isnan(ncv):
            raise ValueError("Unit=1 requires NCV (kJ/kg), but NCV is missing.")
        return a * ncv * ef * 0.01 * of
    if unit == 2:
        if ncv is None or np.isnan(ncv):
            raise ValueError("Unit=2 requires NCV (kJ/m³), but NCV is missing.")
        return a * ncv * ef * 0.1 * of
    if unit == 3:
        return a * 10.0 * ef * of
    raise ValueError(f"Unknown Unit={unit}. Only 1/2/3 are supported.")


def main():
    total_df, enc_total = read_csv_smart(in_total)
    total_df = normalize_cols(total_df)

    factors_df, _ = read_csv_smart(in_factors)
    factors_df = normalize_cols(factors_df)

    if total_df.shape[1] < 8:
        raise ValueError("[ERR] Total_Energy has fewer than 8 columns; cannot locate energy columns (from column 8).")

    required_cols = {"Energy", "EF", "NCV", "Unit"}
    if not required_cols.issubset(set(factors_df.columns)):
        missing = required_cols - set(factors_df.columns)
        raise ValueError(f"[ERR] EF_NCV.csv is missing required columns: {missing}")

    energy_cols = list(total_df.columns[7:])
    print(f"[INFO] Energy columns detected: {len(energy_cols)}. Example: {energy_cols[:6]} ...")

    factors_df["Energy_norm"] = factors_df["Energy"].astype(str).str.strip()
    if factors_df["Energy_norm"].duplicated().any():
        dups = factors_df[factors_df["Energy_norm"].duplicated(keep=False)].sort_values("Energy_norm")
        print("[WARN] Duplicate Energy names found in factor table; the first occurrence will be used:")
        print(dups[["Energy", "EF", "NCV", "Unit"]].to_string(index=False))
        factors_df = factors_df.drop_duplicates(subset=["Energy_norm"], keep="first")

    factors_df = factors_df.set_index("Energy_norm")

    co2_cols = []
    for col in energy_cols:
        energy_name = str(col).strip()
        co2_col = f"{col}_CO2"

        if energy_name not in factors_df.index:
            msg = f"[MISS] Energy column not found in factor table: '{energy_name}'."
            if ON_MISSING_ENERGY == "error":
                raise KeyError(msg)
            print(msg + " Skipping this column.")
            continue

        row = factors_df.loc[energy_name]
        ef = pd.to_numeric(row["EF"], errors="coerce")
        ncv = pd.to_numeric(row["NCV"], errors="coerce")
        try:
            unit = int(pd.to_numeric(row["Unit"], errors="raise"))
        except Exception:
            raise ValueError(f"[ERR] Unit for Energy='{energy_name}' cannot be parsed as int: {row['Unit']}")

        print(f"\n[CONVERT] Energy='{energy_name}' | EF={ef} tCO2/TJ | NCV={ncv} | Unit={unit}")

        if unit in (1, 2) and pd.isna(ncv):
            print(f"[WARN] Energy='{energy_name}' needs NCV for Unit={unit}, but NCV is empty; results will be NaN.")

        before_nonnull = int(total_df[col].notna().sum())
        print(f"  [STAT] Non-null activity cells = {before_nonnull}")

        try:
            co2_series = compute_co2_series(
                total_df[col],
                unit=unit,
                ef=float(ef),
                ncv=(None if pd.isna(ncv) else float(ncv)),
            )
        except Exception as e:
            print(f"  [ERROR] Conversion failed: {e}")
            continue

        after_nonnull = int(co2_series.notna().sum())
        print(f"  [STAT] Non-null CO2 cells = {after_nonnull} (delta={after_nonnull - before_nonnull})")

        total_df[co2_col] = co2_series
        co2_cols.append(co2_col)
        print(f"  [DONE] Added column: {co2_col}")

    if not co2_cols:
        print("[WARN] No CO2 columns were generated. Check if energy column names match the factor table.")

    if MAKE_SUM_CO2 and co2_cols:
        total_df["Total_CO2_sum"] = total_df[co2_cols].sum(axis=1, skipna=True)
        print(f"\n[SUM] Added 'Total_CO2_sum' by summing {len(co2_cols)} *_CO2 columns.")

    total_df.to_csv(out_path, index=False, encoding=enc_total)
    print("\n========== Finished ==========")
    print(f"[OUT] Written to: {out_path.resolve()}")
    print(f"[SHAPE] Rows={total_df.shape[0]}, Cols={total_df.shape[1]}")
    if co2_cols:
        print(f"[INFO] CO2 columns added: {len(co2_cols)}. Example: {co2_cols[:6]} ...")


if __name__ == "__main__":
    main()
