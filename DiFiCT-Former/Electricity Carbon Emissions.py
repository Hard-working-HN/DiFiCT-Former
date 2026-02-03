# -*- coding: utf-8 -*-

import re
import pandas as pd
from pathlib import Path

total_path = Path("Total_Energy.csv")
ef_path = Path("Elec_EF_by_Province.csv")
out_path = Path("Total_Energy_ELEC_CO2.csv")

TRY_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb2312")


def read_csv_smart(p: Path):
    last = None
    for enc in TRY_ENCODINGS:
        try:
            df = pd.read_csv(p, encoding=enc)
            print(f"[READ] {p} OK | encoding={enc} | shape={df.shape}")
            return df, enc
        except Exception as e:
            last = e
    raise RuntimeError(f"[READ-FAIL] {p}: {last}")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def norm_id(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\s+", "", regex=True)
    s = s.str.replace("\u3000", "", regex=False)
    return s


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def find_electricity_columns(cols):
    cands = []
    for c in cols:
        raw = str(c)
        low = raw.lower()
        tokens = low.replace(" ", "")
        if ("electricity" in tokens) or ("电力" in raw):
            cands.append(raw)
        elif re.search(r"(kw\.?h|kwh)", tokens):
            if ("electric" in tokens) or ("电" in raw):
                cands.append(raw)
    return list(dict.fromkeys(cands))


def main():
    total_df, enc_total = read_csv_smart(total_path)
    total_df = normalize_cols(total_df)

    ef_df, _ = read_csv_smart(ef_path)
    ef_df = normalize_cols(ef_df)

    if total_df.shape[1] < 8:
        raise ValueError("[ERR] Total_Energy has fewer than 8 columns; cannot locate energy columns.")
    if total_df.shape[1] < 6:
        raise ValueError("[ERR] Total_Energy has fewer than 6 columns; cannot locate Province_ID (6th column).")

    province_col = total_df.columns[5]
    if str(province_col) != "Province_ID":
        print(f"[WARN] The 6th column name is not 'Province_ID' but '{province_col}'. Using it as the key anyway.")

    energy_cols = list(total_df.columns[7:])
    elec_cols = find_electricity_columns(energy_cols)
    if not elec_cols:
        raise ValueError("[ERR] No electricity columns found after the 8th column. Please check column names.")
    print(f"[INFO] Electricity columns identified: {elec_cols}")

    total_df[province_col] = norm_id(total_df[province_col])

    if ef_df.shape[1] < 2:
        raise ValueError("[ERR] Elec_EF_by_Province.csv must have at least 2 columns: Province_ID, EF")

    ef_pid_col = ef_df.columns[0]
    ef_val_col = ef_df.columns[1]
    ef_df = ef_df.rename(columns={ef_pid_col: province_col, ef_val_col: "EF_kg_per_kWh"})
    ef_df[province_col] = norm_id(ef_df[province_col])
    ef_df["EF_kg_per_kWh"] = to_num(ef_df["EF_kg_per_kWh"])

    print(f"[INFO] EF table fields: key='{province_col}', EF='EF_kg_per_kWh' (kgCO2/kWh)")
    print(
        f"[STAT] EF rows={len(ef_df)}, EF range=({ef_df['EF_kg_per_kWh'].min()}, {ef_df['EF_kg_per_kWh'].max()})"
    )

    merged = total_df.merge(ef_df[[province_col, "EF_kg_per_kWh"]], on=province_col, how="left")

    miss_mask = merged["EF_kg_per_kWh"].isna()
    n_missing = int(miss_mask.sum())
    if n_missing > 0:
        print(f"[WARN] {n_missing} row(s) did not match any EF value. Unmatched Province_ID samples (max 10):")
        print(merged.loc[miss_mask, province_col].drop_duplicates().head(10).to_string(index=False))

        left_ids = set(merged[province_col].dropna().unique().tolist())
        right_ids = set(ef_df[province_col].dropna().unique().tolist())
        only_left = list(left_ids - right_ids)[:10]
        only_right = list(right_ids - left_ids)[:10]
        if only_left:
            print(f"[DIFF] Only in Total_Energy (sample <=10): {only_left}")
        if only_right:
            print(f"[DIFF] Only in EF table (sample <=10): {only_right}")

    added_cols = []
    for elec_col in elec_cols:
        print(f"\n[CONVERT] '{elec_col}' (unit: 1e8 kWh) -> tCO2")
        elec_activity = to_num(merged[elec_col])
        ef_kg_per_kwh = merged["EF_kg_per_kWh"]

        before_nonnull = int(elec_activity.notna().sum())
        print(f"  [STAT] Non-null activity rows = {before_nonnull}")

        co2_t = elec_activity * ef_kg_per_kwh * 1e5
        co2_col = f"{elec_col}_CO2"
        merged[co2_col] = co2_t
        added_cols.append(co2_col)

        n_nonnull = int(merged[co2_col].notna().sum())
        total_sum = merged[co2_col].sum(skipna=True)
        print(f"  [DONE] '{co2_col}' non-null rows = {n_nonnull} / {len(merged)} | sum={total_sum:,.2f} tCO2")

    merged.to_csv(out_path, index=False, encoding=enc_total)
    print("\n========== Finished ==========")
    print(f"[OUT] Written to: {out_path.resolve()}")
    print(f"[INFO] Added columns: {added_cols} (unit: tCO2)")


if __name__ == "__main__":
    main()
