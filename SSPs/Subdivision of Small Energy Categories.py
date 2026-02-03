import re
from pathlib import Path
import pandas as pd

SHARE_FILE = "Origin_Total_Energy_2022_within_big_share.csv"

FOLDER_SECTOR_MAP = {
    "2013-2100工业能源结构": "Industry",
    "2013-2100建筑业能源结构": "Construction",
    "2013-2100交通行业能源结构": "Transportation_Storage_Postal",
    "2013-2100居民生活能源结构": "Resident",
    "2013-2100零售住宿行业能源结构": "Wholesale_Retail_Accommodation_Catering",
    "2013-2100农林牧渔业能源结构": "Agriculture_Forestry_Husbandry_Fishery",
    "2013-2100其它行业能源结构": "Other",
}

BIG_COLS = {
    "Coal": "Coal and coal products_Energy",
    "Petroleum": "Petroleum and petroleum products_Energy",
    "Gas": "Natural gas_Energy",
    "Heat": "Heat_Energy",
    "Electricity": "Electricity_Energy",
}

OUTPUT_SUBDIR_NAME = "Subcategory_Allocation_Results_2022"


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep="\t")
    return df


def normalize_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def share_col_to_energy_col(col: str, existing_cols: set) -> str:
    base = col
    base = re.sub(r"_share_in_.*$", "", base)
    base = base.replace("_share", "")
    if not base.endswith("_Energy"):
        base = base + "_Energy"
    if base in existing_cols:
        base = base + "_detail"
    return base


def load_share_table(share_path: Path) -> pd.DataFrame:
    df = read_csv_auto(share_path)

    if "Code" not in df.columns or "Sector" not in df.columns:
        raise KeyError("The share file must contain 'Code' and 'Sector' columns.")

    energy_share_cols = list(df.columns[3:])
    if len(energy_share_cols) < 29:
        raise ValueError(
            f"Not enough energy-share columns starting from the 4th column: {len(energy_share_cols)}. Need at least 29."
        )

    energy_share_cols = energy_share_cols[:29]

    coal_cols = energy_share_cols[0:11]
    petro_cols = energy_share_cols[11:25]
    gas_cols = energy_share_cols[25:27]
    heat_cols = energy_share_cols[27:28]
    elec_cols = energy_share_cols[28:29]

    df = df.copy()
    df["Code"] = normalize_code(df["Code"])
    df["Sector"] = df["Sector"].astype(str).str.strip()
    df[energy_share_cols] = df[energy_share_cols].apply(pd.to_numeric, errors="coerce")

    def renorm(cols):
        s = df[cols].sum(axis=1)
        mask = s.notna() & (s > 0)
        df.loc[mask, cols] = df.loc[mask, cols].div(s[mask], axis=0)

    renorm(coal_cols)
    renorm(petro_cols)
    renorm(gas_cols)
    renorm(heat_cols)
    renorm(elec_cols)

    df.attrs["energy_share_cols"] = energy_share_cols
    df.attrs["coal_cols"] = coal_cols
    df.attrs["petro_cols"] = petro_cols
    df.attrs["gas_cols"] = gas_cols
    df.attrs["heat_cols"] = heat_cols
    df.attrs["elec_cols"] = elec_cols
    return df


def process_one_sector_folder(folder: Path, sector_value: str, share_df_all: pd.DataFrame):
    out_dir = folder / OUTPUT_SUBDIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    share_cols = share_df_all.attrs["energy_share_cols"]
    coal_cols = share_df_all.attrs["coal_cols"]
    petro_cols = share_df_all.attrs["petro_cols"]
    gas_cols = share_df_all.attrs["gas_cols"]
    heat_cols = share_df_all.attrs["heat_cols"]
    elec_cols = share_df_all.attrs["elec_cols"]

    share_df = share_df_all.loc[share_df_all["Sector"] == sector_value, ["Code"] + share_cols].copy()
    share_df = share_df.drop_duplicates(subset=["Code"], keep="first")

    csv_files = sorted([p for p in folder.glob("*.csv") if p.is_file()])
    if not csv_files:
        print(f"[WARN] No CSV files found under {folder.name}. Skipping.")
        return

    for csv_path in csv_files:
        df = read_csv_auto(csv_path)

        need_cols = {"Code", "Year"} | set(BIG_COLS.values())
        missing = need_cols - set(df.columns)
        if missing:
            print(f"[SKIP] {folder.name}/{csv_path.name} is missing columns: {missing}")
            continue

        df = df.copy()
        df["Code"] = normalize_code(df["Code"])
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        mask_future = df["Year"] > 2022

        merged = df.merge(share_df, on="Code", how="left")

        existing = set(merged.columns)

        outcol_map = {}
        for sc in (coal_cols + petro_cols + gas_cols + heat_cols + elec_cols):
            out_col = share_col_to_energy_col(sc, existing)
            existing.add(out_col)
            outcol_map[sc] = out_col
            merged[out_col] = pd.NA

        def alloc_group(big_key: str, group_share_cols: list):
            big_col = BIG_COLS[big_key]
            big_vals = pd.to_numeric(merged[big_col], errors="coerce")
            for sc in group_share_cols:
                out_col = outcol_map[sc]
                merged.loc[mask_future, out_col] = big_vals[mask_future] * merged.loc[mask_future, sc]

        alloc_group("Coal", coal_cols)
        alloc_group("Petroleum", petro_cols)
        alloc_group("Gas", gas_cols)
        alloc_group("Heat", heat_cols)
        alloc_group("Electricity", elec_cols)

        merged.drop(columns=share_cols, inplace=True)

        out_path = out_dir / csv_path.name
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] {folder.name}/{csv_path.name} -> {out_dir.name}/{out_path.name}")


def main():
    base_dir = Path(".")
    share_path = base_dir / SHARE_FILE
    if not share_path.exists():
        raise FileNotFoundError(f"Share file not found: {share_path.resolve()}")

    share_df_all = load_share_table(share_path)

    for folder_name, sector_value in FOLDER_SECTOR_MAP.items():
        folder = base_dir / folder_name
        if not folder.exists() or not folder.is_dir():
            print(f"[WARN] Subfolder not found: {folder_name}. Skipping.")
            continue
        process_one_sector_folder(folder, sector_value, share_df_all)

    print("\n[DONE] All processing completed.")


if __name__ == "__main__":
    main()
