import os
import numpy as np
import pandas as pd

GDP_PATH = r"Total_Energy_GDP_Pop.csv"
BASE_PATH = r"Baseline_Energy_Share.csv"
SLOPE_PATH = r"energy_share_log_slope_2013_2022_byCodeSectorEnergy.csv"

OUT_ROOT = r"./Sector_Internal_Energy_Structure_2023_2100"
os.makedirs(OUT_ROOT, exist_ok=True)

SECTORS = [
    "Agriculture_Forestry_Husbandry_Fishery",
    "Industry",
    "Construction",
    "Transportation_Storage_Postal",
    "Wholesale_Retail_Accommodation_Catering",
    "Resident",
    "Other",
]

ENERGY_TYPES = [
    "Coal and coal products",
    "Petroleum and petroleum products",
    "Natural gas",
    "Heat",
    "Electricity",
]
IDX_COAL = ENERGY_TYPES.index("Coal and coal products")
IDX_PETR = ENERGY_TYPES.index("Petroleum and petroleum products")
IDX_GAS = ENERGY_TYPES.index("Natural gas")
IDX_HEAT = ENERGY_TYPES.index("Heat")
IDX_ELEC = ENERGY_TYPES.index("Electricity")

BASE_COL_MAP = {
    "Coal and coal products": "Coal and coal products_share_baseline",
    "Petroleum and petroleum products": "Petroleum and petroleum products_share_baseline",
    "Natural gas": "Natural gas_share_baseline",
    "Heat": "Heat_share_baseline",
    "Electricity": "Electricity_share_baseline",
}

BASE_YEAR = 2022
START_YEAR = 2023
END_YEAR = 2100
YEARS = np.arange(START_YEAR, END_YEAR + 1, dtype=int)

SSP_MULT = {
    "SSP0": (1.00, 1.00),
    "SSP1": (1.15, 1.25),
    "SSP2": (0.90, 1.10),
    "SSP3": (0.60, 0.80),
    "SSP5": (0.95, 1.15),
}
SSP4_MULT = {
    "high": (1.05, 1.15),
    "mid": (0.85, 1.05),
    "low": (0.65, 0.95),
}
SSP_ORDER = ["SSP0", "SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]

def time_decay_weight(year: int) -> float:
    if year <= 2030:
        return 1.0
    step = (year - 2031) // 10 + 1
    w = 1.0 - 0.1 * step
    return max(w, 0.0)

W_YEARS = np.array([time_decay_weight(int(y)) for y in YEARS], dtype=float)

def norm_code(x) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s2 = s[:-2]
        if s2.isdigit():
            return s2
    return s

def build_income_group_map(gdp_path: str):
    gdp = pd.read_csv(gdp_path, encoding="utf-8-sig")
    required_gdp = {"Code", "Year", "GDP"}
    miss = required_gdp - set(gdp.columns)
    if miss:
        raise ValueError(f"{gdp_path} is missing columns: {miss}")

    gdp["Code"] = gdp["Code"].apply(norm_code)
    gdp["Year"] = pd.to_numeric(gdp["Year"], errors="coerce")
    gdp["GDP"] = pd.to_numeric(gdp["GDP"], errors="coerce")
    gdp = gdp.dropna(subset=["Code", "Year", "GDP"]).copy()
    gdp["Year"] = gdp["Year"].astype(int)

    gdp_10y = gdp[(gdp["Year"] >= 2013) & (gdp["Year"] <= 2022)].copy()
    gdp_mean = gdp_10y.groupby("Code", as_index=False).agg(GDP_mean=("GDP", "mean"))

    q25 = gdp_mean["GDP_mean"].quantile(0.25) if len(gdp_mean) else np.nan
    q75 = gdp_mean["GDP_mean"].quantile(0.75) if len(gdp_mean) else np.nan

    def income_group(x):
        if pd.isna(x) or pd.isna(q25) or pd.isna(q75):
            return "mid"
        if x >= q75:
            return "high"
        if x < q25:
            return "low"
        return "mid"

    gdp_mean["income_group"] = gdp_mean["GDP_mean"].apply(income_group)
    return dict(zip(gdp_mean["Code"], gdp_mean["income_group"]))

INCOME_GROUP_MAP = build_income_group_map(GDP_PATH)

INF = 10**9
ALLOW_POS_COAL = {"SSP1": 0, "SSP2": 20, "SSP3": 40, "SSP5": 40}
ALLOW_POS_PETR = {"SSP1": 5, "SSP2": 20, "SSP3": 40, "SSP5": 40}

def allow_years_for_code(ssp: str, code: str, which: str) -> int:
    if ssp in ("SSP0",):
        return INF

    if ssp in ("SSP1", "SSP2", "SSP3", "SSP5"):
        d = ALLOW_POS_COAL if which == "coal" else ALLOW_POS_PETR
        return int(d.get(ssp, 0))

    if ssp == "SSP4":
        g = INCOME_GROUP_MAP.get(code, "mid")
        mimic = "SSP1" if g == "high" else ("SSP3" if g == "low" else "SSP2")
        d = ALLOW_POS_COAL if which == "coal" else ALLOW_POS_PETR
        return int(d.get(mimic, 0))

    return 0

base = pd.read_csv(BASE_PATH, encoding="utf-8-sig")
required_base = {"Code", "Sector"} | set(BASE_COL_MAP.values())
miss = required_base - set(base.columns)
if miss:
    raise ValueError(f"{BASE_PATH} is missing columns: {sorted(miss)}")

base["Code"] = base["Code"].apply(norm_code)
base["Sector"] = base["Sector"].astype(str).str.strip()

for et in ENERGY_TYPES:
    col = BASE_COL_MAP[et]
    base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0).clip(lower=0.0)

base_sum = base[[BASE_COL_MAP[et] for et in ENERGY_TYPES]].sum(axis=1).replace(0, np.nan)
base[[BASE_COL_MAP[et] for et in ENERGY_TYPES]] = (
    base[[BASE_COL_MAP[et] for et in ENERGY_TYPES]].div(base_sum, axis=0)
)
base[[BASE_COL_MAP[et] for et in ENERGY_TYPES]] = (
    base[[BASE_COL_MAP[et] for et in ENERGY_TYPES]].fillna(1.0 / len(ENERGY_TYPES))
)

base = base[base["Sector"].isin(SECTORS)].copy()

BASE_BY_SECTOR = {}
for sec in SECTORS:
    g = base[base["Sector"] == sec].copy()
    if len(g) == 0:
        BASE_BY_SECTOR[sec] = {"codes": [], "S0": None, "fixed_zero": None}
        continue
    codes = g["Code"].tolist()
    s0 = g[[BASE_COL_MAP[et] for et in ENERGY_TYPES]].to_numpy(dtype=float)
    fixed_zero = (s0 <= 0.0)
    BASE_BY_SECTOR[sec] = {"codes": codes, "S0": s0, "fixed_zero": fixed_zero}

print("[Info] Baseline loaded.")
for sec in SECTORS:
    print(f"  - {sec}: n_codes={len(BASE_BY_SECTOR[sec]['codes']):,}")

slp = pd.read_csv(SLOPE_PATH, encoding="utf-8-sig")
required_slope = {"Code", "Sector", "EnergyType", "log_slope"}
miss = required_slope - set(slp.columns)
if miss:
    raise ValueError(f"{SLOPE_PATH} is missing columns: {sorted(miss)}")

slp["Code"] = slp["Code"].apply(norm_code)
slp["Sector"] = slp["Sector"].astype(str).str.strip()
slp["EnergyType"] = slp["EnergyType"].astype(str).str.strip()
slp["log_slope"] = pd.to_numeric(slp["log_slope"], errors="coerce").fillna(0.0)

slp = slp[slp["Sector"].isin(SECTORS) & slp["EnergyType"].isin(ENERGY_TYPES)].copy()
slp["rate"] = np.expm1(slp["log_slope"])

rate_pivot = (
    slp.pivot_table(
        index=["Code", "Sector"],
        columns="EnergyType",
        values="rate",
        aggfunc="last",
    )
    .reindex(columns=ENERGY_TYPES)
)

def multiplier_for_codes(ssp: str, codes: list, variant: str) -> np.ndarray:
    if ssp == "SSP0":
        return np.ones(len(codes), dtype=float)

    if ssp in ("SSP1", "SSP2", "SSP3", "SSP5"):
        a, b = SSP_MULT[ssp]
        m = a if variant == "low" else b
        return np.full(len(codes), m, dtype=float)

    if ssp == "SSP4":
        m = np.empty(len(codes), dtype=float)
        for i, c in enumerate(codes):
            g = INCOME_GROUP_MAP.get(c, "mid")
            a, b = SSP4_MULT.get(g, SSP4_MULT["mid"])
            m[i] = a if variant == "low" else b
        return m

    raise ValueError(f"Unknown SSP: {ssp}")

def simulate_one_sector(ssp: str, variant: str, sector: str):
    info = BASE_BY_SECTOR[sector]
    codes = info["codes"]
    if not codes:
        print(f"[Skip] {ssp}/{variant} sector={sector}: no codes")
        return

    if ssp == "SSP1":
        coal_force_neg = -0.18
        petr_force_neg = -0.01
        elec_force_pos = 0.05
    else:
        coal_force_neg = -0.09
        petr_force_neg = -0.005
        elec_force_pos = 0.025

    s = info["S0"].copy()
    fixed_zero = info["fixed_zero"].copy()

    idx = pd.MultiIndex.from_arrays([codes, [sector] * len(codes)], names=["Code", "Sector"])
    r0 = rate_pivot.reindex(index=idx).fillna(0.0).to_numpy(dtype=float)

    r0[fixed_zero] = 0.0

    m_code = multiplier_for_codes(ssp, codes, variant)
    m_mat = m_code[:, None]

    allow_coal = np.array([allow_years_for_code(ssp, c, "coal") for c in codes], dtype=int)
    allow_petr = np.array([allow_years_for_code(ssp, c, "petr") for c in codes], dtype=int)

    out_dir = os.path.join(OUT_ROOT, ssp)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"energy_structure_{sector}_{ssp}_2023_2100_{variant}.csv")

    first_write = True

    for t, y in enumerate(YEARS):
        dt = y - BASE_YEAR
        w = W_YEARS[t]

        r_w = r0 * w
        r_eff = r_w * m_mat

        r_eff[fixed_zero] = 0.0
        r_w[fixed_zero] = 0.0

        mask_pos_coal = (r_w[:, IDX_COAL] > 0)
        if np.any(mask_pos_coal):
            r_eff[mask_pos_coal, IDX_COAL] = r_w[mask_pos_coal, IDX_COAL]

        mask_pos_petr = (r_w[:, IDX_PETR] > 0)
        if np.any(mask_pos_petr):
            r_eff[mask_pos_petr, IDX_PETR] = r_w[mask_pos_petr, IDX_PETR]

        if ssp == "SSP1":
            mask_coal_over = (r_eff[:, IDX_COAL] > coal_force_neg) & (dt > allow_coal)
            if np.any(mask_coal_over):
                r_eff[mask_coal_over, IDX_COAL] = coal_force_neg

            mask_petr_over = (r_eff[:, IDX_PETR] > petr_force_neg) & (dt > allow_petr)
            if np.any(mask_petr_over):
                r_eff[mask_petr_over, IDX_PETR] = petr_force_neg
        else:
            mask_coal_over = (r_eff[:, IDX_COAL] > 0.0) & (dt > allow_coal)
            if np.any(mask_coal_over):
                r_eff[mask_coal_over, IDX_COAL] = coal_force_neg

            mask_petr_over = (r_eff[:, IDX_PETR] > 0.0) & (dt > allow_petr)
            if np.any(mask_petr_over):
                r_eff[mask_petr_over, IDX_PETR] = petr_force_neg

        mask_allneg = (
            (r_eff[:, IDX_COAL] < 0)
            & (r_eff[:, IDX_GAS] < 0)
            & (r_eff[:, IDX_PETR] < 0)
            & (r_eff[:, IDX_ELEC] < 0)
        )
        if np.any(mask_allneg):
            r_eff[mask_allneg, IDX_ELEC] = elec_force_pos

        r_eff = np.maximum(r_eff, -0.999999)

        factors = 1.0 + r_eff
        s_raw = s * factors
        s_raw[fixed_zero] = 0.0
        s_raw = np.clip(s_raw, 0.0, None)

        if ssp == "SSP1":
            coal = s_raw[:, IDX_COAL]
            petr = s_raw[:, IDX_PETR]
            remain = 1.0 - coal - petr
            remain = np.clip(remain, 0.0, None)

            others_idx = [IDX_GAS, IDX_HEAT, IDX_ELEC]
            others = s_raw[:, others_idx]
            o_sum = others.sum(axis=1, keepdims=True)

            ok = (o_sum[:, 0] > 0) & np.isfinite(o_sum[:, 0])
            if np.any(ok):
                others[ok] = others[ok] / o_sum[ok] * remain[ok, None]

            s_raw[:, IDX_COAL] = coal
            s_raw[:, IDX_PETR] = petr
            s_raw[:, others_idx] = others
            s = s_raw
        else:
            rs = s_raw.sum(axis=1, keepdims=True)
            bad = (rs[:, 0] <= 0) | (~np.isfinite(rs[:, 0]))
            if np.any(bad):
                s_raw[bad, :] = s[bad, :]
                rs[bad, 0] = s_raw[bad, :].sum(axis=1)
            s = s_raw / rs

        out_df = pd.DataFrame(s, columns=[f"{et}_share" for et in ENERGY_TYPES])
        out_df.insert(0, "Year", y)
        out_df.insert(0, "Code", codes)

        out_df.to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig",
            mode=("w" if first_write else "a"),
            header=first_write,
        )
        first_write = False

    print(f"[OK] Saved: {out_path} | n_codes={len(codes):,} | years={len(YEARS)}")

def run_one_ssp(ssp: str):
    for variant in ("low", "high"):
        for sector in SECTORS:
            simulate_one_sector(ssp, variant, sector)

if __name__ == "__main__":
    print("[Info] Start simulation ...")
    print(f"[Info] years={len(YEARS)} ({START_YEAR}-{END_YEAR}), sectors={len(SECTORS)}, energy_types={len(ENERGY_TYPES)}")

    for ssp in SSP_ORDER:
        print(f"\n===== Running {ssp} =====")
        run_one_ssp(ssp)

    print("\nAll done! Outputs in:", OUT_ROOT)
