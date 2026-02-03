import os
import numpy as np
import pandas as pd

GDP_PATH = r"Total_Energy_GDP_Pop.csv"
BASE_PATH = r"sector_shares_baseline.csv"
SLOPE_PATH = r"sector_share_log_slope_2013_2022_byCodeSector.csv"

OUT_ROOT = r"./SSP_sector_shares_outputs"
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
INDUSTRY = "Industry"
CONSTR = "Construction"
TRANS = "Transportation_Storage_Postal"

IDX_INDUSTRY = SECTORS.index(INDUSTRY)
IDX_CONSTR = SECTORS.index(CONSTR)
IDX_TRANS = SECTORS.index(TRANS)

LIMITED_SECTORS = {INDUSTRY, CONSTR, TRANS}
COND_MULT_SECTORS = {INDUSTRY, CONSTR}

SSP1_INDUSTRY_THRESHOLD = 0.6

BASE_YEAR = 2022
START_YEAR = 2023
END_YEAR = 2100
YEARS = np.arange(START_YEAR, END_YEAR + 1, dtype=int)

ALLOW_YEARS = {
    "SSP1": 10,
    "SSP2": 20,
    "SSP3": 40,
    "SSP5": 30,
}
FORCE_NEG_IF_POS_AFTER = -0.0005

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

gdp = pd.read_csv(GDP_PATH, encoding="utf-8-sig")
required_gdp = {"Code", "Year", "GDP"}
miss = required_gdp - set(gdp.columns)
if miss:
    raise ValueError(f"{GDP_PATH} is missing columns: {miss}")

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
grp_map = dict(zip(gdp_mean["Code"], gdp_mean["income_group"]))

base = pd.read_csv(BASE_PATH, encoding="utf-8-sig")
if "Code" not in base.columns:
    raise ValueError(f"{BASE_PATH} is missing the Code column")
for s in SECTORS:
    if s not in base.columns:
        raise ValueError(f"{BASE_PATH} is missing sector column: {s}")

base["Code"] = base["Code"].apply(norm_code)
for s in SECTORS:
    base[s] = pd.to_numeric(base[s], errors="coerce").fillna(0.0).clip(lower=0.0)

row_sum = base[SECTORS].sum(axis=1).replace(0, np.nan)
base[SECTORS] = base[SECTORS].div(row_sum, axis=0).fillna(1.0 / len(SECTORS))

codes = base["Code"].drop_duplicates().tolist()
n_codes = len(codes)

S0 = base.set_index("Code").loc[codes, SECTORS].to_numpy(dtype=float)
income = np.array([grp_map.get(c, "mid") for c in codes], dtype=object)

slp = pd.read_csv(SLOPE_PATH, encoding="utf-8-sig")
required_slope = {"Code", "Sector", "log_slope"}
miss = required_slope - set(slp.columns)
if miss:
    raise ValueError(f"{SLOPE_PATH} is missing columns: {miss}")

slp["Code"] = slp["Code"].apply(norm_code)
slp["Sector"] = slp["Sector"].astype(str)
slp["log_slope"] = pd.to_numeric(slp["log_slope"], errors="coerce")

slp = slp[slp["Code"] != "TOTAL"].copy()

pivot = slp.pivot_table(index="Code", columns="Sector", values="log_slope", aggfunc="last")
pivot = pivot.reindex(columns=SECTORS).reindex(index=codes)

for s in SECTORS:
    med = pivot[s].median(skipna=True)
    if pd.isna(med):
        med = 0.0
    pivot[s] = pivot[s].fillna(med)

B0 = pivot.to_numpy(dtype=float)

def get_allow_years_code(ssp: str) -> np.ndarray:
    if ssp == "SSP0":
        return np.full(n_codes, 0, dtype=int)
    if ssp != "SSP4":
        return np.full(n_codes, ALLOW_YEARS.get(ssp, 0), dtype=int)

    out = np.zeros(n_codes, dtype=int)
    for i, g in enumerate(income):
        if g == "high":
            out[i] = ALLOW_YEARS["SSP1"]
        elif g == "low":
            out[i] = ALLOW_YEARS["SSP3"]
        else:
            out[i] = ALLOW_YEARS["SSP2"]
    return out

def multipliers_for_ssp(ssp: str, b_cur: np.ndarray):
    m_low = np.ones_like(b_cur, dtype=float)
    m_high = np.ones_like(b_cur, dtype=float)

    if ssp == "SSP0":
        return m_low, m_high

    cond_mask = np.array([s in COND_MULT_SECTORS for s in SECTORS], dtype=bool)

    if ssp == "SSP2":
        a, b = SSP_MULT["SSP2"]
        m_low[:] = a
        m_high[:] = b
        return m_low, m_high

    if ssp in ("SSP1", "SSP3", "SSP5"):
        a, b = SSP_MULT[ssp]
        m_low[:, ~cond_mask] = a
        m_high[:, ~cond_mask] = b
        neg = (b_cur < 0)
        m_low[:, cond_mask] = np.where(neg[:, cond_mask], a, 1.0)
        m_high[:, cond_mask] = np.where(neg[:, cond_mask], b, 1.0)
        return m_low, m_high

    if ssp == "SSP4":
        a_code = np.empty(n_codes, dtype=float)
        b_code = np.empty(n_codes, dtype=float)
        for i, g in enumerate(income):
            aa, bb = SSP4_MULT.get(g, SSP4_MULT["mid"])
            a_code[i] = aa
            b_code[i] = bb

        m_low[:, ~cond_mask] = a_code[:, None]
        m_high[:, ~cond_mask] = b_code[:, None]
        neg = (b_cur < 0)
        m_low[:, cond_mask] = np.where(neg[:, cond_mask], a_code[:, None], 1.0)
        m_high[:, cond_mask] = np.where(neg[:, cond_mask], b_code[:, None], 1.0)
        return m_low, m_high

    raise ValueError(f"Unknown SSP: {ssp}")

def run_one_ssp(ssp: str):
    out_dir = os.path.join(OUT_ROOT, ssp)
    os.makedirs(out_dir, exist_ok=True)

    allow_base = get_allow_years_code(ssp)
    allow_trans = allow_base * 2

    inf = 10**9
    allow_mat = np.full((n_codes, len(SECTORS)), inf, dtype=int)
    allow_mat[:, IDX_INDUSTRY] = allow_base
    allow_mat[:, IDX_CONSTR] = allow_base
    allow_mat[:, IDX_TRANS] = allow_trans

    ind_forced = np.zeros(n_codes, dtype=bool)

    for variant in ("low", "high"):
        S = S0.copy()
        share_rows = []
        slope_rows = []

        for t, y in enumerate(YEARS):
            dt = y - BASE_YEAR
            w = W_YEARS[t]

            if ssp == "SSP0":
                b_cur = B0.copy()
            else:
                b_cur = B0.copy()
                force_mask = (B0 > 0) & (dt > allow_mat)
                b_cur = np.where(force_mask, FORCE_NEG_IF_POS_AFTER, b_cur)

                if ssp == "SSP1":
                    can_trigger = (B0[:, IDX_INDUSTRY] > 0)
                    hit = can_trigger & (S[:, IDX_INDUSTRY] >= SSP1_INDUSTRY_THRESHOLD)
                    ind_forced = ind_forced | hit
                    b_cur[:, IDX_INDUSTRY] = np.where(ind_forced, FORCE_NEG_IF_POS_AFTER, b_cur[:, IDX_INDUSTRY])

            m_low, m_high = multipliers_for_ssp(ssp, b_cur)
            m = m_low if variant == "low" else m_high

            b_eff = (b_cur * m) * w

            s_raw = S * np.exp(b_eff)

            rs = s_raw.sum(axis=1, keepdims=True)
            bad = (rs[:, 0] <= 0) | (~np.isfinite(rs[:, 0]))
            if np.any(bad):
                s_raw[bad, :] = S[bad, :]
                rs[bad, 0] = s_raw[bad, :].sum(axis=1)

            S = s_raw / rs

            df_share = pd.DataFrame(S, columns=SECTORS)
            df_share.insert(0, "Year", y)
            df_share.insert(0, "Code", codes)
            share_rows.append(df_share)

            df_slope = pd.DataFrame(b_eff, columns=SECTORS)
            df_slope.insert(0, "Year", y)
            df_slope.insert(0, "Code", codes)
            slope_rows.append(df_slope)

        share_out = pd.concat(share_rows, ignore_index=True)
        slope_out = pd.concat(slope_rows, ignore_index=True)

        max_dev = float(np.max(np.abs(share_out[SECTORS].sum(axis=1) - 1.0)))

        p_share = os.path.join(out_dir, f"sector_share_{ssp}_2023_2100_{variant}.csv")
        p_slope = os.path.join(out_dir, f"sector_log_slope_{ssp}_2023_2100_{variant}.csv")
        share_out.to_csv(p_share, index=False, encoding="utf-8-sig")
        slope_out.to_csv(p_slope, index=False, encoding="utf-8-sig")

        print(f"[OK] {ssp}/{variant} saved -> {p_share} | {p_slope} | max(|sum-1|)={max_dev:.3e}")

if __name__ == "__main__":
    print(f"[Info] n_codes={n_codes}, years={len(YEARS)}, sectors={len(SECTORS)}")
    for ssp in SSP_ORDER:
        run_one_ssp(ssp)
    print("\nAll done! Outputs in:", OUT_ROOT)
