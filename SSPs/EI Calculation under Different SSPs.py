import os
from typing import Optional

import numpy as np
import pandas as pd

EI_BASE_PATH = r"EI_Catrgory_Energy.csv"
GDP_PATH = r"Total_Energy_GDP_Pop.csv"
EI_RATE_PATH = r"EI_change_rates_2013_2022_byCode_and_TOTAL.csv"

OUT_DIR = r"./EI_outputs_by_SSP_new"
EI0_COL = "EI_exp_2022_alpha0.9"
GDP_COL = "GDP"

BASE_YEAR = 2022
START_FUTURE = 2023
END_YEAR = 2100

SSP5_POS_HOLD_YEARS = 20

FORCE_DECLINE_IF_NONNEG_B = -0.015
FORCE_R = float(np.expm1(FORCE_DECLINE_IF_NONNEG_B))

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

SSP_CAP_FACTOR = {
    "SSP2": 2.0,
    "SSP3": 10.0,
}

def get_cap_factor(ssp: str, grp: str) -> Optional[float]:
    if ssp in ("SSP2", "SSP3"):
        return SSP_CAP_FACTOR[ssp]
    if ssp == "SSP4":
        if grp == "high":
            return None
        if grp == "mid":
            return SSP_CAP_FACTOR["SSP2"]
        return 5.0
    return None

os.makedirs(OUT_DIR, exist_ok=True)

def time_decay_weight(year: int) -> float:
    if year <= 2030:
        return 1.0
    step = (year - 2031) // 10 + 1
    w = 1.0 - 0.1 * step
    return max(w, 0.0)

def make_r_eff_series(r_scalar: float, w_years: np.ndarray, hold_zero_years: int = 0) -> np.ndarray:
    r_eff = r_scalar * w_years
    if hold_zero_years and hold_zero_years > 0:
        n = min(int(hold_zero_years), len(r_eff))
        r_eff[:n] = 0.0
    r_eff = np.clip(r_eff, -0.999999, None)
    return r_eff

def compound_with_optional_cap(ei_start: float, r_eff: np.ndarray, cap: Optional[float]) -> np.ndarray:
    if cap is None:
        return ei_start * np.cumprod(1.0 + r_eff)

    out = np.empty_like(r_eff, dtype=float)
    ei = float(ei_start)
    capv = float(cap)
    for i in range(len(r_eff)):
        ei = ei * (1.0 + float(r_eff[i]))
        if ei > capv:
            ei = capv
        out[i] = ei
    return out

ei0 = pd.read_csv(EI_BASE_PATH, encoding="utf-8-sig")
if "Code" not in ei0.columns or EI0_COL not in ei0.columns:
    raise ValueError(f"{EI_BASE_PATH} must contain columns: Code and {EI0_COL}")

ei0["Code"] = ei0["Code"].astype(str)
ei0[EI0_COL] = pd.to_numeric(ei0[EI0_COL], errors="coerce")
code_order = ei0["Code"].drop_duplicates().tolist()
ei0_nodup = ei0.drop_duplicates(subset=["Code"], keep="last").copy()

gdp = pd.read_csv(GDP_PATH, encoding="utf-8-sig")
need_gdp = {"Code", "Year", GDP_COL}
missing = need_gdp - set(gdp.columns)
if missing:
    raise ValueError(f"{GDP_PATH} is missing required columns: {missing}")

gdp["Code"] = gdp["Code"].astype(str)
gdp["Year"] = pd.to_numeric(gdp["Year"], errors="coerce")
gdp[GDP_COL] = pd.to_numeric(gdp[GDP_COL], errors="coerce")

gdp_2022 = gdp[gdp["Year"] == BASE_YEAR][["Code", GDP_COL]].copy()
gdp_2022 = gdp_2022[gdp_2022["Code"].isin(code_order)].copy()
gdp_2022 = gdp_2022.groupby("Code", as_index=False).agg(**{GDP_COL: (GDP_COL, "sum")})

q25 = gdp_2022[GDP_COL].quantile(0.25) if len(gdp_2022) else np.nan
q75 = gdp_2022[GDP_COL].quantile(0.75) if len(gdp_2022) else np.nan

def income_group(x):
    if pd.isna(x) or pd.isna(q25) or pd.isna(q75):
        return "mid"
    if x >= q75:
        return "high"
    if x < q25:
        return "low"
    return "mid"

gdp_2022["income_group"] = gdp_2022[GDP_COL].apply(income_group)

rate = pd.read_csv(EI_RATE_PATH, encoding="utf-8-sig")
need_rate = {"Code", "log_slope"}
miss = need_rate - set(rate.columns)
if miss:
    raise ValueError(f"{EI_RATE_PATH} is missing required columns: {miss}")

rate["Code"] = rate["Code"].astype(str)
rate["log_slope"] = pd.to_numeric(rate["log_slope"], errors="coerce")
rate = rate.drop_duplicates(subset=["Code"], keep="last")

row_total = rate[rate["Code"] == "TOTAL"]
total_slope_b = np.nan
if len(row_total) and pd.notna(row_total["log_slope"].iloc[0]):
    total_slope_b = float(row_total["log_slope"].iloc[0])

rate_region = rate[rate["Code"].isin(code_order)][["Code", "log_slope"]].copy()
all_codes_df = pd.DataFrame({"Code": code_order})
rate_region = all_codes_df.merge(rate_region, on="Code", how="left")

fallback_b = total_slope_b if not pd.isna(total_slope_b) else FORCE_DECLINE_IF_NONNEG_B
rate_region["log_slope"] = rate_region["log_slope"].fillna(fallback_b)

rate_region["decline_rate_signed"] = np.expm1(rate_region["log_slope"])

rate_region = rate_region.merge(gdp_2022[["Code", "income_group"]], on="Code", how="left")
rate_region["income_group"] = rate_region["income_group"].fillna("mid")

rate_region = rate_region.merge(
    ei0_nodup[["Code", EI0_COL]].rename(columns={EI0_COL: "EI_2022"}),
    on="Code",
    how="left",
)

if rate_region["EI_2022"].isna().any():
    miss_codes = rate_region.loc[rate_region["EI_2022"].isna(), "Code"].tolist()
    raise ValueError(
        f"Missing EI_2022 for these codes in {EI_BASE_PATH}: {miss_codes[:10]}... (total {len(miss_codes)})"
    )

base_out = rate_region[["Code", "income_group", "EI_2022", "log_slope", "decline_rate_signed"]].copy()
out_base = os.path.join(OUT_DIR, "EI_decline_rate_base_signed_no_m.csv")
base_out.to_csv(out_base, index=False, encoding="utf-8-sig")

years = np.arange(START_FUTURE, END_YEAR + 1, dtype=int)
w_years = np.array([time_decay_weight(int(y)) for y in years], dtype=float)
code_cat = pd.CategoricalDtype(categories=code_order, ordered=True)

def build_rates_for_code(ssp: str, grp: str, r0: float):
    hold_zero_years = 0

    if ssp == "SSP0":
        return 1.0, 1.0, 0.0, 0.0, 10**9, "SSP0_hold_constant_EI2022"

    if ssp == "SSP1":
        m_low, m_high = SSP_MULT["SSP1"]
        r_base = r0 if (r0 < 0) else FORCE_R
        return m_low, m_high, r_base * m_low, r_base * m_high, 0, "SSP1_force_neg_then_mult"

    if ssp == "SSP2":
        m_low, m_high = SSP_MULT["SSP2"]
        return m_low, m_high, r0 * m_low, r0 * m_high, 0, "SSP2_mult_all"

    if ssp == "SSP3":
        if r0 < 0:
            m_low, m_high = SSP_MULT["SSP3"]
            return m_low, m_high, r0 * m_low, r0 * m_high, 0, "SSP3_neg_mult"
        m_low, m_high = 1.20, 1.40
        return m_low, m_high, r0 * m_low, r0 * m_high, 0, "SSP3_pos_mult_1.2_1.4"

    if ssp == "SSP4":
        m_low_g, m_high_g = SSP4_MULT.get(grp, SSP4_MULT["mid"])

        if grp == "high":
            r_base = r0 if (r0 < 0) else FORCE_R
            return m_low_g, m_high_g, r_base * m_low_g, r_base * m_high_g, 0, "SSP4_high_force_neg_then_mult"

        if grp == "mid":
            return m_low_g, m_high_g, r0 * m_low_g, r0 * m_high_g, 0, "SSP4_mid_mult_all"

        if r0 < 0:
            return m_low_g, m_high_g, r0 * m_low_g, r0 * m_high_g, 0, "SSP4_low_neg_mult"
        m_low_p, m_high_p = 1.20, 1.40
        return m_low_p, m_high_p, r0 * m_low_p, r0 * m_high_p, 0, "SSP4_low_pos_mult_1.2_1.4"

    if ssp == "SSP5":
        m_low, m_high = SSP_MULT["SSP5"]
        if r0 > 0:
            hold_zero_years = SSP5_POS_HOLD_YEARS
            r_low = FORCE_R * m_low
            r_high = FORCE_R * m_high
            return m_low, m_high, r_low, r_high, hold_zero_years, "SSP5_pos_hold20_then_force_neg_with_mult"
        return m_low, m_high, r0 * m_low, r0 * m_high, 0, "SSP5_mult_all_for_nonpos"

    raise ValueError(f"Unknown SSP: {ssp}")

for ssp in SSP_ORDER:
    rows_r = []
    for _, row in rate_region.iterrows():
        code = row["Code"]
        grp = row["income_group"]
        r0 = float(row["decline_rate_signed"])

        m_low, m_high, r_low, r_high, hold_zero_years, rule = build_rates_for_code(ssp, grp, r0)

        rows_r.append([
            code,
            grp,
            r0,
            m_low,
            m_high,
            r_low,
            r_high,
            int(hold_zero_years),
            rule,
        ])

    r_param = pd.DataFrame(
        rows_r,
        columns=[
            "Code",
            "income_group",
            "rate_base_r0",
            "m_low",
            "m_high",
            "rate_low",
            "rate_high",
            "hold_zero_years",
            "rule",
        ],
    )
    r_param["SSP"] = ssp

    r_param["Code"] = r_param["Code"].astype(code_cat)
    r_param = r_param.sort_values("Code").reset_index(drop=True)
    r_param["Code"] = r_param["Code"].astype(str)

    out_r = os.path.join(OUT_DIR, f"EI_decline_rate_interval_{ssp}.csv")
    r_param.to_csv(out_r, index=False, encoding="utf-8-sig")

    traj_rows = []
    for code in code_order:
        sub = r_param[r_param["Code"] == code]
        if len(sub) == 0:
            continue

        grp = sub["income_group"].iloc[0]
        ei_2022 = float(rate_region.loc[rate_region["Code"] == code, "EI_2022"].iloc[0])

        r_low = float(sub["rate_low"].iloc[0])
        r_high = float(sub["rate_high"].iloc[0])
        hold_n = int(sub["hold_zero_years"].iloc[0])

        cap_factor = get_cap_factor(ssp, grp)
        cap_value = (cap_factor * ei_2022) if cap_factor is not None else None

        r_low_eff = make_r_eff_series(r_low, w_years, hold_zero_years=hold_n)
        r_high_eff = make_r_eff_series(r_high, w_years, hold_zero_years=hold_n)

        ei_path1 = compound_with_optional_cap(ei_2022, r_low_eff, cap_value)
        ei_path2 = compound_with_optional_cap(ei_2022, r_high_eff, cap_value)

        ei_upper = np.maximum(ei_path1, ei_path2)
        ei_lower = np.minimum(ei_path1, ei_path2)

        for y, e_up, e_lo in zip(years, ei_upper, ei_lower):
            traj_rows.append([code, y, ssp, grp, ei_2022, r_low, r_high, e_up, e_lo])

    traj = pd.DataFrame(
        traj_rows,
        columns=[
            "Code",
            "Year",
            "SSP",
            "income_group",
            "EI_2022",
            "rate_low",
            "rate_high",
            "EI_upper",
            "EI_lower",
        ],
    )
    traj["Code"] = traj["Code"].astype(code_cat)
    traj = traj.sort_values(["Code", "Year"]).reset_index(drop=True)
    traj["Code"] = traj["Code"].astype(str)

    out_ei = os.path.join(OUT_DIR, f"EI_{ssp}_2023_2100_interval.csv")
    traj.to_csv(out_ei, index=False, encoding="utf-8-sig")
    print(f"[Done] {ssp} -> {out_ei} | {out_r}")

print("\nAll done! Outputs in:", OUT_DIR)
