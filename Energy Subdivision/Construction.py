import os
import sys
import traceback
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

SUMMARY_PATH = "Summary_Energy_Statistical_Yearbook_Provinces.csv"
COUNTY_PATH = "Construction.csv"
CORR_PATH = "Corr_Result.csv"

OUT_ALLOC = "WA_cons_allocation.csv"
OUT_WEIGHTS = "WA_cons_weights.csv"
OUT_QUALITY = "WA_cons_quality.csv"
OUT_ALPHA = "WA_alpha_by_year.csv"
ERR_LOG = "WA_error.log"

INDUSTRY_KEY = "Construction"

W0 = {
    "Registered_Population": 0.20,
    "Investment_in_fixed_assets": 0.35,
    "General_budget_expenditure_of_local_finance": 0.15,
    "Urban_Proportion": 0.15,
    "Bare_areas_Proportion": 0.05,
    "Light_Data": 0.10,
}

SOFT_GATING = True
GATE = dict(min_urban=0.02, penalty_low_urban=0.3)

def rho_to_alpha(rho):
    if rho <= 0.5:
        return 0.0
    elif rho >= 0.9:
        return 1.0
    else:
        return float((rho - 0.5) / 0.4)

LAMBDA_SHRINK = 0.3

EPS = 1e-12
MAX_ITERS = 500
TOL = 1e-8

OPT_METHOD = "SLSQP"
PROGRESS_EVERY = 10
VERBOSE_GROUP = True
VERBOSE_ITER = False
ITER_PRINT_EVERY = 5
WARN_ON_NO_IMPROVE = True

def log(msg):
    print(msg, flush=True)

def minmax01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def soft_gate_row(row):
    if not SOFT_GATING:
        return 1.0
    urban = float(row.get("Urban_Proportion", 0.0) or 0.0)
    g = 1.0
    if urban < GATE["min_urban"]:
        g *= GATE["penalty_low_urban"]
    return max(0.0, min(1.0, g))

def project_box_simplex(y, L, U, s=1.0):
    L = np.asarray(L, float)
    U = np.asarray(U, float)
    y = np.asarray(y, float)
    assert L.shape == U.shape == y.shape
    left, right = -1e6, 1e6

    def sum_clip(tau):
        return np.clip(y - tau, L, U).sum()

    for _ in range(60):
        mid = 0.5 * (left + right)
        if sum_clip(mid) > s:
            left = mid
        else:
            right = mid

    x = np.clip(y - 0.5 * (left + right), L, U)
    diff = s - x.sum()
    if abs(diff) > 1e-9:
        free = (x > L + 1e-12) & (x < U - 1e-12)
        if free.any():
            x[free] += diff / free.sum()
            x = np.clip(x, L, U)
    return x

def critic_weights(X):
    X = np.asarray(X, float)
    n, k = X.shape
    if n == 0 or k == 0:
        return np.full(k, 1.0 / k)
    std = X.std(axis=0, ddof=1)
    if np.all(std < 1e-12):
        return np.full(k, 1.0 / k)
    Xc = (X - X.mean(axis=0)) / (std + 1e-12)
    R = (Xc.T @ Xc) / max(n - 1, 1)
    R = np.clip(R, -1.0, 1.0)
    C = std * np.sum(1.0 - np.abs(R), axis=1)
    C = np.clip(C, 0.0, None)
    if C.sum() <= 0:
        return np.full(k, 1.0 / k)
    return C / C.sum()

def cross_entropy(q, p):
    return -np.sum(np.clip(q, EPS, None) * np.log(np.clip(p, EPS, None)))

def group_scale_to_targets(df, targets, energy_cols, alloc_col_suffix="_alloc"):
    def _scale_one(g):
        key = (g["Province_ID"].iloc[0], g["Year"].iloc[0])
        if key not in targets.index:
            return g
        tgt = targets.loc[key, energy_cols].values.astype(float)
        for j, e in enumerate(energy_cols):
            col = f"{e}{alloc_col_suffix}"
            cur = g[col].sum()
            if cur > 0:
                g[col] = g[col] * (tgt[j] / cur)
        return g

    return df.groupby(["Province_ID", "Year"], dropna=False).apply(_scale_one).reset_index(drop=True)

def load_inputs():
    sum_df = pd.read_csv(SUMMARY_PATH, dtype={"Year": int}, encoding="utf-8-sig")
    cty_df = pd.read_csv(COUNTY_PATH, dtype={"Year": int}, encoding="utf-8-sig")
    energy_cols = list(sum_df.columns[4:])
    cons = sum_df.loc[
        sum_df["Industry_Category"] == INDUSTRY_KEY,
        ["Year", "Province_ID"] + energy_cols
    ].copy()
    for e in energy_cols:
        cons[e] = pd.to_numeric(cons[e], errors="coerce").fillna(0.0)
    log(f"[INFO] Loaded: provincial={len(sum_df):,} rows (construction={len(cons):,}); county={len(cty_df):,} rows; energy_cols={len(energy_cols)}")
    return cons, cty_df, energy_cols

def ensure_groups(cty_df, sum_cons):
    pairs = set(zip(sum_cons["Province_ID"], sum_cons["Year"]))
    cty_df["__pair__"] = list(zip(cty_df["Province_ID"], cty_df["Year"]))
    before = len(cty_df)
    cty_df = cty_df[cty_df["__pair__"].isin(pairs)].drop(columns="__pair__").copy()
    log(f"[INFO] County rows after matching province×year: {len(cty_df):,} (dropped {before - len(cty_df):,} unmatched rows)")
    return cty_df

def read_alpha_from_corr_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Correlation CSV not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError("Correlation CSV must have at least two columns: Year and correlation value.")
    year_col = df.columns[0]
    val_col = df.columns[1]
    df = df[[year_col, val_col]].copy()
    df[year_col] = df[year_col].astype(str).str.strip()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    rho_total = None
    alpha_total = None
    alpha_map = {}
    rho_map = {}
    rows = []

    for _, r in df.iterrows():
        y_raw = r[year_col]
        val = r[val_col]
        if pd.isna(val):
            continue
        rho = float(np.clip(val, -1.0, 1.0))
        if y_raw.lower() == "total":
            rho_total = rho
            alpha_total = rho_to_alpha(rho)
            rows.append({"Year": "total", "rho": rho, "alpha": alpha_total})
        else:
            try:
                y = int(float(y_raw))
            except Exception:
                continue
            a = rho_to_alpha(rho)
            alpha_map[y] = a
            rho_map[y] = rho
            rows.append({"Year": y, "rho": rho, "alpha": a})

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df["__order__"] = out_df["Year"].apply(lambda x: 9999 if str(x) == "total" else int(x))
        out_df = out_df.sort_values("__order__").drop(columns="__order__")

    out_df.to_csv(OUT_ALPHA, index=False, encoding="utf-8-sig")
    log(f"[OK] Alpha generated from correlation CSV: {os.path.abspath(OUT_ALPHA)}")
    return alpha_map, rho_map, alpha_total, rho_total, out_df

def main():
    log(f"[INFO] Optimizer method: {OPT_METHOD}")
    sum_cons, cty_df, energy_cols = load_inputs()

    feat_cols = [
        "Registered_Population",
        "Investment_in_fixed_assets",
        "General_budget_expenditure_of_local_finance",
        "Urban_Proportion",
        "Bare_areas_Proportion",
        "Light_Data",
    ]
    for c in feat_cols:
        if c not in cty_df.columns:
            raise ValueError(f"Missing column in county file: {c}")

    cty_df = ensure_groups(cty_df, sum_cons)
    alpha_map, rho_map, alpha_total, rho_total, _ = read_alpha_from_corr_csv(CORR_PATH)

    log("[INFO] Computing soft gate g ...")
    cty_df["__gate__"] = cty_df.apply(soft_gate_row, axis=1).astype(float)

    weight_rows = []
    quality_rows = []
    alloc_frames = []

    total_groups = 0
    ok_groups = 0
    fallback_groups = 0
    ce_improved_cnt = 0
    iter_list = []

    feat_order = list(W0.keys())
    w0_vec = np.array([W0[f] for f in feat_order], dtype=float)
    if not (abs(w0_vec.sum() - 1.0) < 1e-9):
        raise ValueError("W0 weights must sum to 1.")

    grp = cty_df.groupby(["Province_ID", "Year"], dropna=False, sort=True)
    ng = grp.ngroups
    log(f"[INFO] Number of province×year groups: {ng}")
    if ng == 0:
        log("[WARN] No province×year groups found. Exiting.")
        return

    for gi, ((prov, year), gdf) in enumerate(grp, start=1):
        total_groups += 1
        year = int(year)
        if gi % PROGRESS_EVERY == 1 or gi == ng:
            log(f"[INFO] Progress: {gi}/{ng}  group=({prov}, {year})  counties={len(gdf)}")

        alpha = float(alpha_map.get(year, alpha_total if alpha_total is not None else 0.0))
        rho_input = float(rho_map.get(year, rho_total if rho_total is not None else np.nan))

        X = gdf[feat_cols].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
        for col in X.columns:
            X[col] = minmax01(X[col].values)

        g_vec = cty_df.loc[gdf.index, "__gate__"].values.astype(float)

        s0 = g_vec * np.dot(X[feat_order].values, w0_vec)
        p0 = (np.full(len(gdf), 1.0 / len(gdf)) if s0.sum() <= 0 else s0 / s0.sum())

        q = X["Light_Data"].values.astype(float)
        q = (np.full(len(gdf), 1.0 / len(gdf)) if q.sum() <= 0 else q / (q.sum() + EPS))

        lower = 0.8 * w0_vec
        upper = 1.2 * w0_vec
        bounds = [(float(lower[j]), float(upper[j])) for j in range(len(w0_vec))]
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)},)

        def p_of_w(w):
            s = g_vec * np.dot(X[feat_order].values, w)
            s = np.clip(s, 0.0, None)
            return (np.full_like(p0, 1.0 / len(p0)) if s.sum() <= 0 else s / (s.sum() + EPS))

        def obj(w):
            p = p_of_w(w)
            ce = cross_entropy(q, p)
            reg = np.sum((w - w0_vec) ** 2)
            return alpha * ce + (1 - alpha) * LAMBDA_SHRINK * reg

        iter_counter = {"n": 0}

        def cb(w):
            if not VERBOSE_ITER:
                return
            iter_counter["n"] += 1
            if (iter_counter["n"] <= 5) or (iter_counter["n"] % ITER_PRINT_EVERY == 0):
                p = p_of_w(w)
                ce = cross_entropy(q, p)
                reg = np.sum((w - w0_vec) ** 2)
                f = alpha * ce + (1 - alpha) * LAMBDA_SHRINK * reg
                log(f"    [ITER] ({prov}, {year}) it={iter_counter['n']} f={f:.6g} ce={ce:.6g} reg={reg:.6g} sum(w)={w.sum():.6f}")

        w_init = w0_vec.copy()
        res = minimize(
            obj,
            w_init,
            method=OPT_METHOD,
            bounds=bounds,
            constraints=cons,
            options={"maxiter": MAX_ITERS, "ftol": TOL, "disp": False},
            callback=cb if OPT_METHOD.upper() == "SLSQP" else None,
        )

        if (not res.success) or (not np.isfinite(res.fun)):
            w_critic = critic_weights(X[feat_order].values)
            y = (1 - alpha) * w0_vec + alpha * w_critic
            w_star = project_box_simplex(y, lower, upper, s=1.0)
            status = f"fallback:{res.message if hasattr(res, 'message') else 'no-success'}"
            fallback_groups += 1
            nit = getattr(res, "nit", np.nan)
            nfev = getattr(res, "nfev", np.nan)
            njev = getattr(res, "njev", np.nan)
            fun = np.nan
        else:
            w_star = project_box_simplex(res.x, lower, upper, s=1.0)
            status = "ok"
            ok_groups += 1
            nit = getattr(res, "nit", np.nan)
            nfev = getattr(res, "nfev", np.nan)
            njev = getattr(res, "njev", np.nan)
            fun = getattr(res, "fun", np.nan)

        if np.isfinite(nit):
            iter_list.append(nit)

        sum_resid = float(abs(w_star.sum() - 1.0))
        max_low_violation = float(np.maximum(0.0, lower - w_star).max())
        max_high_violation = float(np.maximum(0.0, w_star - upper).max())
        max_bound_violation = max(max_low_violation, max_high_violation)
        max_delta_pct = float(np.max(np.abs((w_star - w0_vec) / (w0_vec + EPS))))

        p_star = p_of_w(w_star)
        ce0 = cross_entropy(q, p0)
        ce1 = cross_entropy(q, p_star)
        try:
            rho0, _ = spearmanr(q, p0)
            rho1, _ = spearmanr(q, p_star)
        except Exception:
            rho0, rho1 = np.nan, np.nan
        mae_prior = float(np.mean(np.abs(p_star - p0)))
        improved = (ce1 < ce0) or (np.isnan(ce0) and (not np.isnan(ce1)))
        if improved:
            ce_improved_cnt += 1

        if VERBOSE_GROUP:
            log(
                f"[DIAG] group=({prov}, {year})  alpha={alpha:.3f}  rho_in={rho_input if np.isfinite(rho_input) else np.nan}\n"
                f"       method={OPT_METHOD}  success={status=='ok'}  nit={nit} nfev={nfev} njev={njev}  fun={fun}\n"
                f"       |sum(w)-1|={sum_resid:.3e}  max_bound_violation={max_bound_violation:.3e}  max_rel_change={max_delta_pct:.1%}\n"
                f"       cross_entropy (prior->after)={ce0:.6g} -> {ce1:.6g}  {'IMPROVED' if improved else 'NO-IMPROVE'}\n"
                f"       spearman (prior->after)={rho0 if np.isfinite(rho0) else np.nan:.3f} -> {rho1 if np.isfinite(rho1) else np.nan:.3f}\n"
                f"       MAE vs prior share={mae_prior:.6g}"
            )
            if WARN_ON_NO_IMPROVE and (alpha > 0) and (not improved):
                log("       [WARN] alpha>0 but cross-entropy did not improve. Nightlight may be weak or bounds may be too tight.")

        for j, feat in enumerate(feat_order):
            w0j = float(w0_vec[j])
            wj = float(w_star[j])
            weight_rows.append(
                {
                    "Province_ID": prov,
                    "Year": year,
                    "Feature": feat,
                    "w0": w0j,
                    "w_star": wj,
                    "delta_pct": (wj - w0j) / (w0j + EPS),
                    "alpha_used": alpha,
                    "rho_input": rho_input,
                    "status": status,
                    "opt_method": OPT_METHOD,
                    "opt_nit": nit,
                    "opt_nfev": nfev,
                    "opt_njev": njev,
                    "sum_residual": sum_resid,
                    "max_bound_violation": max_bound_violation,
                }
            )

        tgt = sum_cons[(sum_cons["Province_ID"] == prov) & (sum_cons["Year"] == year)]
        if not tgt.empty:
            tgt_vec = tgt[energy_cols].values.astype(float).reshape(1, -1)
            out_block = gdf[["Province_ID", "Year", "Code"]].copy()
            for j, e in enumerate(energy_cols):
                out_block[f"{e}_alloc"] = p_star * float(tgt_vec[0, j])
            alloc_frames.append(out_block)

        quality_rows.append(
            {
                "Province_ID": prov,
                "Year": year,
                "alpha_used": alpha,
                "rho_input": rho_input,
                "cross_entropy_prior": ce0,
                "cross_entropy_after": ce1,
                "spearman_with_q_prior": float(rho0) if np.isfinite(rho0) else np.nan,
                "spearman_with_q_after": float(rho1) if np.isfinite(rho1) else np.nan,
                "MAE_vs_prior_share": mae_prior,
                "status": status,
                "opt_method": OPT_METHOD,
                "opt_success": (status == "ok"),
                "opt_nit": nit,
                "opt_nfev": nfev,
                "opt_njev": njev,
                "sum_residual": sum_resid,
                "max_bound_violation": max_bound_violation,
                "improved": bool(improved),
            }
        )

    weights_df = pd.DataFrame(weight_rows)
    weights_df.to_csv(OUT_WEIGHTS, index=False, encoding="utf-8-sig")
    log(f"[OK] Learned weights saved: {os.path.abspath(OUT_WEIGHTS)}  rows={len(weights_df):,}")

    alloc_df = pd.concat(alloc_frames, axis=0, ignore_index=True) if alloc_frames else pd.DataFrame()
    if not alloc_df.empty:
        targets = sum_cons.set_index(["Province_ID", "Year"])[energy_cols]
        alloc_df = group_scale_to_targets(alloc_df, targets, energy_cols, alloc_col_suffix="_alloc")
        alloc_df.to_csv(OUT_ALLOC, index=False, encoding="utf-8-sig")
        log(f"[OK] County allocation saved: {os.path.abspath(OUT_ALLOC)}  rows={len(alloc_df):,}")
    else:
        log("[WARN] No county allocation to write (alloc_frames is empty).")

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(OUT_QUALITY, index=False, encoding="utf-8-sig")
    log(f"[OK] Quality report saved: {os.path.abspath(OUT_QUALITY)}  rows={len(quality_df):,}")

    if total_groups > 0:
        avg_nit = np.mean([x for x in iter_list if np.isfinite(x)]) if iter_list else np.nan
        improve_rate = ce_improved_cnt / total_groups
        log(
            f"[SUMMARY]\n"
            f"   optimizer={OPT_METHOD}\n"
            f"   groups_total={total_groups}  ok={ok_groups}  fallback={fallback_groups}\n"
            f"   avg_nit={avg_nit if np.isfinite(avg_nit) else np.nan:.2f}\n"
            f"   cross_entropy_improve_rate={improve_rate:.1%}"
        )

if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        sys.stderr.write("\n[ERROR] Script terminated due to an exception.\n")
        sys.stderr.write(tb + "\n")
        with open(ERR_LOG, "w", encoding="utf-8") as f:
            f.write(tb)
        print(f"[ERROR] See error log: {os.path.abspath(ERR_LOG)}", flush=True)
        raise
