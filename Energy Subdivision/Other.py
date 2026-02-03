import os
import sys
import traceback
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

# ---------- 文件路径（按需修改） ----------
SUMMARY_PATH = "Summary_Energy_Statistical_Yearbook_Provinces.csv"
COUNTY_PATH  = "Other.csv"          # ← “其他行业”的县级指标文件
CORR_PATH    = "Corr_Result.csv"    # 年度相关性结果CSV（Year, rho；可含 total）

# ---------- 输出文件 ----------
OUT_ALLOC   = "WA_other_allocation.csv"
OUT_WEIGHTS = "WA_other_weights.csv"
OUT_QUALITY = "WA_other_quality.csv"
OUT_ALPHA   = "WA_alpha_by_year.csv"
ERR_LOG     = "WA_other_error.log"

# ---------- 行业键（省级汇总表 Industry_Category 的取值） ----------
INDUSTRY_KEY = "Other"

# ---------- 工程初始权重（W0，和=1） ----------
W0 = {
    "Fixed_line_telephone_users": 0.30,
    "General_budget_revenue_of_local_finance": 0.20,
    "General_budget_expenditure_of_local_finance": 0.20,
    "Urban_Proportion": 0.10,
    "Light_Data": 0.20,
}

# ---------- 软门控（仅 Urban_Proportion） ----------
SOFT_GATING = True
GATE = dict(min_urban=0.02, penalty_low_urban=0.3)  # 城镇占比过低则整体打折

# ---------- ρ→α（按你的规则） ----------
def rho_to_alpha(rho):
    """
    ρ <= 0.5 → α=0（不信夜光）
    ρ >= 0.9 → α=1（完全信夜光）
    0.5 < ρ < 0.9 → 线性插值到 (0,1)
    """
    if rho <= 0.5:
        return 0.0
    elif rho >= 0.9:
        return 1.0
    else:
        return float((rho - 0.5) / 0.4)

LAMBDA_SHRINK = 0.3  # 回到先验的温和收缩

# ---------- 数值&打印设置 ----------
EPS = 1e-12
MAX_ITERS = 500
TOL = 1e-8

OPT_METHOD = "SLSQP"
PROGRESS_EVERY = 10
VERBOSE_GROUP = True
VERBOSE_ITER  = False
ITER_PRINT_EVERY = 5
WARN_ON_NO_IMPROVE = True

def log(msg):
    print(msg, flush=True)

# ============ 工具函数 ============
def minmax01(x):
    x = np.asarray(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def soft_gate_row(row):
    """仅基于 Urban_Proportion 的软门控"""
    if not SOFT_GATING:
        return 1.0
    urban = float(row.get("Urban_Proportion", 0.0) or 0.0)
    g = 1.0
    if urban < GATE["min_urban"]:
        g *= GATE["penalty_low_urban"]
    return max(0.0, min(1.0, g))

def project_box_simplex(y, L, U, s=1.0):
    L = np.asarray(L, float); U = np.asarray(U, float); y = np.asarray(y, float)
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

# ============ IO ============
def load_inputs():
    sum_df = pd.read_csv(SUMMARY_PATH, dtype={"Year": int}, encoding="utf-8-sig")
    cty_df = pd.read_csv(COUNTY_PATH,  dtype={"Year": int}, encoding="utf-8-sig")
    energy_cols = list(sum_df.columns[4:])

    if "Industry_Category" not in sum_df.columns:
        raise ValueError("省级表缺少列：Industry_Category")

    ind_df = sum_df.loc[sum_df["Industry_Category"] == INDUSTRY_KEY,
                        ["Year", "Province_ID"] + energy_cols].copy()

    if ind_df.empty:
        cats = sum_df["Industry_Category"].dropna().unique().tolist()
        raise ValueError(
            f"省级表中找不到行业名 '{INDUSTRY_KEY}'。可用 Industry_Category 样例（前10个）：{cats[:10]}"
        )

    for e in energy_cols:
        ind_df[e] = pd.to_numeric(ind_df[e], errors="coerce").fillna(0.0)

    log(f"[INFO] 读入：省级表 {len(sum_df):,} 行（Other {len(ind_df):,}）；县级表 {len(cty_df):,} 行；能源列 {len(energy_cols)} 个")
    return ind_df, cty_df, energy_cols

def ensure_groups(cty_df, sum_ind):
    pairs = set(zip(sum_ind["Province_ID"], sum_ind["Year"]))
    cty_df["__pair__"] = list(zip(cty_df["Province_ID"], cty_df["Year"]))
    before = len(cty_df)
    cty_df = cty_df[cty_df["__pair__"].isin(pairs)].drop(columns="__pair__").copy()
    log(f"[INFO] 县级数据匹配到省×年后：{len(cty_df):,} 行（丢弃 {before - len(cty_df):,} 行未匹配）")
    return cty_df

def read_alpha_from_corr_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到相关性CSV：{path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError("相关性CSV 至少两列：Year 与 相关性值")

    year_col = df.columns[0]
    val_col  = df.columns[1]
    df = df[[year_col, val_col]].copy()
    df[year_col] = df[year_col].astype(str).str.strip()
    df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")

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
    log(f"[OK] 已读取相关性CSV并生成 α：{os.path.abspath(OUT_ALPHA)}")
    return alpha_map, rho_map, alpha_total, rho_total, out_df

# ============ 主流程 ============
def main():
    log(f"[INFO] 选择的优化器方法：{OPT_METHOD}")
    sum_ind, cty_df, energy_cols = load_inputs()

    feat_cols = [
        "Fixed_line_telephone_users",
        "General_budget_revenue_of_local_finance",
        "General_budget_expenditure_of_local_finance",
        "Urban_Proportion",
        "Light_Data",
    ]
    for c in feat_cols:
        if c not in cty_df.columns:
            raise ValueError(f"县级表缺少列：{c}")

    cty_df = ensure_groups(cty_df, sum_ind)
    alpha_map, rho_map, alpha_total, rho_total, _ = read_alpha_from_corr_csv(CORR_PATH)

    log("[INFO] 计算软门控 g（仅 Urban_Proportion）...")
    cty_df["__gate__"] = cty_df.apply(soft_gate_row, axis=1).astype(float)

    weight_rows, quality_rows, alloc_frames = [], [], []

    total_groups = 0
    ok_groups = 0
    fallback_groups = 0
    ce_improved_cnt = 0
    iter_list = []

    feat_order = list(W0.keys())
    w0_vec = np.array([W0[f] for f in feat_order], dtype=float)
    if not (abs(w0_vec.sum() - 1.0) < 1e-9):
        raise ValueError(f"W0 权重之和必须为 1，当前为 {w0_vec.sum()}")

    grp = cty_df.groupby(["Province_ID", "Year"], dropna=False, sort=True)
    ng = grp.ngroups
    log(f"[INFO] 省×年 组数：{ng}")
    if ng == 0:
        log("[WARN] 找不到任何省×年组合，脚本结束。")
        return

    for gi, ((prov, year), gdf) in enumerate(grp, start=1):
        total_groups += 1
        year = int(year)

        if gi % PROGRESS_EVERY == 1 or gi == ng:
            log(f"[INFO] 处理进度：{gi}/{ng}  当前组=({prov}, {year})  县数={len(gdf)}")

        alpha = float(alpha_map.get(year, alpha_total if alpha_total is not None else 0.0))
        rho_input = float(rho_map.get(year, rho_total if rho_total is not None else np.nan))

        # 组内特征预处理（全部 Min–Max；Light_Data 不做 log1p）
        X = gdf[feat_cols].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
            X[col] = minmax01(X[col].values)

        g_vec = gdf["__gate__"].values.astype(float)

        # 先验份额 p0
        s0 = g_vec * np.dot(X[feat_order].values, w0_vec)
        p0 = (np.full(len(gdf), 1.0 / len(gdf)) if s0.sum() <= 0 else s0 / s0.sum())

        # 夜光目标份额 q
        q = X["Light_Data"].values.astype(float)
        q = (np.full(len(gdf), 1.0 / len(gdf)) if q.sum() <= 0 else q / (q.sum() + EPS))

        # 盒约束
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

        # 优化
        w_init = w0_vec.copy()
        res = minimize(
            obj, w_init, method=OPT_METHOD, bounds=bounds, constraints=cons,
            options={"maxiter": MAX_ITERS, "ftol": TOL, "disp": False},
            callback=cb if OPT_METHOD.upper() == "SLSQP" else None
        )

        if not res.success or not np.isfinite(getattr(res, "fun", np.nan)):
            # 兜底：CRITIC → 与 w0 按 α 混合 → 投影
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

        # 诊断
        sum_resid = float(abs(w_star.sum() - 1.0))
        max_low_violation = float(np.maximum(0.0, lower - w_star).max())
        max_high_violation = float(np.maximum(0.0, w_star - upper).max())
        max_bound_violation = max(max_low_violation, max_high_violation)
        max_delta_pct = float(np.max(np.abs((w_star - w0_vec) / (w0_vec + EPS))))

        # 质量指标
        p_star = p_of_w(w_star)
        ce0 = cross_entropy(q, p0)
        ce1 = cross_entropy(q, p_star)
        try:
            rho0, _ = spearmanr(q, p0)
            rho1, _ = spearmanr(q, p_star)
        except Exception:
            rho0, rho1 = np.nan, np.nan
        mae_prior = float(np.mean(np.abs(p_star - p0)))
        improved = (ce1 < ce0) or (np.isnan(ce0) and not np.isnan(ce1))
        if improved:
            ce_improved_cnt += 1

        if VERBOSE_GROUP:
            log(
f"""[DIAG] 组=({prov}, {year})  α={alpha:.3f}  ρ_in={rho_input if np.isfinite(rho_input) else np.nan}
       方法={OPT_METHOD}  成功={status=='ok'}  nit={nit} nfev={nfev} njev={njev}  fun={fun}
       约束残差|sum(w)-1|={sum_resid:.3e}  边界最大偏离={max_bound_violation:.3e}  最大相对变动={max_delta_pct:.1%}
       交叉熵(前/后)={ce0:.6g} → {ce1:.6g}  {'✅改进' if improved else '— 无改进'}
       相关(前/后)={rho0 if np.isfinite(rho0) else np.nan:.3f} → {rho1 if np.isfinite(rho1) else np.nan:.3f}
       与先验份额MAE={mae_prior:.6g}"""
            )
            if WARN_ON_NO_IMPROVE and (alpha > 0) and (not improved):
                log("   ⚠️  α>0 但交叉熵未改善：可能该组夜光信号弱/指标已很贴近/±20%边界限制太紧。")

        # 保存权重（逐指标）
        for j, feat in enumerate(feat_order):
            w0j = float(w0_vec[j])
            wj = float(w_star[j])
            weight_rows.append({
                "Province_ID": prov, "Year": year, "Feature": feat,
                "w0": w0j, "w_star": wj,
                "delta_pct": (wj - w0j) / (w0j + EPS),
                "alpha_used": alpha, "rho_input": rho_input, "status": status,
                "opt_method": OPT_METHOD, "opt_nit": nit, "opt_nfev": nfev, "opt_njev": njev,
                "sum_residual": sum_resid, "max_bound_violation": max_bound_violation
            })

        # 县级分配
        tgt = sum_ind[(sum_ind["Province_ID"] == prov) & (sum_ind["Year"] == year)]
        if not tgt.empty:
            tgt_vec = tgt[energy_cols].values.astype(float).reshape(1, -1)
            out_block = gdf[["Province_ID", "Year", "Code"]].copy()
            for j, e in enumerate(energy_cols):
                out_block[f"{e}_alloc"] = p_star * float(tgt_vec[0, j])
            alloc_frames.append(out_block)

        # 质量行
        quality_rows.append({
            "Province_ID": prov, "Year": year,
            "alpha_used": alpha, "rho_input": rho_input,
            "cross_entropy_prior": ce0, "cross_entropy_after": ce1,
            "spearman_with_q_prior": float(rho0) if np.isfinite(rho0) else np.nan,
            "spearman_with_q_after": float(rho1) if np.isfinite(rho1) else np.nan,
            "MAE_vs_prior_share": mae_prior, "status": status,
            "opt_method": OPT_METHOD, "opt_success": (status == "ok"),
            "opt_nit": nit, "opt_nfev": nfev, "opt_njev": njev,
            "sum_residual": sum_resid, "max_bound_violation": max_bound_violation,
            "improved": bool(improved)
        })

    # 写出结果
    weights_df = pd.DataFrame(weight_rows)
    weights_df.to_csv(OUT_WEIGHTS, index=False, encoding="utf-8-sig")
    log(f"[OK] 学习后的工程权重：{os.path.abspath(OUT_WEIGHTS)}  行数={len(weights_df):,}")

    alloc_df = pd.concat(alloc_frames, axis=0, ignore_index=True) if alloc_frames else pd.DataFrame()
    if not alloc_df.empty:
        targets = sum_ind.set_index(["Province_ID", "Year"])[energy_cols]
        alloc_df = group_scale_to_targets(alloc_df, targets, energy_cols, alloc_col_suffix="_alloc")
        alloc_df.to_csv(OUT_ALLOC, index=False, encoding="utf-8-sig")
        log(f"[OK] 县级分配：{os.path.abspath(OUT_ALLOC)}  行数={len(alloc_df):,}")
    else:
        log("[WARN] 无可写出的县级分配（alloc_frames 为空）")

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(OUT_QUALITY, index=False, encoding="utf-8-sig")
    log(f"[OK] 质量报告：{os.path.abspath(OUT_QUALITY)}  行数={len(quality_df):,}")

    # 期末汇总
    if total_groups > 0:
        avg_nit = np.mean([x for x in iter_list if np.isfinite(x)]) if iter_list else np.nan
        improve_rate = ce_improved_cnt / total_groups
        log(
f"""[SUMMARY]
   选择的优化器：{OPT_METHOD}
   组总数={total_groups}  成功(ok)={ok_groups}  兜底(fallback)={fallback_groups}
   平均迭代步数(nit)={avg_nit if np.isfinite(avg_nit) else np.nan:.2f}
   交叉熵改善的组比例={improve_rate:.1%}"""
        )

if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        sys.stderr.write("\n[ERROR] 脚本异常退出！\n")
        sys.stderr.write(tb + "\n")
        with open(ERR_LOG, "w", encoding="utf-8") as f:
            f.write(tb)
        print(f"[ERROR] 详见错误日志：{os.path.abspath(ERR_LOG)}", flush=True)
        raise
