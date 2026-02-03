import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

INPUT_CSV = r"Data_Prefecture_Total_Missing.csv"

RESULT_DIR = Path("./RFResult")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULT_DIR / "Data_Prefecture_Total_Missing_filled.csv"
METRICS_CSV = RESULT_DIR / "RF_eval_metrics.csv"
EVAL_DETAIL_CSV = RESULT_DIR / "RF_eval_detail.csv"
MODEL_DIR = RESULT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DECREASING_COL = "Fixed_line_telephone_users"

df_orig = pd.read_csv(INPUT_CSV)
df = df_orig.copy()

code_col = df.columns[2]
year_col = df.columns[3]
reg_pop_col = df.columns[6]
target_cols = list(df.columns[7:18])

print("[Info] Code column:", code_col)
print("[Info] Year column:", year_col)
print("[Info] Registered_Population column:", reg_pop_col)
print("[Info] Target columns to impute (cols 8-18):", target_cols)

if DECREASING_COL not in target_cols:
    print(f"[Warn] {DECREASING_COL} was not found in target columns. Please verify the column name.")

base_feature_cols = [code_col, year_col, reg_pop_col] + list(df.columns[18:])
print("[Info] Base feature columns:", base_feature_cols)

if df[base_feature_cols].isna().any().any():
    raise ValueError("Missing values found in base feature columns. Please handle them before running this script.")

df[code_col] = pd.to_numeric(df[code_col], errors="coerce")
df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
df_orig[code_col] = pd.to_numeric(df_orig[code_col], errors="coerce")
df_orig[year_col] = pd.to_numeric(df_orig[year_col], errors="coerce")

code_means = {}
prefix4_means = {}
prefix2_means = {}
global_means = {}

for col in target_cols:
    tmp = df_orig[[code_col, col]].dropna()
    if tmp.empty:
        code_means[col] = {}
        prefix4_means[col] = {}
        prefix2_means[col] = {}
        global_means[col] = np.nan
        continue

    tmp = tmp.copy()
    tmp["code_str"] = tmp[code_col].astype(int).astype(str).str.zfill(6)
    tmp["p4"] = tmp["code_str"].str[:4]
    tmp["p2"] = tmp["code_str"].str[:2]

    code_means[col] = tmp.groupby("code_str")[col].mean().to_dict()
    prefix4_means[col] = tmp.groupby("p4")[col].mean().to_dict()
    prefix2_means[col] = tmp.groupby("p2")[col].mean().to_dict()
    global_means[col] = tmp[col].mean()


def get_fallback_mean(col, code_val):
    if col not in global_means:
        return np.nan

    global_mean = global_means[col]

    if pd.isna(code_val):
        return global_mean

    try:
        code_int = int(code_val)
    except Exception:
        return global_mean

    code_str = f"{code_int:06d}"

    d_code = code_means.get(col, {})
    if code_str in d_code:
        return d_code[code_str]

    p4 = code_str[:4]
    d4 = prefix4_means.get(col, {})
    if p4 in d4:
        return d4[p4]

    p2 = code_str[:2]
    d2 = prefix2_means.get(col, {})
    if p2 in d2:
        return d2[p2]

    return global_mean


def impute_with_dynamic_rf(df_in, return_models=False):
    df_imp = df_in.copy()
    imputed_masks = {col: pd.Series(False, index=df_imp.index) for col in target_cols}

    missing_counts = {col: df_imp[col].isna().sum() for col in target_cols}
    sorted_targets = sorted(target_cols, key=lambda c: missing_counts[c])

    print("\n[Impute-Core] Missing counts per target column (fewest to most):")
    for col in sorted_targets:
        print(f"  {col}: {missing_counts[col]}")

    dynamic_feature_cols = base_feature_cols.copy()
    for col in target_cols:
        if missing_counts[col] == 0 and col not in dynamic_feature_cols:
            dynamic_feature_cols.append(col)

    print("\n[Impute-Core] Initial dynamic feature columns:", dynamic_feature_cols)

    rf_models = {}

    for col in sorted_targets:
        n_missing = missing_counts[col]
        if n_missing == 0:
            print(f"\n[Impute-Core] Column {col} has no missing values. Skipping imputation (still usable as a feature).")
            if col not in dynamic_feature_cols:
                dynamic_feature_cols.append(col)
            continue

        print(f"\n[Impute-Core] Imputing column {col}, missing count = {n_missing}")

        mask_missing = df_imp[col].isna()
        mask_not_missing = ~mask_missing

        feature_cols_for_this = [c for c in dynamic_feature_cols if c != col]

        X_train = df_imp.loc[mask_not_missing, feature_cols_for_this]
        y_train = df_imp.loc[mask_not_missing, col]
        X_pred = df_imp.loc[mask_missing, feature_cols_for_this]

        if X_train.shape[0] == 0:
            print(f"[Impute-Core-Warn] No training samples for column {col}. Falling back to hierarchical mean imputation.")
            for idx in X_pred.index:
                code_val = df_imp.loc[idx, code_col]
                fb = get_fallback_mean(col, code_val)
                df_imp.at[idx, col] = fb
                imputed_masks[col].at[idx] = True
        else:
            if X_train.shape[0] < 10:
                print(f"[Impute-Core-Warn] Very few training samples for column {col} ({X_train.shape[0]} rows). Use with caution.")

            rf = RandomForestRegressor(
                n_estimators=1000,
                random_state=3407,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_pred)
            y_pred_rounded = np.round(y_pred).astype(float)

            for j, row_idx in enumerate(X_pred.index):
                if y_pred_rounded[j] < 0:
                    code_val = df_imp.loc[row_idx, code_col]
                    fb = get_fallback_mean(col, code_val)
                    if not np.isnan(fb):
                        y_pred_rounded[j] = fb

            df_imp.loc[mask_missing, col] = y_pred_rounded
            imputed_masks[col].loc[mask_missing] = True

            if return_models:
                rf_models[col] = rf

            print(f"[Impute-Core] Column {col} imputed. Example predictions (first 5): {y_pred_rounded[:5]}")

        if col not in dynamic_feature_cols:
            dynamic_feature_cols.append(col)
        print(f"[Impute-Core] Current dynamic feature column count: {len(dynamic_feature_cols)}")

    return df_imp, imputed_masks, rf_models


def apply_time_constraints(df_in, imputed_masks):
    df_out = df_in.copy()

    print("\n[Constraint] Applying time-series constraints...")
    print("  Increasing features: +2% to +15% per year (when previous year > 0)")
    print(f"  Decreasing feature {DECREASING_COL}: -4% to -20% per year (when previous year > 0)")

    for col in target_cols:
        mask_imputed = imputed_masks.get(col, None)
        if mask_imputed is None or not mask_imputed.any():
            continue

        print(f"[Constraint] Processing target column: {col}")

        for code, idx in df_out.groupby(code_col).groups.items():
            idx = list(idx)
            sub = df_out.loc[idx, [year_col, col]].sort_values(by=year_col)
            sorted_idx = sub.index

            last_val = None

            for row_idx in sorted_idx:
                val = df_out.at[row_idx, col]
                if pd.isna(val):
                    continue

                is_imputed = bool(mask_imputed.loc[row_idx])

                if not is_imputed:
                    last_val = val
                    continue

                if last_val is None:
                    adj = float(int(round(val)))
                else:
                    adj = float(val)
                    if last_val > 0:
                        if col == DECREASING_COL:
                            min_allowed = last_val * 0.80
                            max_allowed = last_val * 0.96
                            if adj > max_allowed:
                                adj = max_allowed
                            if adj < min_allowed:
                                adj = min_allowed
                        else:
                            min_allowed = last_val * 1.02
                            max_allowed = last_val * 1.15
                            if adj < min_allowed:
                                adj = min_allowed
                            if adj > max_allowed:
                                adj = max_allowed
                    adj = float(int(round(adj)))

                df_out.at[row_idx, col] = adj
                last_val = adj

    return df_out


print("\n[Eval] Starting evaluation: hide ~10% of non-missing values per target column, then run full imputation + constraints...")

rng = np.random.RandomState(3407)
metrics_list = []
eval_detail_rows = []

for eval_col in target_cols:
    non_missing_idx = df_orig[~df_orig[eval_col].isna()].index.to_numpy()
    n_total_non_missing = len(non_missing_idx)

    if n_total_non_missing == 0:
        print(f"[Eval] Column {eval_col} is entirely missing. Skipping evaluation.")
        metrics_list.append({
            "target_col": eval_col,
            "n_total_non_missing": 0,
            "n_eval": 0,
            "R2": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "WMAPE_percent": np.nan,
        })
        continue

    n_eval = max(1, int(n_total_non_missing * 0.10))
    eval_idx = rng.choice(non_missing_idx, size=n_eval, replace=False)

    y_true = df_orig.loc[eval_idx, eval_col].astype(float).to_numpy()

    df_eval = df_orig.copy()
    df_eval.loc[eval_idx, eval_col] = np.nan

    df_eval_imp, imputed_masks_eval, _ = impute_with_dynamic_rf(df_eval, return_models=False)
    df_eval_imp = apply_time_constraints(df_eval_imp, imputed_masks_eval)

    y_pred = df_eval_imp.loc[eval_idx, eval_col].astype(float).to_numpy()

    try:
        r2 = r2_score(y_true, y_pred)
    except ValueError:
        r2 = np.nan

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    total_abs_err = np.abs(y_true - y_pred).sum()
    total_actual = np.abs(y_true).sum()
    wmape = total_abs_err / total_actual * 100.0 if total_actual > 0 else np.nan

    metrics_list.append({
        "target_col": eval_col,
        "n_total_non_missing": int(n_total_non_missing),
        "n_eval": int(n_eval),
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "WMAPE_percent": float(wmape) if np.isfinite(wmape) else np.nan,
    })

    for i, idx in enumerate(eval_idx):
        true_val = float(y_true[i])
        pred_val = float(y_pred[i])
        abs_err = abs(pred_val - true_val)
        rel_err = abs_err / true_val if true_val != 0 else np.nan

        eval_detail_rows.append({
            "target_col": eval_col,
            "row_index": int(idx),
            code_col: df_orig.loc[idx, code_col],
            year_col: df_orig.loc[idx, year_col],
            "y_true": true_val,
            "y_pred": pred_val,
            "abs_error": abs_err,
            "rel_error": rel_err,
        })

    r2_str = f"{r2:.4f}" if np.isfinite(r2) else "nan"
    wmape_str = f"{wmape:.2f}" if np.isfinite(wmape) else "nan"
    print(
        f"[Eval] Column {eval_col}: n_eval={n_eval}, "
        f"R2={r2_str}, MAE={mae:.4f}, RMSE={rmse:.4f}, WMAPE={wmape_str}%"
    )

metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"\n[Eval] Evaluation metrics saved to: {METRICS_CSV}")

if eval_detail_rows:
    eval_detail_df = pd.DataFrame(eval_detail_rows)
    eval_detail_df.to_csv(EVAL_DETAIL_CSV, index=False, encoding="utf-8-sig")
    print(f"[Eval] Evaluation sample details saved to: {EVAL_DETAIL_CSV}")

print("\n[Impute-Final] Running final imputation on original data and saving models...")

df_filled, imputed_masks_final, rf_models = impute_with_dynamic_rf(df, return_models=True)
df_filled_constrained = apply_time_constraints(df_filled, imputed_masks_final)

df_filled_constrained.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"[Impute-Final] Imputation + time constraints complete. Output saved to: {OUTPUT_CSV}")

print("[Model] Saving Random Forest models for each target column...")
for col, model in rf_models.items():
    if model is None:
        continue
    safe_col = re.sub(r"[^0-9a-zA-Z_]+", "_", col)
    model_path = MODEL_DIR / f"RF_model_{safe_col}.joblib"
    dump(model, model_path)
    print(f"  Saved model: {model_path}")

print("\n[All Done] Evaluation, imputed output, and models have been saved under ./RFResult")
