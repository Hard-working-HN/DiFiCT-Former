# -*- coding: utf-8 -*-

import os
import math
import pickle
import warnings
import random
import platform
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler

import shap
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def iter_progress(iterable, desc=None, total=None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, ncols=80, leave=False)


warnings.filterwarnings("ignore")

USE_ALL_CPU = True
USE_GPU_FOR_REGION = True
USE_GPU_FOR_TIME = True
EXPLAIN_SIZE = 2048
HEAT_SAMPLES_PER_YEAR = 1024
SPATIAL_SAMPLES_PER_PROV = 128
REGION_SWAP_K = 64
TIME_SHUFFLE_K = 64
SCALE_FACTOR_TIME_REGION = 0.2

BEESWARM_TRIM_LOWER_Q = 1
BEESWARM_TRIM_UPPER_Q = 99

FILE_PATH = "Industry.csv"
WEIGHT_DIR = "weight"
BEST_MODEL_PATH = os.path.join(WEIGHT_DIR, "stcformer_best.pt")
PREP_PATH = os.path.join(WEIGHT_DIR, "preprocess_stcformer.pkl")

TRAIN_YEARS = list(range(2013, 2020))
VAL_YEARS = [2020]
TEST_YEARS = [2021, 2022]
TEST_YEARS_PREF = [2022, 2021]

SEED = 3704
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if USE_ALL_CPU:
    num_threads = max(1, os.cpu_count() or 1)
else:
    num_threads = 1
torch.set_num_threads(num_threads)

print("===== STCFormer SHAP (Permutation, district Code) =====", flush=True)
print(
    f"Python {platform.python_version()} | torch {torch.__version__} | shap {getattr(shap, '__version__', '?')}",
    flush=True,
)
print(f"Threads={torch.get_num_threads()} | USE_ALL_CPU={USE_ALL_CPU} | SEED={SEED}", flush=True)
print(
    f"EXPLAIN_SIZE={EXPLAIN_SIZE}, HEAT_SAMPLES_PER_YEAR={HEAT_SAMPLES_PER_YEAR}, "
    f"REGION_SWAP_K={REGION_SWAP_K}, TIME_SHUFFLE_K={TIME_SHUFFLE_K}",
    flush=True,
)


def compute_time_encoding(year_series: pd.Series) -> pd.DataFrame:
    ymin, ymax = year_series.min(), year_series.max()
    denom = (ymax - ymin) if ymax != ymin else 1.0
    yr_norm = (year_series - ymin) / denom
    return pd.DataFrame(
        {"year_sin": np.sin(2 * math.pi * yr_norm), "year_cos": np.cos(2 * math.pi * yr_norm)},
        index=year_series.index,
    )


def build_causal_mask(L: int, device=None) -> torch.Tensor:
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)


def map_with_categories(series: pd.Series, categories: List) -> pd.Series:
    cat_map = {str(v): i for i, v in enumerate(categories)}
    return series.map(lambda x: cat_map.get(str(x), -1)).astype(int)


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def trim_shap_tail(values: np.ndarray, data: np.ndarray, lower_q=5, upper_q=95):
    v = values.copy()
    d = data.copy()
    F = v.shape[1]
    for j in range(F):
        col = v[:, j]
        lo, hi = np.nanpercentile(col, [lower_q, upper_q])
        m = (col < lo) | (col > hi)
        v[m, j] = np.nan
        d[m, j] = np.nan
    return v, d


assert os.path.exists(PREP_PATH), f"Missing preprocess file: {PREP_PATH}"
with open(PREP_PATH, "rb") as f:
    prep = pickle.load(f)

scaler_X: StandardScaler = prep["scaler_X"]
scaler_y: StandardScaler = prep["scaler_y"]
prov_categories: List = prep["prov_categories"]
city_categories: List = prep["city_categories"]
feat_cols: List[str] = prep["feature_cols"]
year_col_saved = prep["year_col"]
y_col_saved = prep["y_col"]
district_col_saved = prep.get("district_col", "Code")
district_categories: List = prep.get("district_categories", None)

cfg = prep["config"]
D_MODEL = cfg["D_MODEL"]
NHEAD = cfg["NHEAD"]
NLAYERS = cfg["NLAYERS"]
FFN_DIM = cfg["FFN_DIM"]
DROPOUT = cfg["DROPOUT"]
PROV_EMB_DIM = cfg["PROV_EMB_DIM"]
CITY_EMB_DIM = cfg["CITY_EMB_DIM"]
FILM_SCALE_G = cfg["FILM_SCALE_G"]
FILM_SCALE_B = cfg["FILM_SCALE_B"]

df_raw = pd.read_csv(FILE_PATH)
assert district_col_saved in df_raw.columns, f"Missing district id column in raw data: {district_col_saved}"

year_col = year_col_saved
y_col = y_col_saved

x_cols = [c for c in feat_cols if c not in ("year_sin", "year_cos")]

possible_prov_cols = [
    c for c in df_raw.columns if c.lower().startswith("prov") or c.lower().endswith("province") or c == "Province_ID"
]
possible_city_cols = [c for c in df_raw.columns if c.lower().startswith("city") or c == "City_ID"]

prov_col, city_col = None, None
for c in df_raw.columns:
    if c in ["prov_code", "province", "Province", "Province_ID"] or "prov" in c.lower():
        prov_col = c
    if c in ["city_code", "city", "City", "City_ID"] or "city" in c.lower():
        city_col = c

if prov_col is None and possible_prov_cols:
    prov_col = possible_prov_cols[0]
if city_col is None and possible_city_cols:
    city_col = possible_city_cols[0]

assert prov_col is not None and city_col is not None, "Cannot locate province/city columns in raw data."

need_cols = [district_col_saved, year_col, prov_col, city_col] + x_cols + [y_col]
df = df_raw[need_cols].copy()

df[x_cols + [y_col]] = safe_numeric(df[x_cols + [y_col]])
df = df.dropna(subset=[district_col_saved, year_col, prov_col, city_col, y_col]).reset_index(drop=True)

te = compute_time_encoding(df[year_col])
df["year_sin"], df["year_cos"] = te["year_sin"], te["year_cos"]

if district_categories is None:
    raise RuntimeError("Missing district_categories in preprocess. Use preprocess from district-Code training run.")

df["prov_code"] = map_with_categories(df[prov_col], prov_categories)
df["city_code"] = map_with_categories(df[city_col], city_categories)
df["district_code"] = map_with_categories(df[district_col_saved], district_categories)

before = len(df)
df = df[(df["prov_code"] >= 0) & (df["city_code"] >= 0) & (df["district_code"] >= 0)].copy()
dropped = before - len(df)
if dropped > 0:
    print(f"[INFO] Dropped {dropped} row(s) not covered by training categories.", flush=True)

FALL = x_cols + ["year_sin", "year_cos"]
X_raw_all = df[FALL].values.astype(np.float32)
X_std_all = scaler_X.transform(X_raw_all)
for i, c in enumerate(FALL):
    df[c] = X_std_all[:, i]

segments = []
for d, g in df.groupby(df["district_code"], sort=False):
    if isinstance(d, tuple):
        d = d[0]
    g = g.sort_values(year_col)
    prov_mode = int(g["prov_code"].mode(dropna=False).iloc[0])
    city_mode = int(g["city_code"].mode(dropna=False).iloc[0])
    years = g[year_col].values.astype(int)
    X = g[FALL].values.astype(np.float32)
    y = scaler_y.transform(g[y_col].values.astype(np.float32).reshape(-1, 1))
    segments.append(dict(district=int(d), prov=prov_mode, city=city_mode, years=years, X=X, y=y))

for s in segments:
    yrs = s["years"]
    s["m_tr"] = np.isin(yrs, TRAIN_YEARS).astype(np.uint8)
    s["m_va"] = np.isin(yrs, VAL_YEARS).astype(np.uint8)
    s["m_te"] = np.isin(yrs, TEST_YEARS).astype(np.uint8)

test_segs = [s for s in segments if s["m_te"].any()]
assert len(test_segs) > 0, "No test-year data found (2021/2022)."


def pick_target_index(years: np.ndarray) -> int:
    for ty in TEST_YEARS_PREF:
        idx = np.where(years == ty)[0]
        if len(idx) == 1:
            return int(idx[0])
    idxs = np.where(np.isin(years, TEST_YEARS))[0]
    return int(idxs[-1])


if len(test_segs) >= EXPLAIN_SIZE:
    sel = np.random.choice(len(test_segs), size=EXPLAIN_SIZE, replace=False)
else:
    sel = np.random.choice(len(test_segs), size=EXPLAIN_SIZE, replace=True)
exp_segs = [test_segs[i] for i in sel]


def pack_batch(segs: List[dict]):
    Xs, years, provs, citys, dists = [], [], [], [], []
    for s in segs:
        Xs.append(torch.tensor(s["X"], dtype=torch.float32))
        years.append(torch.tensor(s["years"], dtype=torch.long))
        provs.append(s["prov"])
        citys.append(s["city"])
        dists.append(s["district"])
    X_pad = pad_sequence(Xs, batch_first=True)
    yr_pad = pad_sequence(years, batch_first=True, padding_value=-1)
    pad_mask = yr_pad == -1
    prov_ids = torch.tensor(provs, dtype=torch.long)
    city_ids = torch.tensor(citys, dtype=torch.long)
    dist_ids = torch.tensor(dists, dtype=torch.long)
    return X_pad, yr_pad, pad_mask, prov_ids, city_ids, dist_ids


X_exp, years_exp, pad_exp, prov_exp, city_exp, dist_exp = pack_batch(exp_segs)
B_full, L_full, F_full = X_exp.shape
tidx_exp = torch.tensor([pick_target_index(s["years"]) for s in exp_segs], dtype=torch.long)
causal_mask = build_causal_mask(L_full, torch.device("cpu"))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        return self.dropout(x + self.pe[:, :L, :])


class STCFormerBlock(nn.Module):
    def __init__(self, d_model, nhead, ffn_dim, dropout, prov_dim, city_dim, film_scale_g=0.1, film_scale_b=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)
        self.film_scale_g = film_scale_g
        self.film_scale_b = film_scale_b
        self.film_gen = nn.Sequential(nn.Linear(prov_dim + city_dim, d_model * 2))

    def forward(self, x, causal_mask, pad_mask, static_ctx):
        gb = self.film_gen(static_ctx)
        B, L, D = x.shape
        gamma, beta = gb[:, :D], gb[:, D:]
        gamma = torch.tanh(gamma) * self.film_scale_g
        beta = torch.tanh(beta) * self.film_scale_b
        gamma = gamma.unsqueeze(1).expand(B, L, D)
        beta = beta.unsqueeze(1).expand(B, L, D)
        x = x * (1.0 + gamma) + beta

        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, attn_mask=causal_mask, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop1(attn_out)

        z = self.norm2(x)
        x = x + self.drop2(self.ffn(z))
        return x


class STCFormer(nn.Module):
    def __init__(
        self,
        in_feat,
        n_prov,
        n_city,
        d_model=128,
        nhead=8,
        num_layers=3,
        ffn_dim=256,
        dropout=0.1,
        prov_emb_dim=8,
        city_emb_dim=8,
        film_scale_g=0.1,
        film_scale_b=0.1,
    ):
        super().__init__()
        self.prov_emb = nn.Embedding(n_prov, prov_emb_dim)
        self.city_emb = nn.Embedding(n_city, city_emb_dim)
        nn.init.normal_(self.prov_emb.weight, std=0.02)
        nn.init.normal_(self.city_emb.weight, std=0.02)
        self.in_proj = nn.Linear(in_feat, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                STCFormerBlock(
                    d_model,
                    nhead,
                    ffn_dim,
                    dropout,
                    prov_emb_dim,
                    city_emb_dim,
                    film_scale_g=FILM_SCALE_G,
                    film_scale_b=FILM_SCALE_B,
                )
                for _ in range(NLAYERS)
            ]
        )
        self.norm_final = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, X, prov_ids, city_ids, pad_mask, causal_mask):
        pe = self.prov_emb(prov_ids)
        ce = self.city_emb(city_ids)
        static_ctx = torch.cat([pe, ce], dim=-1)
        z = self.in_proj(X)
        z = self.pos(z)
        for blk in self.blocks:
            z = blk(z, causal_mask, pad_mask, static_ctx)
        z = self.norm_final(z)
        return self.head(z)


assert os.path.exists(BEST_MODEL_PATH), f"Missing model weights: {BEST_MODEL_PATH}"
n_prov = len(prov_categories)
n_city = len(city_categories)
in_feat = len(feat_cols)

model = STCFormer(
    in_feat,
    n_prov,
    n_city,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NLAYERS,
    ffn_dim=FFN_DIM,
    dropout=DROPOUT,
    prov_emb_dim=PROV_EMB_DIM,
    city_emb_dim=CITY_EMB_DIM,
).to("cpu")

state = torch.load(BEST_MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()
print("[OK] Model weights loaded.", flush=True)

X_exp = X_exp.to("cpu")
pad_exp = pad_exp.to("cpu")
prov_exp = prov_exp.to("cpu")
city_exp = city_exp.to("cpu")
tidx_exp = tidx_exp.to("cpu")
causal_mask = causal_mask.to("cpu")

idx_year_sin = feat_cols.index("year_sin")
idx_year_cos = feat_cols.index("year_cos")
orig_idx = [feat_cols.index(c) for c in x_cols]
orig_idx_t = torch.tensor(orig_idx, dtype=torch.long)

X_target_all = torch.gather(X_exp, dim=1, index=tidx_exp.view(-1, 1, 1).expand(-1, 1, F_full)).squeeze(1)


class RowContextPredictor:
    def __init__(self, model, X_full, prov, city, pad, causal_mask, t_indices, col_idx_tensor):
        self.model = model
        self.X_full = X_full
        self.prov = prov
        self.city = city
        self.pad = pad
        self.causal = causal_mask
        self.tidx = t_indices
        self.cols = col_idx_tensor
        self.row = None

    def __call__(self, X_step_np: np.ndarray) -> np.ndarray:
        assert self.row is not None, "predictor.row is not set"
        n_eval = X_step_np.shape[0]
        X_row = self.X_full[self.row : self.row + 1].repeat(n_eval, 1, 1).clone()
        prov_row = self.prov[self.row : self.row + 1].repeat(n_eval)
        city_row = self.city[self.row : self.row + 1].repeat(n_eval)
        pad_row = self.pad[self.row : self.row + 1].repeat(n_eval, 1)
        t = int(self.tidx[self.row].item())

        X_step = torch.from_numpy(X_step_np).to(torch.float32)
        rr = torch.arange(n_eval)
        X_row[rr.unsqueeze(1), torch.tensor([t]).repeat(n_eval).unsqueeze(1), self.cols.unsqueeze(0)] = X_step

        with torch.no_grad():
            yhat = self.model(X_row, prov_row, city_row, pad_row, self.causal)
            out = yhat[:, t, 0]
        return out.detach().cpu().numpy()


X_step_exp_orig = X_target_all.index_select(dim=1, index=orig_idx_t)
X_step_exp_orig_np = X_step_exp_orig.detach().cpu().numpy()
B_used, F_orig = X_step_exp_orig_np.shape
max_evals_orig = 2 * F_orig + 1
print(f"[Beeswarm] Per-row explanation: B={B_used}, F={F_orig}, max_evals={max_evals_orig}", flush=True)

masker_orig = shap.maskers.Independent(X_step_exp_orig_np)
predictor_orig = RowContextPredictor(model, X_exp, prov_exp, city_exp, pad_exp, causal_mask, tidx_exp, orig_idx_t)
explainer_orig = shap.Explainer(predictor_orig, masker_orig, algorithm="permutation")

shap_vals_list, base_vals_list = [], []
for i in iter_progress(range(B_used), desc="Beeswarm rows", total=B_used):
    predictor_orig.row = i
    ev = explainer_orig(X_step_exp_orig_np[i : i + 1, :], max_evals=max_evals_orig)
    shap_vals_list.append(ev.values[0])
    base_vals_list.append(ev.base_values[0] if np.ndim(ev.base_values) > 0 else ev.base_values)
shap_vals_orig = np.vstack(shap_vals_list)

feat_raw_for_bee = np.zeros_like(shap_vals_orig, dtype=np.float32)
df_lookup = df_raw[[district_col_saved, year_col] + x_cols].copy()

for b, s in enumerate(exp_segs):
    t = int(pick_target_index(s["years"]))
    dist_code = s["district"]
    dist_str = district_categories[dist_code]
    row = df_lookup[
        (df_lookup[district_col_saved].astype(str) == str(dist_str)) & (df_lookup[year_col].astype(int) == int(s["years"][t]))
    ]
    if len(row) == 0:
        feat_raw_for_bee[b, :] = 0.0
    else:
        feat_raw_for_bee[b, :] = row.iloc[0][x_cols].astype(np.float32).values

order_idx = np.argsort(np.nanmean(np.abs(shap_vals_orig), axis=0))[::-1]

shap_vals_trim, feat_raw_trim = trim_shap_tail(
    shap_vals_orig, feat_raw_for_bee, lower_q=BEESWARM_TRIM_LOWER_Q, upper_q=BEESWARM_TRIM_UPPER_Q
)

exp_bee = shap.Explanation(
    values=shap_vals_trim,
    base_values=np.array(base_vals_list),
    data=feat_raw_trim,
    feature_names=x_cols,
)

os.makedirs(WEIGHT_DIR, exist_ok=True)
plt.figure(figsize=(10, max(6, len(x_cols) * 0.5)))
shap.plots.beeswarm(exp_bee, show=False, max_display=min(30, len(x_cols)), order=order_idx)
plt.tight_layout()
plt.savefig(os.path.join(WEIGHT_DIR, "shap_beeswarm_original.png"), dpi=300)
plt.close()
print("[OK] Saved: weight/shap_beeswarm_original.png", flush=True)

rank_map = {int(j): int(r) for r, j in enumerate(order_idx)}
bee_rows = []
for b, s in enumerate(exp_segs):
    t_idx = int(pick_target_index(s["years"]))
    t_year = int(s["years"][t_idx])
    dist_idx = int(s["district"])
    dist_str = district_categories[dist_idx]
    base_v = float(base_vals_list[b])
    for j, feat in enumerate(x_cols):
        bee_rows.append(
            {
                "sample_id": b,
                "district_cat_idx": dist_idx,
                "district_code": dist_str,
                "target_year": t_year,
                "feature": feat,
                "feature_rank_by_mean_abs_shap": int(rank_map[j]) + 1,
                "shap_value": float(shap_vals_orig[b, j]),
                "shap_value_trim": (None if np.isnan(shap_vals_trim[b, j]) else float(shap_vals_trim[b, j])),
                "feature_value_raw": float(feat_raw_for_bee[b, j]),
                "feature_value_raw_trim": (None if np.isnan(feat_raw_trim[b, j]) else float(feat_raw_trim[b, j])),
                "base_value": base_v,
            }
        )

pd.DataFrame(bee_rows).to_csv(os.path.join(WEIGHT_DIR, "shap_beeswarm_original_data.csv"), index=False, encoding="utf-8")
print("[OK] Saved: weight/shap_beeswarm_original_data.csv", flush=True)

years_exp_np = years_exp.detach().cpu().numpy()
years_all = sorted({int(y) for row in years_exp_np for y in row if y >= 0})
heat_year = pd.DataFrame(0.0, index=years_all, columns=x_cols)

for yr in iter_progress(years_all, desc="Years for heatmap", total=len(years_all)):
    rows = []
    for b in range(B_full):
        idxs = np.where(years_exp_np[b] == yr)[0]
        if len(idxs):
            rows.append(b)
    if not rows:
        continue
    if len(rows) > HEAT_SAMPLES_PER_YEAR:
        rows = list(np.random.choice(rows, HEAT_SAMPLES_PER_YEAR, replace=False))
    vals = np.abs(shap_vals_orig[np.array(rows)])
    heat_year.loc[yr, :] = np.mean(vals, axis=0)

plt.figure(figsize=(1.0 * len(x_cols) + 4, 0.45 * len(years_all) + 3))
sns.heatmap(heat_year, cmap="viridis")
plt.title("Mean |SHAP| by Year × Feature")
plt.xlabel("Feature")
plt.ylabel("Year")
plt.tight_layout()
plt.savefig(os.path.join(WEIGHT_DIR, "shap_heatmap_year_feature.png"), dpi=300)
plt.close()
pd.DataFrame(heat_year).to_csv(os.path.join(WEIGHT_DIR, "shap_heatmap_year_feature.csv"), index=True, encoding="utf-8")
print("[OK] Saved: weight/shap_heatmap_year_feature.png and .csv", flush=True)

prov_ids = sorted(set(int(p) for p in prov_exp.numpy().tolist()))
heat_prov = pd.DataFrame(0.0, index=[str(prov_categories[p]) for p in prov_ids], columns=x_cols)
prov_np = prov_exp.numpy()

for p in iter_progress(prov_ids, desc="Provinces for heatmap", total=len(prov_ids)):
    rows = np.where(prov_np == p)[0].tolist()
    if not rows:
        continue
    if len(rows) > SPATIAL_SAMPLES_PER_PROV:
        rows = list(np.random.choice(rows, SPATIAL_SAMPLES_PER_PROV, replace=False))
    vals = np.abs(shap_vals_orig[np.array(rows)])
    heat_prov.loc[str(prov_categories[p]), :] = np.mean(vals, axis=0)

plt.figure(figsize=(1.0 * len(x_cols) + 4, 0.45 * len(prov_ids) + 3))
sns.heatmap(heat_prov, cmap="viridis")
plt.title("Mean |SHAP| by Province × Feature")
plt.xlabel("Feature")
plt.ylabel("Province")
plt.tight_layout()
plt.savefig(os.path.join(WEIGHT_DIR, "shap_heatmap_province_feature.png"), dpi=300)
plt.close()
pd.DataFrame(heat_prov).to_csv(os.path.join(WEIGHT_DIR, "shap_heatmap_province_feature.csv"), index=True, encoding="utf-8")
print("[OK] Saved: weight/shap_heatmap_province_feature.png and .csv", flush=True)

imp_orig_features = np.mean(np.abs(shap_vals_orig), axis=0)

device_region = torch.device("cuda:0") if (USE_GPU_FOR_REGION and torch.cuda.is_available()) else torch.device("cpu")
print(f"[Region(ctx)] device = {device_region}", flush=True)

model_region = model.to(device_region)
X_exp_dev = X_exp.to(device_region)
pad_exp_dev = pad_exp.to(device_region)
prov_exp_dev = prov_exp.to(device_region)
city_exp_dev = city_exp.to(device_region)
tidx_exp_dev = tidx_exp.to(device_region)
causal_mask_dev = causal_mask.to(device_region)

with torch.no_grad():
    base_full = model_region(X_exp_dev, prov_exp_dev, city_exp_dev, pad_exp_dev, causal_mask_dev)
    base_pred = base_full[torch.arange(B_full, device=device_region), tidx_exp_dev, 0]

deltas_mean_region = []
for b in iter_progress(range(B_full), desc="Region(ctx) rows", total=B_full):
    idx = torch.randint(0, B_full, (REGION_SWAP_K,), device=device_region)
    prov_batch = prov_exp_dev.index_select(0, idx).clone()
    city_batch = city_exp_dev.index_select(0, idx).clone()

    X_b = X_exp_dev[b : b + 1]
    pad_b = pad_exp_dev[b : b + 1]
    t = int(tidx_exp_dev[b].item())
    X_batch = X_b.repeat(REGION_SWAP_K, 1, 1)[:, : t + 1, :]
    pad_batch = pad_b.repeat(REGION_SWAP_K, 1)[:, : t + 1]
    causal_t = causal_mask_dev[: t + 1, : t + 1]

    with torch.no_grad():
        pred_alt = model_region(X_batch, prov_batch, city_batch, pad_batch, causal_t)[:, t, 0]
        delta = torch.abs(pred_alt - base_pred[b])
    deltas_mean_region.append(float(delta.mean().item()))

imp_region_raw = float(np.mean(deltas_mean_region))

device_time = torch.device("cuda:0") if (USE_GPU_FOR_TIME and torch.cuda.is_available()) else torch.device("cpu")
print(f"[Time(shuffle)] device = {device_time}", flush=True)

if device_time != device_region:
    model_time = model.to(device_time)
    X_exp_time = X_exp.to(device_time)
    pad_exp_time = pad_exp.to(device_time)
    prov_exp_time = prov_exp.to(device_time)
    city_exp_time = city_exp.to(device_time)
    tidx_exp_time = tidx_exp.to(device_time)
    causal_mask_time = causal_mask.to(device_time)
    with torch.no_grad():
        base_full_time = model_time(X_exp_time, prov_exp_time, city_exp_time, pad_exp_time, causal_mask_time)
        base_pred_time = base_full_time[torch.arange(B_full, device=device_time), tidx_exp_time, 0]
else:
    model_time = model_region
    X_exp_time, pad_exp_time = X_exp_dev, pad_exp_dev
    prov_exp_time, city_exp_time = prov_exp_dev, city_exp_dev
    tidx_exp_time, causal_mask_time = tidx_exp_dev, causal_mask_dev
    base_pred_time = base_pred

time_idx_list = [idx_year_sin, idx_year_cos]
deltas_mean_time = []
for b in iter_progress(range(B_full), desc="Time(shuffle) rows", total=B_full):
    X_b = X_exp_time[b : b + 1]
    pad_b = pad_exp_time[b : b + 1]
    t = int(tidx_exp_time[b].item())
    Lb = t + 1

    X_batch = X_b.repeat(TIME_SHUFFLE_K, 1, 1)
    X_pref = X_b[0, :Lb, :]

    X_time_pref = X_pref[:, time_idx_list]
    X_time_pref_exp = X_time_pref.unsqueeze(0).repeat(TIME_SHUFFLE_K, 1, 1)

    perms = torch.stack([torch.randperm(Lb, device=device_time) for _ in range(TIME_SHUFFLE_K)], dim=0)
    perms_2d = perms.unsqueeze(-1).expand(TIME_SHUFFLE_K, Lb, 2)
    X_time_shuffled = torch.gather(X_time_pref_exp, 1, perms_2d)

    X_batch[:, :Lb, time_idx_list[0]] = X_time_shuffled[:, :, 0]
    X_batch[:, :Lb, time_idx_list[1]] = X_time_shuffled[:, :, 1]

    pad_batch = pad_b.repeat(TIME_SHUFFLE_K, 1)[:, :Lb]
    causal_t = causal_mask_time[:Lb, :Lb]

    with torch.no_grad():
        pred_alt = model_time(
            X_batch[:, :Lb, :],
            prov_exp_time[b : b + 1].repeat(TIME_SHUFFLE_K),
            city_exp_time[b : b + 1].repeat(TIME_SHUFFLE_K),
            pad_batch,
            causal_t,
        )[:, t, 0]
        delta = torch.abs(pred_alt - base_pred_time[b])

    deltas_mean_time.append(float(delta.mean().item()))

imp_time_raw = float(np.mean(deltas_mean_time))

names = x_cols + ["Year(shuffle-prefix)", "Region(context-swap)"]
values_original = list(imp_orig_features) + [imp_time_raw, imp_region_raw]
values_scaled = list(imp_orig_features) + [imp_time_raw * SCALE_FACTOR_TIME_REGION, imp_region_raw * SCALE_FACTOR_TIME_REGION]

imp_df_original = pd.DataFrame({"feature": names, "mean_abs_contrib": values_original}).sort_values(
    "mean_abs_contrib", ascending=False
)
imp_df_scaled = pd.DataFrame({"feature": names, "mean_abs_contrib": values_scaled}).sort_values(
    "mean_abs_contrib", ascending=False
)

imp_df_original.to_csv(os.path.join(WEIGHT_DIR, "global_importance_original.csv"), index=False, encoding="utf-8")
imp_df_scaled.to_csv(os.path.join(WEIGHT_DIR, "global_importance_scaled.csv"), index=False, encoding="utf-8")

plt.figure(figsize=(10, max(5, len(names) * 0.5)))
sns.barplot(data=imp_df_scaled, x="mean_abs_contrib", y="feature", orient="h")
plt.title(
    "Global Importance (Permutation SHAP @ Target step)\n"
    "Original features + Time(shuffle) + Region(swap)\n"
    f"[Time/Region shown ×{SCALE_FACTOR_TIME_REGION}]"
)
plt.xlabel("Mean |Contribution| (standardized output units)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(WEIGHT_DIR, "global_importance_bar.png"), dpi=300)
plt.close()

print("[OK] Outputs saved:", flush=True)
print(" - weight/shap_beeswarm_original.png and weight/shap_beeswarm_original_data.csv", flush=True)
print(" - weight/shap_heatmap_year_feature.png and weight/shap_heatmap_year_feature.csv", flush=True)
print(" - weight/shap_heatmap_province_feature.png and weight/shap_heatmap_province_feature.csv", flush=True)
print(" - weight/global_importance_original.csv", flush=True)
print(f" - weight/global_importance_scaled.csv (Time/Region ×{SCALE_FACTOR_TIME_REGION})", flush=True)
print(" - weight/global_importance_bar.png", flush=True)
