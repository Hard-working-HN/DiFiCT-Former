# -*- coding: utf-8 -*-
"""
Spatio-Temporal Causal Transformer (STCFormer, 区县级序列版)
- 无滑动窗口：每个 district(Code) 为一条整段序列（而非 (prov, city)）
- 因果注意力（causal mask）+ 变长 padding mask
- 年份划分：Train=2013-2019, Val=2020, Test=2021-2022
- 空间上下文：省/市ID小维度嵌入 + 低幅度 FiLM 条件化（上下文，不作为分组键）
依赖：torch>=1.12, numpy, pandas, scikit-learn
"""

import os, math, pickle, warnings, csv, json
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

# ====================== 配置 ======================
FILE_PATH = "Industry.csv"

# 列位置（与原始脚本一致）：年份/省/市索引；目标列索引；X 起始索引
YEAR_COL_IDX, PROV_COL_IDX, CITY_COL_IDX = 3, 4, 5
X_START_IDX, Y_COL_IDX = 6, -1

# 区县唯一编号列名（必须存在）
DISTRICT_COL_NAME = "Code"

# 年份划分
TRAIN_YEARS = list(range(2013, 2020))
VAL_YEARS   = [2020]
TEST_YEARS  = [2021, 2022]

# 训练配置
EPOCHS = 999
BATCH_SIZE = 1024                 # 批里是“段”的数量（每段=一个区县的全时序）
LR = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.01
SEED = 3704
PRINT_EVERY = 10  # 仅控制打印频率，不影响“每个epoch验证+保存”
WEIGHT_DIR = "weight"
BEST_MODEL_PATH = os.path.join(WEIGHT_DIR, "stcformer_best.pt")
METRICS_CSV = os.path.join(WEIGHT_DIR, "stcformer_metrics.csv")

# STCFormer 主体超参
D_MODEL = 256
NHEAD = 8
NLAYERS = 8
FFN_DIM = 256

# 空间嵌入（弱编码，用作静态上下文）
PROV_EMB_DIM = 8
CITY_EMB_DIM = 8
FILM_SCALE_G = 0.80   # γ 缩放（越小越弱）
FILM_SCALE_B = 0.80   # β 缩放（越小越弱）

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)

os.makedirs(WEIGHT_DIR, exist_ok=True)

# ====================== 工具函数 ======================
def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def compute_time_encoding(year_series: pd.Series) -> pd.DataFrame:
    ymin, ymax = year_series.min(), year_series.max()
    denom = (ymax - ymin) if ymax != ymin else 1.0
    yr_norm = (year_series - ymin) / denom
    return pd.DataFrame({
        "year_sin": np.sin(2*math.pi*yr_norm),
        "year_cos": np.cos(2*math.pi*yr_norm)
    }, index=year_series.index)

def build_causal_mask(L: int, device=None) -> torch.Tensor:
    """Transformer Encoder 的因果 mask（True=遮蔽不可见）"""
    return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

# ====================== 读数据与预处理 ======================
df_raw = pd.read_csv(FILE_PATH)
assert DISTRICT_COL_NAME in df_raw.columns, f"原始数据缺少区县唯一编号列：{DISTRICT_COL_NAME}"

cols = df_raw.columns.tolist()
year_col, prov_col, city_col = cols[YEAR_COL_IDX], cols[PROV_COL_IDX], cols[CITY_COL_IDX]
y_col = cols[Y_COL_IDX]
x_cols = cols[X_START_IDX:Y_COL_IDX]   # 原始 X（不含 year_sin/cos；不含 Code/prov/city）

# 选取所需列
df = df_raw[[DISTRICT_COL_NAME, year_col, prov_col, city_col] + x_cols + [y_col]].copy()
df[x_cols + [y_col]] = safe_numeric(df[x_cols + [y_col]])
df = df.dropna(subset=[DISTRICT_COL_NAME, year_col, prov_col, city_col, y_col]).reset_index(drop=True)

# 缺失插补（特征）
for c in x_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# 类别编码（固定类别顺序）
prov_cat = pd.Categorical(df[prov_col])
city_cat = pd.Categorical(df[city_col])
dist_cat = pd.Categorical(df[DISTRICT_COL_NAME])  # 关键：区县唯一编号编码
df["prov_code"]    = prov_cat.codes.astype(int)
df["city_code"]    = city_cat.codes.astype(int)
df["district_code"]= dist_cat.codes.astype(int)

n_prov, n_city = len(prov_cat.categories), len(city_cat.categories)
n_dist = len(dist_cat.categories)

# 时间编码
te = compute_time_encoding(df[year_col])
df["year_sin"], df["year_cos"] = te["year_sin"], te["year_cos"]
feat_cols = x_cols + ["year_sin", "year_cos"]  # F = 原始X + 2

# —— 构造整段序列（每个 district_code 一“段”） —— #
# —— 构造整段序列（每个 district_code 一“段”） —— #
segments = []
# 关键：用 df["district_code"] 这个 Series 来分组，确保分组键是标量而不是 tuple
for d, g in df.groupby(df["district_code"], sort=False):
    # 有些 pandas 版本即便只分 1 列也可能返回 tuple，保险起见做一次解包
    if isinstance(d, tuple):
        d = d[0]

    # 按年份排序
    g = g.sort_values(year_col)

    # 省/市在同一个区县内应唯一；若出现多个，记录告警并取众数
    prov_vals = g["prov_code"].unique()
    city_vals = g["city_code"].unique()
    if len(prov_vals) > 1 or len(city_vals) > 1:
        print(f"[WARN] district_code={d} 在不同年份出现多个省/市编码：prov={prov_vals}, city={city_vals}；将使用众数。")

    prov_mode = int(g["prov_code"].mode(dropna=False).iloc[0])
    city_mode = int(g["city_code"].mode(dropna=False).iloc[0])

    years = g[year_col].values.astype(int)
    X = g[feat_cols].values.astype(np.float32)            # (T, F)
    y = g[y_col].values.astype(np.float32).reshape(-1,1)  # (T, 1)

    segments.append(dict(
        district=int(d),           # 现在 d 一定是标量，转换安全
        prov=prov_mode,
        city=city_mode,
        years=years,
        X=X,
        y=y
    ))

# 标准化：仅在训练年份 fit
scaler_X, scaler_y = StandardScaler(), StandardScaler()
Xtr, ytr = [], []
for s in segments:
    m = np.isin(s["years"], TRAIN_YEARS)
    if m.any():
        Xtr.append(s["X"][m]); ytr.append(s["y"][m])
Xtr = np.concatenate(Xtr, axis=0); ytr = np.concatenate(ytr, axis=0)
scaler_X.fit(Xtr); scaler_y.fit(ytr)

for s in segments:
    s["X"] = scaler_X.transform(s["X"])
    s["y"] = scaler_y.transform(s["y"])
    yrs = s["years"]
    s["m_tr"] = np.isin(yrs, TRAIN_YEARS).astype(np.uint8)
    s["m_va"] = np.isin(yrs, VAL_YEARS).astype(np.uint8)
    s["m_te"] = np.isin(yrs, TEST_YEARS).astype(np.uint8)

def pick_split(segs, key):
    return [s for s in segs if s[key].any()]

train_segs = pick_split(segments, "m_tr")
val_segs   = pick_split(segments, "m_va")
test_segs  = pick_split(segments, "m_te")

# ====================== Dataset / Collate ======================
class FullSeqDataset(Dataset):
    def __init__(self, segs): self.segs = segs
    def __len__(self): return len(self.segs)
    def __getitem__(self, idx):
        s = self.segs[idx]
        return (torch.tensor(s["X"], dtype=torch.float32),    # (T,F)
                torch.tensor(s["y"], dtype=torch.float32),    # (T,1)
                torch.tensor(s["prov"], dtype=torch.long),
                torch.tensor(s["city"], dtype=torch.long),
                torch.tensor(s["district"], dtype=torch.long),
                torch.tensor(s["m_tr"], dtype=torch.uint8),   # (T,)
                torch.tensor(s["m_va"], dtype=torch.uint8),
                torch.tensor(s["m_te"], dtype=torch.uint8))

def collate_fn(batch):
    Xs, ys, ps, cs, ds, mtr, mva, mte = zip(*batch)
    lens = torch.tensor([x.size(0) for x in Xs], dtype=torch.long)
    X_pad  = pad_sequence(Xs,  batch_first=True)   # (B, Lmax, F)
    y_pad  = pad_sequence(ys,  batch_first=True)   # (B, Lmax, 1)
    mtr_pd = pad_sequence(mtr, batch_first=True)   # (B, Lmax)
    mva_pd = pad_sequence(mva, batch_first=True)
    mte_pd = pad_sequence(mte, batch_first=True)
    p = torch.stack(ps); c = torch.stack(cs); d = torch.stack(ds)
    # True 表示 padding
    pad_mask = torch.arange(X_pad.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
    return X_pad, y_pad, p, c, d, mtr_pd, mva_pd, mte_pd, pad_mask, lens

train_loader = DataLoader(FullSeqDataset(train_segs), batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(FullSeqDataset(val_segs),   batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(FullSeqDataset(test_segs),  batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn)

# ====================== 模型：STCFormer（简化实现） ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)
    def forward(self, x):  # x: (B,L,D)
        L = x.size(1)
        return self.dropout(x + self.pe[:, :L, :])

class STCFormerBlock(nn.Module):
    """
    Spatio-Temporal Causal Transformer Block（弱空间条件化）
    - Temporal: Multi-head causal self-attention
    - Spatial: 由 (prov, city) 嵌入生成 FiLM 参数，对通道维进行弱调制
    """
    def __init__(self, d_model, nhead, ffn_dim, dropout,
                 prov_dim, city_dim, film_scale_g=0.1, film_scale_b=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                           dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model)
        )
        self.drop2 = nn.Dropout(dropout)

        # —— 弱空间条件化：FiLM(γ, β) 来自省/市嵌入（小幅度）
        self.film_scale_g = film_scale_g
        self.film_scale_b = film_scale_b
        self.film_gen = nn.Sequential(
            nn.Linear(prov_dim + city_dim, d_model*2),
        )

    def forward(self, x, causal_mask, pad_mask, static_ctx):
        """
        x: (B,L,D)
        causal_mask: (L,L) True=遮蔽未来
        pad_mask: (B,L) True=padding
        static_ctx: (B, Pdim+Cdim)
        """
        # 1) 弱 FiLM 条件化（对每个时间步相同的 γ/β）
        gb = self.film_gen(static_ctx)                    # (B, 2D)
        B, L, D = x.shape
        gamma, beta = gb[:, :D], gb[:, D:]                # (B,D), (B,D)
        gamma = torch.tanh(gamma) * self.film_scale_g
        beta  = torch.tanh(beta)  * self.film_scale_b
        gamma = gamma.unsqueeze(1).expand(B, L, D)
        beta  = beta .unsqueeze(1).expand(B, L, D)
        x = x * (1.0 + gamma) + beta

        # 2) 因果自注意力
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, attn_mask=causal_mask, key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop1(attn_out)

        # 3) FFN
        z = self.norm2(x)
        x = x + self.drop2(self.ffn(z))
        return x

class STCFormer(nn.Module):
    def __init__(self, in_feat, n_prov, n_city,
                 d_model=128, nhead=8, num_layers=3, ffn_dim=256, dropout=0.1,
                 prov_emb_dim=8, city_emb_dim=8, film_scale_g=0.1, film_scale_b=0.1):
        super().__init__()
        # 空间嵌入（弱）
        self.prov_emb = nn.Embedding(n_prov, prov_emb_dim)
        self.city_emb = nn.Embedding(n_city, city_emb_dim)
        nn.init.normal_(self.prov_emb.weight, std=0.02)
        nn.init.normal_(self.city_emb.weight, std=0.02)

        # 输入升维 + 位置编码
        self.in_proj = nn.Linear(in_feat, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        # 堆叠若干 STCFormerBlock
        self.blocks = nn.ModuleList([
            STCFormerBlock(d_model, nhead, ffn_dim, dropout,
                           prov_emb_dim, city_emb_dim,
                           film_scale_g=film_scale_g,
                           film_scale_b=film_scale_b)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)

        # 回归头（逐时间步）
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, X, prov_ids, city_ids, pad_mask, causal_mask):
        """
        X: (B,L,F)
        pad_mask: (B,L) True=padding
        causal_mask: (L,L) True=遮蔽未来
        """
        pe = self.prov_emb(prov_ids)   # (B, Pdim)
        ce = self.city_emb(city_ids)   # (B, Cdim)
        static_ctx = torch.cat([pe, ce], dim=-1)  # (B, Pdim+Cdim)

        z = self.in_proj(X)            # (B,L,D)
        z = self.pos(z)
        for blk in self.blocks:
            z = blk(z, causal_mask, pad_mask, static_ctx)

        z = self.norm_final(z)
        yhat = self.head(z)            # (B,L,1)
        return yhat

# ====================== 训练 / 评估（基础） ======================
in_feat = len(feat_cols)
model = STCFormer(in_feat, n_prov, n_city,
                  d_model=D_MODEL, nhead=NHEAD, num_layers=NLAYERS,
                  ffn_dim=FFN_DIM, dropout=DROPOUT,
                  prov_emb_dim=PROV_EMB_DIM, city_emb_dim=CITY_EMB_DIM,
                  film_scale_g=FILM_SCALE_G, film_scale_b=FILM_SCALE_B).to(DEVICE)

criterion = nn.MSELoss(reduction="none")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def adjust_lr(optimizer, factor: float):
    """按给定因子缩放当前学习率，例如 factor=0.8 表示降低 20%"""
    for pg in optimizer.param_groups:
        pg["lr"] *= factor

def one_epoch(loader, which="train"):
    is_train = which=="train"
    model.train() if is_train else model.eval()
    tot_loss, tot_steps = 0.0, 0
    ys_true, ys_pred = [], []

    for Xb, yb, pb, cb, db, mtr, mva, mte, pad_mask, lens in loader:
        Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
        pb = pb.to(DEVICE); cb = cb.to(DEVICE)
        pad_mask = pad_mask.to(DEVICE)     # True=padding
        Lmax = Xb.size(1)
        causal = build_causal_mask(Lmax, DEVICE)

        # 选择该阶段的年份 mask
        m = {"train": mtr, "val": mva, "test": mte}[which].to(DEVICE).float()  # (B,L)
        valid = (~pad_mask).float() * m

        with torch.set_grad_enabled(is_train):
            yhat = model(Xb, pb, cb, pad_mask, causal)  # (B,L,1)
            mse = criterion(yhat, yb).squeeze(-1)       # (B,L)
            loss = (mse * valid).sum() / (valid.sum().clamp(min=1.0))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        tot_loss += (mse * valid).sum().item()
        tot_steps += int(valid.sum().item())
        sel = valid.bool()
        if sel.any():
            ys_true.append(yb[sel].detach().cpu().numpy())
            ys_pred.append(yhat[sel].detach().cpu().numpy())

    if tot_steps == 0 or len(ys_true) == 0:
        return float("inf"), 0., float("inf"), float("inf")

    y_true = np.concatenate(ys_true, axis=0).reshape(-1,1)
    y_pred = np.concatenate(ys_pred, axis=0).reshape(-1,1)
    # 反标准化
    y_true = scaler_y.inverse_transform(y_true)
    y_pred = scaler_y.inverse_transform(y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred)/(np.abs(y_true) + 1e-8))) * 100)
    return tot_loss/max(tot_steps,1), r2, rmse, mape

# 初始化度量日志
if not os.path.exists(METRICS_CSV):
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch",
                    "train_loss","train_r2","train_rmse","train_mape",
                    "val_loss","val_r2","val_rmse","val_mape"])

# —— 训练并保存“验证最优”（每个 epoch 都评估+判优+保存）
best_metric, best_state, best_epoch = float("inf"), None, -1
for ep in range(1, EPOCHS+1):

    # ===== 学习率调度：500 次训练之后，每 50 次训练学习率降低 20% =====
    # 第一次衰减：epoch=550，此后 600, 650, ...
    if ep > 500 and (ep - 500) % 50 == 0:
        adjust_lr(optimizer, 0.8)
        print(f"[LR] Epoch {ep}: lr -> {optimizer.param_groups[0]['lr']:.6e}")

    tr_loss, tr_r2, tr_rmse, tr_mape = one_epoch(train_loader, "train")
    va_loss, va_r2, va_rmse, va_mape = one_epoch(val_loader, "val")

    # 写入度量日志
    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([ep,
                    f"{tr_loss:.6f}", f"{tr_r2:.6f}", f"{tr_rmse:.6f}", f"{tr_mape:.4f}",
                    f"{va_loss:.6f}", f"{va_r2:.6f}", f"{va_rmse:.6f}", f"{va_mape:.4f}"])

    # 控制台打印
    if ep % PRINT_EVERY == 0 or ep == 1:
        print(f"Epoch {ep:3d} | Train MAPE={tr_mape:.2f}% | Val R2={va_r2:.4f} RMSE={va_rmse:.2f} MAPE={va_mape:.2f}%")

    # —— 选优与保存（每个 epoch 都判断）
    score = va_loss  # 也可换成 va_mape（越小越好）或 -va_r2（越小越好）
    if score < best_metric:
        best_metric, best_epoch = score, ep
        best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save(best_state, BEST_MODEL_PATH)
        print(f"[BEST] epoch={ep}  val_loss={va_loss:.6f} -> saved to {BEST_MODEL_PATH}")

# —— 加载最优并在 Test 上评估（基础指标） —— #
assert os.path.exists(BEST_MODEL_PATH), "未找到最优模型文件！"
state = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)

tr = one_epoch(train_loader, "train")
va = one_epoch(val_loader,   "val")
te = one_epoch(test_loader,  "test")

print("\n===== 最终结果（基于验证最优模型） =====")
print(f"[Best @ Epoch {best_epoch}]")
print(f"Train: R2={tr[1]:.4f}, RMSE={tr[2]:.4f}, MAPE={tr[3]:.2f}%")
print(f"Valid: R2={va[1]:.4f}, RMSE={va[2]:.4f}, MAPE={va[3]:.2f}%")
print(f"Test : R2={te[1]:.4f}, RMSE={te[2]:.4f}, MAPE={te[3]:.2f}%")

# ====================== 额外评估（测试集更丰富指标 & 落盘） ======================
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    eps = 1e-8
    err = y_pred - y_true
    ae  = np.abs(err)
    pe  = ae / (np.abs(y_true) + eps)

    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(ae))
    medae= float(np.median(ae))
    mape = float(np.mean(pe) * 100.0)
    smape= float(np.mean(2.0*ae/(np.abs(y_true)+np.abs(y_pred)+eps)) * 100.0)
    wmape= float(ae.sum() / (np.abs(y_true).sum() + eps) * 100.0)
    nrmse_range = float(rmse / (y_true.max() - y_true.min() + eps))
    bias = float(np.mean(err))
    # R2
    r2 = float(1.0 - np.sum(err**2) / (np.sum((y_true - y_true.mean())**2) + eps))
    # Pearson r
    std_t, std_p = np.std(y_true), np.std(y_pred)
    if std_t < eps or std_p < eps:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0,1])
    # Relative Absolute Error 相对均值基线
    rae = float(np.sum(ae) / (np.sum(np.abs(y_true - y_true.mean())) + eps))

    return {
        "R2": r2, "RMSE": rmse, "MAE": mae, "MedianAE": medae,
        "MAPE_%": mape, "sMAPE_%": smape, "WMAPE_%": wmape,
        "NRMSE_range": nrmse_range, "Pearson_r": pearson_r,
        "Bias": bias, "RAE": rae
    }

def _pack_batch_for_eval(segs_batch, device):
    """把一批 test 段 pad 成同长度，返回张量与对齐的年份/掩膜。"""
    Xs, ys, years, provs, citys, dists, mtes = [], [], [], [], [], [], []
    for s in segs_batch:
        Xs.append(torch.tensor(s["X"], dtype=torch.float32))
        ys.append(torch.tensor(s["y"], dtype=torch.float32))
        years.append(torch.tensor(s["years"], dtype=torch.long))
        provs.append(s["prov"]); citys.append(s["city"]); dists.append(s["district"])
        mtes.append(torch.tensor(s["m_te"], dtype=torch.uint8))
    X_pad  = pad_sequence(Xs, batch_first=True)   # (B, L, F)
    y_pad  = pad_sequence(ys, batch_first=True)   # (B, L, 1)
    yr_pad = pad_sequence(years, batch_first=True, padding_value=-1)  # (B, L)
    te_pad = pad_sequence(mtes, batch_first=True, padding_value=0)    # (B, L)
    pad_mask = (yr_pad == -1)                                         # True=padding
    prov_ids = torch.tensor(provs, dtype=torch.long)
    city_ids = torch.tensor(citys, dtype=torch.long)
    dist_ids = torch.tensor(dists, dtype=torch.long)
    # to device
    return (X_pad.to(device), y_pad.to(device), yr_pad, te_pad.to(device),
            prov_ids.to(device), city_ids.to(device), dist_ids, pad_mask.to(device))

def evaluate_and_save_on_test(model, test_segs, scaler_y, out_pred_csv, out_year_csv, out_summary_json,
                              device=DEVICE, batch_size=128):
    model.eval()
    rows = []  # 收集逐条结果

    with torch.no_grad():
        for i in range(0, len(test_segs), batch_size):
            batch = test_segs[i:i+batch_size]
            Xb, yb, years_b_cpu, mte_b, prov_b, city_b, dist_b, pad_mask_b = _pack_batch_for_eval(batch, device)
            B, L, _ = Xb.shape
            causal = build_causal_mask(L, device)

            yhat = model(Xb, prov_b, city_b, pad_mask_b, causal)  # (B, L, 1)

            # 选择有效的“测试年份”位置
            valid = (~pad_mask_b) & (mte_b.bool())  # (B, L)
            if valid.any():
                idx_b, idx_t = torch.where(valid)
                y_true_std = yb[idx_b, idx_t, 0].cpu().numpy().reshape(-1,1)
                y_pred_std = yhat[idx_b, idx_t, 0].cpu().numpy().reshape(-1,1)

                # 反标准化到原尺度
                y_true = scaler_y.inverse_transform(y_true_std).reshape(-1)
                y_pred = scaler_y.inverse_transform(y_pred_std).reshape(-1)

                years_np = years_b_cpu.numpy()
                prov_np  = prov_b.detach().cpu().numpy()
                city_np  = city_b.detach().cpu().numpy()
                dist_np  = dist_b.detach().cpu().numpy()

                for k in range(len(idx_b)):
                    b = int(idx_b[k]); t = int(idx_t[k])
                    yr = int(years_np[b, t])
                    rows.append({
                        "district_code": int(dist_np[b]),
                        "prov_code": int(prov_np[b]),
                        "city_code": int(city_np[b]),
                        "year": yr,
                        "y_true": float(y_true[k]),
                        "y_pred": float(y_pred[k]),
                        "residual": float(y_pred[k] - y_true[k]),
                        "APE_%": float(abs(y_pred[k] - y_true[k]) / (abs(y_true[k]) + 1e-8) * 100.0)
                    })

    assert len(rows) > 0, "测试集没有可评估的点（请检查 TEST_YEARS 掩膜）"
    df = pd.DataFrame(rows)
    df.sort_values(["year", "district_code"], inplace=True)
    os.makedirs(os.path.dirname(out_pred_csv), exist_ok=True)
    df.to_csv(out_pred_csv, index=False, encoding="utf-8")

    # 总体指标
    metrics_all = _compute_metrics(df["y_true"].values, df["y_pred"].values)

    # 按年份聚合指标
    year_rows = []
    for y, g in df.groupby("year"):
        m = _compute_metrics(g["y_true"].values, g["y_pred"].values)
        m["year"] = int(y)
        year_rows.append(m)
    df_year = pd.DataFrame(year_rows).sort_values("year")
    df_year.to_csv(out_year_csv, index=False, encoding="utf-8")

    # 落盘 JSON
    with open(out_summary_json, "w", encoding="utf-8") as f:
        json.dump({
            "overall": metrics_all,
            "by_year_csv": os.path.basename(out_year_csv),
            "predictions_csv": os.path.basename(out_pred_csv)
        }, f, ensure_ascii=False, indent=2)

    # 控制台打印一个简表
    print("\n===== Test split: extra metrics =====")
    for k, v in metrics_all.items():
        if isinstance(v, float):
            print(f"{k:>12s}: {v:.6f}")
        else:
            print(f"{k:>12s}: {v}")
    print(f"\n已保存：\n - {out_pred_csv}\n - {out_year_csv}\n - {out_summary_json}")

# —— 调用额外评估
evaluate_and_save_on_test(
    model=model,
    test_segs=test_segs,
    scaler_y=scaler_y,
    out_pred_csv=os.path.join(WEIGHT_DIR, "test_predictions.csv"),
    out_year_csv=os.path.join(WEIGHT_DIR, "test_metrics_by_year.csv"),
    out_summary_json=os.path.join(WEIGHT_DIR, "test_metrics.json"),
    device=DEVICE,
    batch_size=128
)

# —— 保存预处理与元信息（便于后续推理/SHAP） —— #
with open(os.path.join(WEIGHT_DIR, "preprocess_stcformer.pkl"), "wb") as f:
    pickle.dump({
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "prov_categories": list(prov_cat.categories),
        "city_categories": list(city_cat.categories),
        "district_categories": list(dist_cat.categories),   # 新增：区县类别
        "feature_cols": feat_cols,           # = x_cols + ["year_sin","year_cos"]
        "year_col": year_col,
        "y_col": y_col,
        "district_col": DISTRICT_COL_NAME,   # 新增：列名保存
        "config": {
            "TRAIN_YEARS": TRAIN_YEARS, "VAL_YEARS": VAL_YEARS, "TEST_YEARS": TEST_YEARS,
            "D_MODEL": D_MODEL, "NHEAD": NHEAD, "NLAYERS": NLAYERS, "FFN_DIM": FFN_DIM,
            "DROPOUT": DROPOUT, "SEED": SEED,
            "PROV_EMB_DIM": PROV_EMB_DIM, "CITY_EMB_DIM": CITY_EMB_DIM,
            "FILM_SCALE_G": FILM_SCALE_G, "FILM_SCALE_B": FILM_SCALE_B
        }
    }, f)

print(f"\n已保存: {BEST_MODEL_PATH}, {os.path.join(WEIGHT_DIR, 'preprocess_stcformer.pkl')}")
print(f"训练&验证度量日志: {METRICS_CSV}")
