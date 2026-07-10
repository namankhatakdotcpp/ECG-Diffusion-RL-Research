"""
step05_baseline_eval.py — Baseline quantitative evaluation of the diffusion model.

Produces Table 1 of the paper — the numbers RL fine-tuning must beat.
All metrics are reported as mean ± std across 3 seeds (config.eval.seeds).

Metrics:
  DTW       — Dynamic Time Warping (Lead II, nearest-neighbour to test set)
  MMD       — Maximum Mean Discrepancy with RBF kernel (Lead II)
  FED       — Fréchet ECG Distance in CNN embedding space (Lead II, 128-dim)
  MorphVal  — % of generated ECGs with valid PQRST morphology (neurokit2)
  TSTR      — Train on Synthetic, Test on Real (12-lead 1D CNN, macro F1)
  TRTR      — Train on Real,      Test on Real (same architecture, macro F1)

Reads from:
  outputs/models/diffusion_best.pt
  outputs/processed/{X_train,X_val,X_test}.npy
  outputs/processed/record_ids_{train,val,test}.npy
  outputs/processed/{class_names,class_mapping}.json
  outputs/processed/morphology_stats.json
  outputs/processed/preprocessing_stats.json
  data/ptbxl/ptbxl_database.csv

Writes to:
  outputs/results/baseline_metrics.json
  outputs/results/baseline_metrics_table.tex
  outputs/results/fig03_real_vs_generated_baseline.{pdf,png}
"""

from __future__ import annotations

import ast
import json
import math
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg as sp_linalg
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed
from step04_transformer_diffusion import (
    ECGTransformerDiffusion, GaussianDiffusion, EMA, generate_ecg,
)

warnings.filterwarnings("ignore")

LEAD_II = 1      # index of Lead II in the 12-lead array
PUBSTYLE = {
    "font.size": 11, "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


# ──────────────────────────────────────────────────────────────────────────────
# Neural network definitions
# ──────────────────────────────────────────────────────────────────────────────

class _ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x):
        return self.net(x)


class Simple1DCNN(nn.Module):
    """
    12-lead ECG classifier for TSTR / TRTR evaluation.
    Input: (B, 12, 1000)  Output: (B, n_classes) logits
    """

    def __init__(self, n_classes: int, in_channels: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBlock1D(in_channels, 32,  7, pool=4),   # → (B, 32,  250)
            _ConvBlock1D(32,          64,  5, pool=4),   # → (B, 64,   62)
            _ConvBlock1D(64,          128, 5, pool=2),   # → (B, 128,  31)
            nn.AdaptiveAvgPool1d(1),                     # → (B, 128,   1)
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x).squeeze(-1))


class FEDEncoder(nn.Module):
    """
    Small CNN encoder for Fréchet ECG Distance.
    Trained as a classifier on real Lead-II data, then the 128-dim penultimate
    layer is used as the embedding for distribution comparison.
    Input: (B, 1, 1000)  Output: (B, 128) embedding
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            _ConvBlock1D(1,  32,  7, pool=4),   # → (B, 32,  250)
            _ConvBlock1D(32, 64,  5, pool=4),   # → (B, 64,   62)
            _ConvBlock1D(64, 128, 5, pool=2),   # → (B, 128,  31)
            nn.AdaptiveAvgPool1d(1),             # → (B, 128,   1)
        )
        self.head = nn.Linear(128, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return 128-dim embedding without classification head."""
        return self.encoder(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_scp(raw: str) -> dict[str, float]:
    try:
        return ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        try:
            return json.loads(str(raw).replace("'", '"'))
        except Exception:
            return {}


def _load_class_labels(
    record_ids:    np.ndarray,
    ptbxl_db:     pd.DataFrame,
    class_mapping: dict[str, str],
    class_names:  list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Map record_ids → class indices via ptbxl_database.csv + class_mapping.json."""
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    valid, labels = [], []
    for i, eid in enumerate(record_ids):
        eid = int(eid)
        if eid not in ptbxl_db.index:
            continue
        scp = _parse_scp(ptbxl_db.at[eid, "scp_codes"])
        best_cls, best_conf = None, -1.0
        for code, conf in scp.items():
            m = class_mapping.get(code.upper())
            if m and m in name_to_idx and conf > best_conf:
                best_cls, best_conf = m, conf
        if best_cls is None:
            if "OTHER" in name_to_idx:
                best_cls = "OTHER"
            else:
                continue
        valid.append(i)
        labels.append(name_to_idx[best_cls])
    return np.array(valid, dtype=np.int64), np.array(labels, dtype=np.int64)


def _load_real_data(cfg, log) -> dict:
    """Load and label X_{train,val,test}.npy using the dynamic class mapping."""
    proc = Path(cfg.paths.outputs.processed)
    db_path = Path(cfg.paths.data.ptbxl) / "ptbxl_database.csv"

    with open(proc / "class_names.json")   as f: class_names   = json.load(f)
    with open(proc / "class_mapping.json") as f: class_mapping = json.load(f)
    log.info(f"Classes: {class_names}")

    ptbxl_db = pd.read_csv(str(db_path), index_col="ecg_id")

    out = {"class_names": class_names, "class_mapping": class_mapping}
    for split in ("train", "val", "test"):
        X   = np.load(str(proc / f"X_{split}.npy"))          # (N, 1000, 12)
        ids = np.load(str(proc / f"record_ids_{split}.npy"))
        vi, lbls = _load_class_labels(ids, ptbxl_db, class_mapping, class_names)
        out[f"X_{split}"]  = X[vi]
        out[f"y_{split}"]  = lbls
        log.info(f"  {split}: {X[vi].shape}  dist={dict(Counter(class_names[l] for l in lbls.tolist()))}")

    stats_path = proc / "preprocessing_stats.json"
    out["prep_stats"] = json.load(open(stats_path)) if stats_path.exists() else None

    morph_path = proc / "morphology_stats.json"
    out["morph_stats"] = json.load(open(morph_path)) if morph_path.exists() else {}

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Model loading and generation
# ──────────────────────────────────────────────────────────────────────────────

def _load_diffusion_model(cfg, log) -> tuple:
    """Load diffusion_best.pt → (model, diffusion, ema, class_names)."""
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    best_path = Path(cfg.paths.outputs.models) / "diffusion_best.pt"

    if not best_path.exists():
        log.error(f"diffusion_best.pt not found at {best_path}. Run step04 first.")
        raise FileNotFoundError(best_path)

    log.info(f"Loading diffusion model from {best_path} …")
    ckpt        = torch.load(str(best_path), map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = ckpt["n_classes"]

    model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ema = EMA(model, decay=float(cfg.diffusion.ema_decay))
    ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}

    diffusion = GaussianDiffusion(
        T=int(cfg.diffusion.T),
        beta_schedule=str(cfg.diffusion.beta_schedule),
        device=device,
    )
    log.info(f"Model loaded. n_classes={n_classes}, classes={class_names}")
    return model, diffusion, ema, class_names, device


def _generate_all_classes(
    model:       nn.Module,
    diffusion:   GaussianDiffusion,
    ema:         EMA,
    class_names: list[str],
    n_per_class: int,
    cfg,
    seed:        int,
    device:      str,
    prep_stats:  Optional[dict],
    log,
) -> dict[str, np.ndarray]:
    """
    Generate n_per_class ECGs for every class using EMA weights.

    Returns dict: class_name → (n_per_class, 1000, 12) in z-score space
    (denormalisation is done externally if needed for figures).
    """
    log.info(f"  Generating {n_per_class}/class × {len(class_names)} classes (seed={seed}) …")
    gen: dict[str, np.ndarray] = {}

    with ema.ema_scope(model):
        for cls_idx, cls_name in enumerate(class_names):
            samples = generate_ecg(
                model, diffusion,
                class_label=cls_idx,
                n_samples=n_per_class,
                device=device,
                cfg=cfg,
                seed=seed + cls_idx,
                stats=None,   # keep in z-score space for metric computation
            )  # (n_per_class, 1000, 12)
            gen[cls_name] = samples

    return gen


# ──────────────────────────────────────────────────────────────────────────────
# Metric 1 — DTW
# ──────────────────────────────────────────────────────────────────────────────

def _dtw_distance_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Fast DTW via tslearn (if available) else plain O(n²) DP."""
    try:
        from tslearn.metrics import dtw as tslearn_dtw
        return float(tslearn_dtw(a, b))
    except ImportError:
        # Pure-NumPy fallback (slower but always available)
        n, m = len(a), len(b)
        dp = np.full((n + 1, m + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = (a[i - 1] - b[j - 1]) ** 2
                dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(np.sqrt(dp[n, m]))


def _metric_dtw(
    gen:         dict[str, np.ndarray],
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_names: list[str],
    n_subsample: int,
    rng:         np.random.Generator,
) -> dict[str, float]:
    """
    For each generated ECG (Lead II), compute DTW distance to nearest real ECG
    of the same class in the test set.

    Subsampled to n_subsample generated × n_subsample real per class for speed.
    Returns per-class mean DTW and overall mean.
    """
    per_class: dict[str, float] = {}

    for cls_idx, cls_name in enumerate(class_names):
        gen_cls  = gen[cls_name][:, :, LEAD_II]            # (N_gen, 1000)
        real_mask = (y_test == cls_idx)
        if real_mask.sum() == 0:
            continue

        real_cls = X_test[real_mask][:, :, LEAD_II]        # (M, 1000)

        # Subsample
        g_idx = rng.choice(len(gen_cls),  size=min(n_subsample, len(gen_cls)),  replace=False)
        r_idx = rng.choice(len(real_cls), size=min(n_subsample, len(real_cls)), replace=False)
        gen_sub  = gen_cls[g_idx]
        real_sub = real_cls[r_idx]

        nn_dists: list[float] = []
        for g in gen_sub:
            dists = [_dtw_distance_pair(g, r) for r in real_sub]
            nn_dists.append(min(dists))

        per_class[cls_name] = float(np.mean(nn_dists))

    overall = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    return {"per_class": per_class, "overall": overall}


# ──────────────────────────────────────────────────────────────────────────────
# Metric 2 — MMD
# ──────────────────────────────────────────────────────────────────────────────

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """RBF kernel matrix K(X, Y) with bandwidth sigma."""
    XY = -2.0 * X @ Y.T + (X ** 2).sum(1, keepdims=True) + (Y ** 2).sum(1)
    return np.exp(-XY / (2.0 * sigma ** 2))


def _mmd_rbf(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased MMD² estimate with RBF kernel, bandwidth from median heuristic.
    X, Y: (n, d) float arrays.
    """
    # Median heuristic: bandwidth = median pairwise distance / sqrt(2 log n)
    XY     = np.vstack([X, Y])
    pairwise = np.sum((XY[:, None, :] - XY[None, :, :]) ** 2, axis=-1)
    sigma  = float(np.sqrt(np.median(pairwise[pairwise > 0]) / 2.0))
    if sigma < 1e-8:
        sigma = 1.0

    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)

    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    mmd2 = (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
            - 2.0 * Kxy.mean())
    return float(max(mmd2, 0.0))


def _metric_mmd(
    gen:         dict[str, np.ndarray],
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_names: list[str],
    max_samples: int = 500,
    rng:         Optional[np.random.Generator] = None,
) -> dict[str, float]:
    """MMD² (Lead II) per class and overall."""
    if rng is None:
        rng = np.random.default_rng(42)
    per_class: dict[str, float] = {}

    for cls_idx, cls_name in enumerate(class_names):
        gen_cls  = gen[cls_name][:, :, LEAD_II]      # (N, 1000)
        real_mask = (y_test == cls_idx)
        if real_mask.sum() == 0:
            continue
        real_cls = X_test[real_mask][:, :, LEAD_II]

        # Cap for tractability
        gi = rng.choice(len(gen_cls),  size=min(max_samples, len(gen_cls)),  replace=False)
        ri = rng.choice(len(real_cls), size=min(max_samples, len(real_cls)), replace=False)

        per_class[cls_name] = _mmd_rbf(gen_cls[gi], real_cls[ri])

    overall = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    return {"per_class": per_class, "overall": overall}


# ──────────────────────────────────────────────────────────────────────────────
# Metric 3 — Fréchet ECG Distance (FED)
# ──────────────────────────────────────────────────────────────────────────────

def _frechet_distance(
    mu1: np.ndarray, sig1: np.ndarray,
    mu2: np.ndarray, sig2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Fréchet distance between two multivariate Gaussians N(μ₁,Σ₁) and N(μ₂,Σ₂).

    FD = ||μ₁-μ₂||² + Tr(Σ₁ + Σ₂ - 2√(Σ₁Σ₂))
    """
    diff = mu1 - mu2
    # Regularise covariances to ensure positive-definiteness
    sig1 = sig1 + eps * np.eye(sig1.shape[0])
    sig2 = sig2 + eps * np.eye(sig2.shape[0])

    covmean, _ = sp_linalg.sqrtm(sig1 @ sig2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # discard negligible imaginary part

    fd = float(diff @ diff + np.trace(sig1 + sig2 - 2.0 * covmean))
    return max(fd, 0.0)


def _train_fed_encoder(
    X_train:  np.ndarray,   # (N, 1000, 12) z-score normalised
    y_train:  np.ndarray,   # (N,) integer labels
    n_classes: int,
    cfg,
    device:   str,
    log,
) -> FEDEncoder:
    """
    Train the FED encoder on real Lead-II data.
    Uses CrossEntropyLoss. Runs once, result reused across evaluation seeds.
    """
    ecfg  = cfg.eval
    enc   = FEDEncoder(n_classes=n_classes).to(device)
    opt   = torch.optim.Adam(enc.parameters(), lr=float(ecfg.fed_encoder_lr))

    # Lead II only: (N, 1, 1000)
    X = torch.from_numpy(X_train[:, :, LEAD_II : LEAD_II + 1].transpose(0, 2, 1)).float()
    y = torch.from_numpy(y_train).long()

    counts   = Counter(y_train.tolist())
    wts      = torch.tensor([1.0 / counts[i] for i in range(n_classes)], dtype=torch.float32)
    sampler  = WeightedRandomSampler(
        weights=[1.0 / counts[int(l)] for l in y_train.tolist()],
        num_samples=len(y_train), replacement=True,
    )
    loader   = DataLoader(
        TensorDataset(X, y),
        batch_size=int(ecfg.fed_encoder_batch_size),
        sampler=sampler,
    )
    criterion = nn.CrossEntropyLoss(weight=wts.to(device))

    log.info(f"Training FED encoder for {int(ecfg.fed_encoder_epochs)} epochs …")
    enc.train()
    for ep in range(1, int(ecfg.fed_encoder_epochs) + 1):
        ep_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = enc(bx)
            loss   = criterion(logits, by)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        if ep % 5 == 0:
            log.info(f"  FED encoder epoch {ep:02d}/{int(ecfg.fed_encoder_epochs)} "
                     f"loss={ep_loss/len(loader):.4f}")

    enc.eval()
    return enc


@torch.no_grad()
def _embed(
    encoder: FEDEncoder,
    X:       np.ndarray,   # (N, 1000, 12)
    device:  str,
    batch:   int = 256,
) -> np.ndarray:
    """Extract 128-dim embeddings from Lead II using the FED encoder."""
    X_t = torch.from_numpy(X[:, :, LEAD_II : LEAD_II + 1].transpose(0, 2, 1)).float()
    embs: list[np.ndarray] = []
    for i in range(0, len(X_t), batch):
        embs.append(encoder.embed(X_t[i : i + batch].to(device)).cpu().numpy())
    return np.vstack(embs)


def _metric_fed(
    gen:      dict[str, np.ndarray],
    X_test:   np.ndarray,
    y_test:   np.ndarray,
    class_names: list[str],
    encoder:  FEDEncoder,
    device:   str,
) -> dict[str, float]:
    """Fréchet ECG Distance (FED) per class and overall."""
    per_class: dict[str, float] = {}

    for cls_idx, cls_name in enumerate(class_names):
        gen_cls   = gen[cls_name]                         # (N, 1000, 12)
        real_mask = (y_test == cls_idx)
        if real_mask.sum() < 10:
            continue

        emb_gen  = _embed(encoder, gen_cls,                 device)  # (N, 128)
        emb_real = _embed(encoder, X_test[real_mask], device)  # (M, 128)

        mu_g, sig_g = emb_gen.mean(0),  np.cov(emb_gen.T)
        mu_r, sig_r = emb_real.mean(0), np.cov(emb_real.T)
        per_class[cls_name] = _frechet_distance(mu_g, sig_g, mu_r, sig_r)

    # Overall: pool all embeddings
    all_gen  = np.vstack([_embed(encoder, gen[c],          device) for c in class_names])
    all_real = _embed(encoder, X_test, device)
    mu_g, sig_g = all_gen.mean(0),  np.cov(all_gen.T)
    mu_r, sig_r = all_real.mean(0), np.cov(all_real.T)
    overall = _frechet_distance(mu_g, sig_g, mu_r, sig_r)

    return {"per_class": per_class, "overall": overall}


# ──────────────────────────────────────────────────────────────────────────────
# Metric 4 — Morphological Validity
# ──────────────────────────────────────────────────────────────────────────────

def _extract_morphology_one(signal_lead2: np.ndarray, fs: float) -> Optional[dict[str, float]]:
    """
    Run neurokit2 on a single Lead-II trace and return median beat intervals (ms).
    Returns None on delineation failure.
    """
    try:
        import neurokit2 as nk
    except ImportError:
        raise ImportError("neurokit2 required — pip install neurokit2")

    try:
        signals, info = nk.ecg_process(signal_lead2.astype(np.float64), sampling_rate=fs)
    except Exception:
        return None

    r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    if len(r_peaks) < 3:
        return None

    ms = 1000.0 / fs

    def _locs(col: str) -> np.ndarray:
        return np.where(signals.get(col, pd.Series(dtype=int)).fillna(0).astype(int) == 1)[0]

    p_locs = _locs("ECG_P_Peaks")
    q_locs = _locs("ECG_Q_Peaks")
    s_locs = _locs("ECG_S_Peaks")
    t_locs = _locs("ECG_T_Peaks")

    q_win  = int(0.06 * fs)
    t_win  = int(0.25 * fs)
    pr_list, qrs_list, qt_list = [], [], []

    for r in r_peaks:
        pre_p  = p_locs[p_locs < r]
        q_near = q_locs[(q_locs < r) & (q_locs > r - q_win)]
        s_near = s_locs[(s_locs > r) & (s_locs < r + q_win)]
        t_near = t_locs[(t_locs > r) & (t_locs < r + t_win)]

        if len(pre_p) and len(q_near):
            pr = (r - pre_p[-1]) * ms
            if 60 < pr < 400:
                pr_list.append(pr)
        if len(q_near) and len(s_near):
            qrs = (s_near[0] - q_near[-1]) * ms
            if 40 < qrs < 200:
                qrs_list.append(qrs)
        if len(q_near) and len(t_near):
            qt = (t_near[0] - q_near[-1]) * ms
            if 200 < qt < 650:
                qt_list.append(qt)

    result: dict[str, float] = {}
    if pr_list:  result["pr_ms"]  = float(np.median(pr_list))
    if qrs_list: result["qrs_ms"] = float(np.median(qrs_list))
    if qt_list:  result["qt_ms"]  = float(np.median(qt_list))
    return result if result else None


def _metric_morphology(
    gen:         dict[str, np.ndarray],
    morph_stats: dict,
    class_names: list[str],
    fs:          float,
    n_eval:      int,
    rng:         np.random.Generator,
    log,
) -> dict[str, float]:
    """
    % of generated ECGs whose PQRST intervals fall within mean ± 2σ of real stats.

    Only classes with reference morphology stats are scored.
    """
    try:
        import neurokit2  # noqa: F401
    except ImportError:
        log.warning("neurokit2 not installed — skipping morphology validity.")
        return {"per_class": {}, "overall": float("nan")}

    per_class: dict[str, float] = {}

    for cls_name in class_names:
        ref = morph_stats.get(cls_name, {})
        if not ref:
            log.info(f"  No morphology reference for {cls_name} — skipped.")
            continue

        gen_cls = gen[cls_name][:, :, LEAD_II]   # (N, 1000)
        idx     = rng.choice(len(gen_cls), size=min(n_eval, len(gen_cls)), replace=False)
        n_valid = 0
        n_ok    = 0
        n_fail  = 0

        for i in idx:
            result = _extract_morphology_one(gen_cls[i], fs)
            if result is None:
                n_fail += 1
                continue
            n_valid += 1

            ecg_ok = True
            for key in ("pr_ms", "qrs_ms", "qt_ms"):
                if key not in ref or key not in result:
                    continue
                ref_mean = ref[key]["mean"]
                ref_std  = ref[key]["std"]
                lo, hi   = ref_mean - 2 * ref_std, ref_mean + 2 * ref_std
                if not (lo <= result[key] <= hi):
                    ecg_ok = False
                    break
            if ecg_ok:
                n_ok += 1

        if n_valid > 0:
            pct = 100.0 * n_ok / n_valid
            per_class[cls_name] = pct
            log.info(f"  {cls_name}: {pct:.1f}% valid  ({n_ok}/{n_valid} scored, {n_fail} nk2 failures)")

    overall = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    return {"per_class": per_class, "overall": overall}


# ──────────────────────────────────────────────────────────────────────────────
# Metric 5 — TSTR / TRTR
# ──────────────────────────────────────────────────────────────────────────────

def _train_eval_cnn(
    X_train:   np.ndarray,    # (N, 1000, 12)
    y_train:   np.ndarray,    # (N,)
    X_test:    np.ndarray,    # (M, 1000, 12)
    y_test:    np.ndarray,    # (M,)
    n_classes: int,
    cfg,
    device:    str,
    label:     str = "CNN",
    log=None,
    save_path: Optional[Path] = None,
) -> dict[str, float]:
    """
    Train Simple1DCNN on (X_train, y_train) and evaluate on (X_test, y_test).
    Returns per-class F1 and macro F1.
    """
    ecfg = cfg.eval

    # Build tensors — input shape (B, 12, 1000)
    Xtr = torch.from_numpy(X_train.transpose(0, 2, 1)).float()
    ytr = torch.from_numpy(y_train).long()
    Xte = torch.from_numpy(X_test.transpose(0, 2, 1)).float()
    yte = torch.from_numpy(y_test).long()

    counts  = Counter(y_train.tolist())
    sampler = WeightedRandomSampler(
        weights=[1.0 / counts[int(l)] for l in y_train.tolist()],
        num_samples=len(y_train), replacement=True,
    )
    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=int(ecfg.tstr_batch_size),
        sampler=sampler,
        drop_last=True,
    )
    model_cnn = Simple1DCNN(n_classes=n_classes).to(device)
    opt       = torch.optim.Adam(model_cnn.parameters(), lr=float(ecfg.tstr_lr))
    criterion = nn.CrossEntropyLoss()

    model_cnn.train()
    for ep in range(1, int(ecfg.tstr_epochs) + 1):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model_cnn(bx)
            loss   = criterion(logits, by)
            opt.zero_grad(); loss.backward(); opt.step()

    # Evaluate
    model_cnn.eval()
    all_pred: list[int] = []
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=256, shuffle=False)
    with torch.no_grad():
        for bx, _ in test_loader:
            preds = model_cnn(bx.to(device)).argmax(dim=1).cpu().numpy()
            all_pred.extend(preds.tolist())

    preds_arr = np.array(all_pred)
    accuracy  = float(accuracy_score(y_test, preds_arr))
    macro_f1  = float(f1_score(y_test, preds_arr, average="macro", zero_division=0))
    per_class_f1 = f1_score(y_test, preds_arr, average=None, zero_division=0)
    if log:
        log.info(f"  {label}: accuracy={accuracy:.4f}  macro_F1={macro_f1:.4f}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model_cnn.state_dict(), "n_classes": n_classes}, str(save_path))

    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class_f1": per_class_f1.tolist()}


def _metric_tstr_trtr(
    gen:         dict[str, np.ndarray],   # synthetic, z-score space
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_names: list[str],
    n_per_class: int,
    cfg,
    device:      str,
    log,
    trtr_cache:  Optional[dict] = None,
) -> tuple[dict, dict]:
    """
    TSTR: train Simple1DCNN on synthetic (balanced, n_per_class/class),
          test on X_test.
    TRTR: train same architecture on X_train (weighted sampling), test on X_test.
    Returns (tstr_result, trtr_result).
    """
    n_classes = len(class_names)

    # ── TSTR — synthetic training data ────────────────────────────────────────
    X_syn_list, y_syn_list = [], []
    for cls_idx, cls_name in enumerate(class_names):
        samps = gen[cls_name][:n_per_class]   # (n_per_class, 1000, 12)
        X_syn_list.append(samps)
        y_syn_list.append(np.full(len(samps), cls_idx, dtype=np.int64))
    X_syn = np.concatenate(X_syn_list, axis=0)
    y_syn = np.concatenate(y_syn_list, axis=0)
    shuffle = np.random.permutation(len(X_syn))
    X_syn, y_syn = X_syn[shuffle], y_syn[shuffle]

    tstr_save = Path(cfg.paths.outputs.models) / "tstr_classifier.pt"
    tstr = _train_eval_cnn(
        X_syn, y_syn, X_test, y_test, n_classes, cfg, device, "TSTR", log,
        save_path=tstr_save if not tstr_save.exists() else None,
    )

    # ── TRTR — real training data (cached across seeds since data is identical) ─
    # Saved to disk (mirrors tstr_save above) so DiagnosticUtilityReward can use
    # a real-data-trained classifier instead of tstr_classifier.pt, which is
    # trained entirely on synthetic samples from the baseline diffusion model --
    # a reward-hacking risk when used to fine-tune that same model (see
    # Roadmap/Stage_4_Optimization/Decisions.md).
    trtr_save = Path(cfg.paths.outputs.models) / "trtr_classifier.pt"
    if trtr_cache is not None and "macro_f1" in trtr_cache:
        trtr = trtr_cache
    else:
        trtr = _train_eval_cnn(
            X_train, y_train, X_test, y_test, n_classes, cfg, device, "TRTR", log,
            save_path=trtr_save if not trtr_save.exists() else None,
        )
        if trtr_cache is not None:
            trtr_cache.update(trtr)

    # Explicit, separate artifact -- not just a log line -- so this number is
    # easy to pull into Decisions.md before the Diagnostic reward weight is
    # treated as final, same evidentiary bar as the Mentor Classifier's own
    # reported accuracy/macro-F1.
    trtr_eval_path = Path(cfg.paths.outputs.models) / "trtr_classifier_eval.json"
    import json as _json
    trtr_eval_path.write_text(_json.dumps(
        {"accuracy": trtr.get("accuracy"), "macro_f1": trtr.get("macro_f1"),
         "per_class_f1": trtr.get("per_class_f1")}, indent=2
    ))
    if log:
        log.info(
            f"  TRTR classifier (real-data, for RL reward use): "
            f"accuracy={trtr.get('accuracy')}  macro_f1={trtr.get('macro_f1')} "
            f"-- saved {trtr_save.name}, eval written to {trtr_eval_path.name}"
        )

    return tstr, trtr


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Real vs Generated comparison
# ──────────────────────────────────────────────────────────────────────────────

def _make_figure3(
    gen:         dict[str, np.ndarray],
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_names: list[str],
    fs:          float,
    results_dir: Path,
    rng:         np.random.Generator,
) -> None:
    """
    Paper Figure 3: Real vs Generated ECG examples (Lead II, 10 s).

    Layout: 4 rows × 6 columns
      Row 0: Real NORM   (6 examples)
      Row 1: Generated NORM (6 examples)
      Row 2: Real MI     (6 examples)
      Row 3: Generated MI   (6 examples)
    """
    target_pairs = [("NORM", "#2ca02c"), ("MI", "#d62728")]
    target_pairs = [(c, col) for c, col in target_pairs if c in class_names]
    if not target_pairs:
        target_pairs = [(class_names[0], "#1f77b4")]

    n_cols = 6
    n_rows = 2 * len(target_pairs)
    t_axis = np.arange(int(fs * 10)) / fs   # 1000 samples / 100 Hz = 10 s

    with plt.rc_context(PUBSTYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows),
                                 constrained_layout=True)
        if n_rows == 2:
            axes = axes[np.newaxis, :, :]   # uniform 3-D indexing

        for pair_idx, (cls_name, color) in enumerate(target_pairs):
            cls_idx  = class_names.index(cls_name)
            row_real = pair_idx * 2
            row_gen  = pair_idx * 2 + 1

            # Real examples from test set
            real_mask = (y_test == cls_idx)
            real_idx  = np.where(real_mask)[0]
            chosen_r  = rng.choice(real_idx, size=min(n_cols, len(real_idx)), replace=False)

            # Generated examples
            gen_cls    = gen[cls_name]   # (N, 1000, 12)
            chosen_g   = rng.choice(len(gen_cls), size=n_cols, replace=False)

            for col in range(n_cols):
                # Real
                ax_r = axes[row_real, col]
                if col < len(chosen_r):
                    ax_r.plot(t_axis, X_test[chosen_r[col], :, LEAD_II],
                              color=color, linewidth=0.8, alpha=0.9)
                ax_r.set_ylim(-4.5, 4.5)
                ax_r.spines["top"].set_visible(False)
                ax_r.spines["right"].set_visible(False)
                ax_r.set_xticks([])
                if col == 0:
                    ax_r.set_ylabel(f"Real\n{cls_name}", fontsize=9, color=color, fontweight="bold")

                # Generated
                ax_g = axes[row_gen, col]
                ax_g.plot(t_axis, gen_cls[chosen_g[col], :, LEAD_II],
                          color=color, linewidth=0.8, alpha=0.9, linestyle="--")
                ax_g.set_ylim(-4.5, 4.5)
                ax_g.spines["top"].set_visible(False)
                ax_g.spines["right"].set_visible(False)
                if row_gen == n_rows - 1:
                    ax_g.set_xlabel("Time (s)", fontsize=8)
                else:
                    ax_g.set_xticks([])
                if col == 0:
                    ax_g.set_ylabel(f"Generated\n{cls_name}", fontsize=9,
                                    color=color, fontweight="bold", style="italic")

        fig.suptitle(
            "Lead II — Real vs Baseline Diffusion Model (ECGTransformerDiffusion)\n"
            "Each column = one 10-second 12-lead ECG excerpt",
            fontsize=11, y=1.01,
        )

    for ext in ("pdf", "png"):
        fig.savefig(str(results_dir / f"fig03_real_vs_generated_baseline.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Results serialisation
# ──────────────────────────────────────────────────────────────────────────────

def _agg(values: list[float]) -> tuple[float, float]:
    """Mean and std of a list of seed-level measurements."""
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def _save_results(
    seed_results:  list[dict],
    class_names:   list[str],
    results_dir:   Path,
    log,
) -> dict:
    """Aggregate across seeds and save JSON + LaTeX table."""

    # ── Aggregate top-level scalar metrics across seeds ───────────────────────
    def _collect(key: str) -> list[float]:
        return [r[key] for r in seed_results if key in r]

    dtw_vals   = _collect("dtw_overall")
    mmd_vals   = _collect("mmd_overall")
    fed_vals   = _collect("fed_overall")
    morph_vals = _collect("morph_overall")
    tstr_vals  = _collect("tstr_macro_f1")
    trtr_vals  = _collect("trtr_macro_f1")
    gap_vals   = [t - s for t, s in zip(_collect("trtr_macro_f1"), _collect("tstr_macro_f1"))]

    dtw_m,  dtw_s   = _agg(dtw_vals)
    mmd_m,  mmd_s   = _agg(mmd_vals)
    fed_m,  fed_s   = _agg(fed_vals)
    morph_m, morph_s = _agg(morph_vals)
    tstr_m, tstr_s  = _agg(tstr_vals)
    trtr_m, trtr_s  = _agg(trtr_vals)
    gap_m,  gap_s   = _agg(gap_vals)

    # ── Per-class aggregation ─────────────────────────────────────────────────
    per_class_agg: dict[str, dict] = {c: {} for c in class_names}
    for metric, key in [("dtw", "dtw_per_class"), ("mmd", "mmd_per_class"),
                         ("fed", "fed_per_class"), ("morph", "morph_per_class")]:
        for c in class_names:
            vals = [r[key].get(c, float("nan")) for r in seed_results if key in r]
            m, s = _agg(vals)
            per_class_agg[c][f"{metric}_mean"] = m
            per_class_agg[c][f"{metric}_std"]  = s

    # ── Per-class F1 ──────────────────────────────────────────────────────────
    for mode in ("tstr", "trtr"):
        key = f"{mode}_per_class_f1"
        for ci, c in enumerate(class_names):
            vals = [r[key][ci] for r in seed_results if key in r and ci < len(r[key])]
            m, s = _agg(vals)
            per_class_agg[c][f"{mode}_f1_mean"] = m
            per_class_agg[c][f"{mode}_f1_std"]  = s

    summary = {
        "n_seeds":  len(seed_results),
        "metrics": {
            "DTW":   {"mean": dtw_m,   "std": dtw_s,   "direction": "lower_is_better"},
            "MMD":   {"mean": mmd_m,   "std": mmd_s,   "direction": "lower_is_better"},
            "FED":   {"mean": fed_m,   "std": fed_s,   "direction": "lower_is_better"},
            "MorphValidity": {"mean": morph_m, "std": morph_s, "direction": "higher_is_better"},
            "TSTR_macro_F1": {"mean": tstr_m, "std": tstr_s, "direction": "higher_is_better"},
            "TRTR_macro_F1": {"mean": trtr_m, "std": trtr_s, "direction": "higher_is_better"},
            "TSTR_TRTR_gap": {"mean": gap_m,  "std": gap_s,  "direction": "lower_is_better"},
        },
        "per_class": per_class_agg,
        "raw_seeds": seed_results,
    }

    json_path = results_dir / "baseline_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved {json_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    def _fmt(m: float, s: float) -> str:
        if math.isnan(m):
            return "—"
        return f"{m:.4f} \\pm {s:.4f}"

    rows = [
        ("DTW $\\downarrow$",              _fmt(dtw_m,   dtw_s)),
        ("MMD $\\downarrow$",              _fmt(mmd_m,   mmd_s)),
        ("FED $\\downarrow$",              _fmt(fed_m,   fed_s)),
        ("Morph Validity (\\%) $\\uparrow$", _fmt(morph_m, morph_s)),
        ("TSTR macro F1 $\\uparrow$",      _fmt(tstr_m,  tstr_s)),
        ("TRTR macro F1 $\\uparrow$",      _fmt(trtr_m,  trtr_s)),
        ("TSTR-TRTR gap $\\downarrow$",    _fmt(gap_m,   gap_s)),
    ]

    latex = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Baseline Diffusion Model Evaluation (mean $\\pm$ std, "
        + str(len(seed_results)) + " seeds)}\n"
        "\\label{tab:baseline_metrics}\n"
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "Metric & Value \\\\\n"
        "\\midrule\n"
    )
    for metric, val in rows:
        latex += f"{metric} & ${val}$ \\\\\n"
    latex += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    tex_path = results_dir / "baseline_metrics_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    log.info(f"Saved {tex_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    log.info("")
    log.info("┌─────────────────────────────────────────────────┐")
    log.info("│            BASELINE EVALUATION RESULTS          │")
    log.info("├──────────────────────────────┬──────────────────┤")
    for metric, val in rows:
        clean = metric.replace("$\\downarrow$", "↓").replace("$\\uparrow$", "↑").replace("\\", "")
        val_display = val.replace('\\pm', '±')
        log.info(f"│ {clean:<28} │ {val_display:>16} │")
    log.info("└──────────────────────────────┴──────────────────┘")

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(cfg, log) -> float:
    results_dir = Path(cfg.paths.outputs.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load real data and model ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Loading real data …")
    log.info("=" * 60)
    real = _load_real_data(cfg, log)

    class_names = real["class_names"]
    n_classes   = len(class_names)
    X_train, y_train = real["X_train"], real["y_train"]
    X_test,  y_test  = real["X_test"],  real["y_test"]
    fs = float(cfg.ptbxl.sampling_rate)

    log.info("Loading diffusion model …")
    model, diffusion, ema, ckpt_classes, device = _load_diffusion_model(cfg, log)
    # Use checkpoint's class list if available (may differ from step03 if re-run)
    if ckpt_classes != class_names:
        log.warning(f"Checkpoint class list {ckpt_classes} differs from step03 {class_names}. "
                    "Using checkpoint class list.")
        class_names = ckpt_classes
        n_classes   = len(class_names)

    ecfg = cfg.eval
    eval_seeds  = list(ecfg.seeds)
    n_per_class = int(ecfg.n_synthetic_per_class)

    # ── Train FED encoder once (seed-independent) ─────────────────────────────
    log.info("=" * 60)
    log.info("Training FED encoder on real data …")
    log.info("=" * 60)
    fed_encoder = _train_fed_encoder(X_train, y_train, n_classes, cfg, device, log)

    # ── Cache TRTR result (same real data across seeds) ───────────────────────
    log.info("=" * 60)
    log.info("Computing TRTR baseline (train on real, test on real) …")
    log.info("=" * 60)
    trtr_cache: dict = {}
    trtr_save = Path(cfg.paths.outputs.models) / "trtr_classifier.pt"
    trtr_result = _train_eval_cnn(
        X_train, y_train, X_test, y_test, n_classes, cfg, device, "TRTR", log,
        save_path=trtr_save if not trtr_save.exists() else None,
    )
    trtr_cache.update(trtr_result)

    rng_global = np.random.default_rng(42)

    # ── Per-seed evaluation loop ──────────────────────────────────────────────
    seed_results: list[dict] = []
    gen_for_figure: Optional[dict] = None   # save first seed's gen for Figure 3

    for seed in eval_seeds:
        log.info("=" * 60)
        log.info(f"Seed {seed} …")
        log.info("=" * 60)
        set_seed(seed)
        rng = np.random.default_rng(seed)

        # Generate
        gen = _generate_all_classes(
            model, diffusion, ema, class_names,
            n_per_class=n_per_class, cfg=cfg, seed=seed,
            device=device, prep_stats=real.get("prep_stats"), log=log,
        )
        if gen_for_figure is None:
            gen_for_figure = gen

        # DTW
        log.info("  → DTW …")
        dtw_res = _metric_dtw(
            gen, X_test, y_test, class_names,
            n_subsample=int(ecfg.dtw_subsample), rng=rng,
        )

        # MMD
        log.info("  → MMD …")
        mmd_res = _metric_mmd(gen, X_test, y_test, class_names, rng=rng)

        # FED
        log.info("  → FED …")
        fed_res = _metric_fed(gen, X_test, y_test, class_names, fed_encoder, device)

        # Morphology validity
        log.info("  → Morphological validity …")
        morph_res = _metric_morphology(
            gen, real["morph_stats"], class_names, fs,
            n_eval=int(ecfg.n_morphology_eval), rng=rng, log=log,
        )

        # TSTR
        log.info("  → TSTR …")
        tstr_res, _ = _metric_tstr_trtr(
            gen, X_train, y_train, X_test, y_test,
            class_names, n_per_class, cfg, device, log,
            trtr_cache=trtr_cache,
        )

        seed_results.append({
            "seed":            seed,
            "dtw_overall":     dtw_res["overall"],
            "dtw_per_class":   dtw_res["per_class"],
            "mmd_overall":     mmd_res["overall"],
            "mmd_per_class":   mmd_res["per_class"],
            "fed_overall":     fed_res["overall"],
            "fed_per_class":   fed_res["per_class"],
            "morph_overall":   morph_res["overall"],
            "morph_per_class": morph_res["per_class"],
            "tstr_macro_f1":   tstr_res["macro_f1"],
            "tstr_per_class_f1": tstr_res["per_class_f1"],
            "trtr_macro_f1":   trtr_cache["macro_f1"],
            "trtr_per_class_f1": trtr_cache["per_class_f1"],
        })

        log.info(
            f"  Seed {seed} summary: "
            f"DTW={dtw_res['overall']:.4f}  "
            f"MMD={mmd_res['overall']:.6f}  "
            f"FED={fed_res['overall']:.2f}  "
            f"Morph={morph_res['overall']:.1f}%  "
            f"TSTR-F1={tstr_res['macro_f1']:.4f}  "
            f"TRTR-F1={trtr_cache['macro_f1']:.4f}"
        )

    # ── Figure 3 ─────────────────────────────────────────────────────────────
    log.info("Generating Figure 3 (Real vs Generated) …")
    if gen_for_figure is not None:
        _make_figure3(gen_for_figure, X_test, y_test, class_names, fs, results_dir, rng_global)
        log.info("Saved fig03_real_vs_generated_baseline.{pdf,png}")

    # ── Aggregate and save ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Saving results …")
    log.info("=" * 60)
    summary = _save_results(seed_results, class_names, results_dir, log)

    tstr_macro = summary["metrics"]["TSTR_macro_F1"]["mean"]
    return tstr_macro


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step05_baseline_eval", cfg=cfg)
    set_seed(cfg.seeds[0])

    tstr_f1 = evaluate(cfg, log)
    print(f"✓ Baseline evaluation complete. TSTR macro F1: {tstr_f1:.3f}")


if __name__ == "__main__":
    main()
