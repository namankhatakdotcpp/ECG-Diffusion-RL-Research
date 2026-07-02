"""
step04_transformer_diffusion.py — Transformer-backbone conditional diffusion model.

Architecture: ECGTransformerDiffusion
  Patchify 12-lead ECG → 600 tokens (12 leads × 50 patches of size 20)
  BERT-style Transformer encoder (pre-norm LN, 6 layers, 8 heads)
  Condition: sinusoidal time emb + class emb broadcast to every token
  Unpatchify back to (B, 12, 1000)
  Training objective: ε-prediction (Ho et al. DDPM 2020)
  Sampling: DDIM, 50 deterministic steps (η=0)

Reads from:
  outputs/processed/X_train.npy            (N, 1000, 12) preprocessed signals
  outputs/processed/X_val.npy
  outputs/processed/record_ids_train.npy   ecg_id for each training record
  outputs/processed/record_ids_val.npy
  outputs/processed/class_names.json       ordered class list from step03
  outputs/processed/class_mapping.json     SCP code → class name
  outputs/processed/preprocessing_stats.json  per-lead mean/std (for denorm)
  data/ptbxl/ptbxl_database.csv            metadata (scp_codes, strat_fold)

Writes to:
  outputs/models/diffusion_best.pt
  outputs/models/diffusion_ckpt_ep{epoch:04d}.pt
  outputs/models/diffusion_architecture.json
  outputs/logs/diffusion_training_log.csv
  outputs/generated/baseline_samples/class_{NAME}_sample_{N:04d}.npy
  outputs/results/diffusion_val_ep{epoch:04d}.png
"""

from __future__ import annotations

import ast
import csv
import json
import math
import time
from collections import Counter
from contextlib import contextmanager
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
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed, assign_primary_class
from utils.backup import snapshot_before_write

# ──────────────────────────────────────────────────────────────────────────────
# Noise schedule helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_betas(T: int, s: float = 0.008) -> Tensor:
    """Improved cosine schedule (Nichol & Dhariwal 2021)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    f = torch.cos(((x / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-4, 0.9999)


def _sinusoidal_time_emb(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal time step embedding → (B, dim)."""
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    x = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    return torch.cat([x.sin(), x.cos()], dim=-1)      # (B, dim)


def _sinusoidal_pos_emb(seq_len: int, dim: int) -> Tensor:
    """Static sinusoidal position embedding (Vaswani et al.) → (seq_len, dim)."""
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim))
    emb = torch.zeros(seq_len, dim)
    emb[:, 0::2] = torch.sin(pos * div)
    emb[:, 1::2] = torch.cos(pos * div)
    return emb


# ──────────────────────────────────────────────────────────────────────────────
# Model components
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbed1D(nn.Module):
    """
    Patchify a 12-lead ECG into 600 tokens and project to model_dim.

    Each lead is split into n_patches = signal_len // patch_size non-overlapping
    patches, linearly projected, then enriched with:
      - a learnable lead embedding  (same for all 50 patches of one lead)
      - a static sinusoidal position embedding  (same across all 12 leads)
    Result: (B, 12 * n_patches, model_dim) = (B, 600, 256)
    """

    def __init__(self, n_leads: int, signal_len: int, patch_size: int, model_dim: int):
        super().__init__()
        assert signal_len % patch_size == 0, \
            f"signal_len {signal_len} must be divisible by patch_size {patch_size}"
        self.n_leads    = n_leads
        self.n_patches  = signal_len // patch_size  # 50
        self.patch_size = patch_size

        self.proj      = nn.Linear(patch_size, model_dim)
        self.lead_emb  = nn.Embedding(n_leads, model_dim)
        # Static position embedding registered as a buffer (not updated by optimizer)
        self.register_buffer("pos_emb", _sinusoidal_pos_emb(self.n_patches, model_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, 12, 1000)
        B = x.shape[0]

        # Reshape into patches: (B, 12, 50, patch_size)
        tokens = x.reshape(B, self.n_leads, self.n_patches, self.patch_size)
        tokens = self.proj(tokens)  # (B, 12, 50, model_dim)

        # Lead embedding: (12, model_dim) → (1, 12, 1, model_dim)
        lead_ids = torch.arange(self.n_leads, device=x.device)
        tokens   = tokens + self.lead_emb(lead_ids).view(1, self.n_leads, 1, -1)

        # Position embedding: (50, model_dim) → (1, 1, 50, model_dim)
        tokens = tokens + self.pos_emb.view(1, 1, self.n_patches, -1)

        # Flatten leads × patches into a single sequence: (B, 600, model_dim)
        return tokens.reshape(B, self.n_leads * self.n_patches, -1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """FiLM-style feature modulation: scale and shift every token."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block (BERT/GPT-style).

    Ordering: LN → MHA → residual → LN → FFN → residual
    Pre-norm is more stable than post-norm for deep networks.
    """

    def __init__(self, model_dim: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff    = nn.Sequential(
            nn.Linear(model_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, model_dim),
            nn.Dropout(dropout),
        )
        self.adaLN = nn.Linear(2 * model_dim, 4 * model_dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift1, scale1, shift2, scale2 = self.adaLN(cond).chunk(4, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(modulate(self.norm2(x), shift2, scale2))
        return x


class ECGTransformerDiffusion(nn.Module):
    """
    Denoising network: (x_t, t, class_label) → predicted noise ε.

    Input:  x_t (B, 12, 1000), t (B,) ints, class_label (B,) ints
    Output: ε   (B, 12, 1000)
    """

    def __init__(self, cfg, n_classes: int):
        super().__init__()
        d         = cfg.diffusion
        model_dim = int(d.model_dim)
        n_leads   = 12
        sig_len   = int(cfg.ptbxl.signal_length)
        patch_sz  = int(d.patch_size)

        # ── Patchification ───────────────────────────────────────────────────
        self.patch_embed = PatchEmbed1D(
            n_leads=n_leads,
            signal_len=sig_len,
            patch_size=patch_sz,
            model_dim=model_dim,
        )

        # ── Condition embeddings ──────────────────────────────────────────────
        # Time: sinusoidal → 2-layer MLP → (B, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        # Class: embedding table — n_classes real classes + 1 null/unconditional token
        # null_class_index = n_classes (the last row) is used during CFG training and sampling.
        self.null_class_index = n_classes
        self.class_emb = nn.Embedding(n_classes + 1, model_dim)

        # ── Transformer encoder (pre-norm BERT-style) ─────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                model_dim=model_dim,
                n_heads=int(d.n_heads),
                d_ff=int(d.d_ff),
                dropout=float(d.dropout),
            )
            for _ in range(int(d.n_transformer_layers))
        ])
        self.final_norm = nn.LayerNorm(model_dim)

        # ── Unpatchify ───────────────────────────────────────────────────────
        self.unproj    = nn.Linear(model_dim, patch_sz)
        self.n_leads   = n_leads
        self.n_patches = sig_len // patch_sz   # 50
        self.patch_size = patch_sz
        self._dim      = model_dim

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                if m is self.class_emb:
                    nn.init.normal_(m.weight, std=1.0)  # was 0.02 — see commit message
                else:
                    nn.init.normal_(m.weight, std=0.02)
        # Zero-init unproj so predictions start near zero
        nn.init.zeros_(self.unproj.weight)
        nn.init.zeros_(self.unproj.bias)
        # Zero-init adaLN in every TransformerBlock — must run AFTER the general
        # nn.Linear loop above, which would otherwise overwrite these with xavier_uniform_
        for block in self.blocks:
            nn.init.zeros_(block.adaLN.weight)
            nn.init.zeros_(block.adaLN.bias)

    def forward(self, x_t: Tensor, t: Tensor, class_label: Tensor) -> Tensor:
        B = x_t.shape[0]

        # Patchify: (B, 12, 1000) → (B, 600, model_dim)
        tokens = self.patch_embed(x_t)

        # Build condition: (B, model_dim)
        t_emb = _sinusoidal_time_emb(t, self._dim)
        t_emb = self.time_mlp(t_emb)
        c_emb     = self.class_emb(class_label)
        cond      = t_emb + c_emb                          # (B, model_dim)  — token injection only
        cond_film = torch.cat([t_emb, c_emb], dim=-1)     # (B, 2*model_dim) — adaLN input only

        # Broadcast summed condition to every token (unchanged from prior PR)
        tokens = tokens + cond.unsqueeze(1)  # (B, 600, model_dim)

        # Transformer blocks — cond_film (decoupled concat) feeds adaLN; cond is NOT used here
        for block in self.blocks:
            tokens = block(tokens, cond_film)
        tokens = self.final_norm(tokens)

        # Unpatchify: (B, 600, model_dim) → (B, 12, 1000)
        out = self.unproj(tokens)                                            # (B, 600, patch_size)
        out = out.reshape(B, self.n_leads, self.n_patches, self.patch_size)  # (B, 12, 50, 20)
        out = out.reshape(B, self.n_leads, -1)                               # (B, 12, 1000)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian diffusion process
# ──────────────────────────────────────────────────────────────────────────────

class GaussianDiffusion:
    """
    Precomputed DDPM noise schedule and DDIM sampler.

    Schedule tensors live on `device` and are indexed by t ∈ [0, T-1].
    """

    def __init__(self, T: int, beta_schedule: str = "cosine", device: str = "cpu"):
        self.T = T

        if beta_schedule == "cosine":
            betas = _cosine_betas(T)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule!r}")

        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        def _to(x: Tensor) -> Tensor:
            return x.float().to(device)

        self.betas                    = _to(betas)
        self.alphas                   = _to(alphas)
        self.alpha_bar                = _to(alpha_bar)
        self.sqrt_alpha_bar           = _to(alpha_bar.sqrt())
        self.sqrt_one_minus_alpha_bar = _to((1.0 - alpha_bar).sqrt())
        self.device                   = device

    def to(self, device: str) -> "GaussianDiffusion":
        for attr in ("betas", "alphas", "alpha_bar",
                     "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar"):
            setattr(self, attr, getattr(self, attr).to(device))
        self.device = device
        return self

    # ── Forward process ───────────────────────────────────────────────────────

    def q_sample(
        self,
        x0:    Tensor,
        t:     Tensor,
        noise: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample x_t | x_0 via reparameterisation: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε."""
        if noise is None:
            noise = torch.randn_like(x0)
        sab  = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        smab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sab * x0 + smab * noise, noise

    # ── DDIM sampling ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model:          nn.Module,
        shape:          tuple,
        class_label:    Tensor,
        n_steps:        int            = 50,
        clip_x0:        float          = 4.0,
        guidance_scale: Optional[float] = None,
    ) -> Tensor:
        """
        DDIM reverse process (η=0, fully deterministic).

        Uniformly-spaced timesteps from T-1 → 0. Clamps x̂_0 estimate to
        [-clip_x0, clip_x0] matching the preprocessing z-score clip range.

        If guidance_scale is not None, runs CFG: a single batched forward
        pass with [real_labels, null_labels] concatenated along dim 0 produces
        eps_cond and eps_uncond in one call, then combines them as:
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        If guidance_scale is None, runs the original single-pass behavior —
        zero change for callers that don't pass it.
        """
        device = class_label.device
        B      = shape[0]

        ts = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=device)
        x  = torch.randn(shape, device=device)

        for i, t_curr in enumerate(ts):
            t_batch = t_curr.expand(B)

            if guidance_scale is not None:
                # Batched two-pass CFG: concat real + null along batch dim,
                # single forward call, then split. Avoids two sequential calls
                # and halves Python overhead vs. two model() invocations.
                null_label = torch.full_like(class_label, model.null_class_index)
                x_double   = torch.cat([x, x], dim=0)
                t_double   = torch.cat([t_batch, t_batch], dim=0)
                lbl_double = torch.cat([class_label, null_label], dim=0)
                eps_double = model(x_double, t_double, lbl_double)
                eps_cond, eps_uncond = eps_double[:B], eps_double[B:]
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = model(x, t_batch, class_label)

            ab_t = self.alpha_bar[t_curr]

            # x_0 estimate
            x0_hat = (x - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-8)
            x0_hat = x0_hat.clamp(-clip_x0, clip_x0)

            # Previous alpha_bar (= 1 at the last step, giving x = x0_hat cleanly)
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < n_steps else torch.ones(1, device=device)

            x = ab_prev.sqrt() * x0_hat + (1.0 - ab_prev).sqrt() * eps

        return x


# ──────────────────────────────────────────────────────────────────────────────
# Exponential moving average
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Shadow-copy EMA of model parameters for stable generation quality."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow: dict[str, Tensor] = {
            name: param.data.clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.float(), alpha=1.0 - self.decay
                )

    @contextmanager
    def ema_scope(self, model: nn.Module):
        """Temporarily swap live weights → EMA weights, then restore."""
        live = {n: p.data.clone() for n, p in model.named_parameters() if n in self.shadow}
        try:
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].to(param.dtype))
            yield
        finally:
            for name, param in model.named_parameters():
                if name in live:
                    param.data.copy_(live[name])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset and label loading
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
    log,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map each record_id → integer class index using class_mapping.json.

    Uses the highest-confidence SCP code that maps to a final class. Ties
    at the maximum confidence are broken by
    utils.label_assignment.TIE_BREAK_PRIORITY (a clinical-severity
    ordering, not dict-iteration order — see
    Roadmap/Stage_0_Pipeline_Audit/Reports/Pipeline_Code_Audit.md
    Finding 5), via the same shared function step03's _assign_primary()
    uses, so the two selection rules cannot silently diverge.
    Records with no recognisable code are assigned to OTHER (if present) or dropped.

    Returns:
        valid_indices — indices into record_ids that were successfully mapped
        class_idxs   — integer class index for each valid record
    """
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    valid:  list[int] = []
    labels: list[int] = []

    for i, eid in enumerate(record_ids):
        eid_int = int(eid)
        if eid_int not in ptbxl_db.index:
            continue

        scp = _parse_scp(ptbxl_db.at[eid_int, "scp_codes"])
        if not scp:
            continue

        best_cls = assign_primary_class(scp, class_mapping)

        if best_cls is None or best_cls not in name_to_idx:
            if "OTHER" in name_to_idx:
                best_cls = "OTHER"
            else:
                continue

        valid.append(i)
        labels.append(name_to_idx[best_cls])

    return np.array(valid, dtype=np.int64), np.array(labels, dtype=np.int64)


class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, labels: np.ndarray):
        # X: (N, 1000, 12) → transpose to (N, 12, 1000) as model expects
        self.X      = torch.from_numpy(X.transpose(0, 2, 1)).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.X[idx], self.labels[idx]


def _make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Inverse-frequency sampling so rare classes are seen as often as frequent ones."""
    counts  = Counter(labels.tolist())
    weights = np.array([1.0 / counts[int(l)] for l in labels], dtype=np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public generation API
# ──────────────────────────────────────────────────────────────────────────────

def generate_ecg(
    model:          nn.Module,
    diffusion:      GaussianDiffusion,
    class_label:    int,
    n_samples:      int            = 10,
    device:         str            = "cuda",
    cfg=None,
    seed:           int            = 42,
    stats:          Optional[dict] = None,
    guidance_scale: Optional[float] = None,
) -> np.ndarray:
    """
    Generate n_samples ECGs for the given class label using DDIM sampling.

    Args:
        model:          trained ECGTransformerDiffusion (will be set to eval mode)
        diffusion:      GaussianDiffusion with schedule tensors already on device
        class_label:    integer class index matching class_names.json ordering
        n_samples:      number of ECGs to generate
        device:         "cuda" or "cpu"
        cfg:            OmegaConf config (for signal shape / ddim_steps)
        seed:           RNG seed for reproducibility
        stats:          preprocessing_stats dict {'mean': [...], 'std': [...]}
                        for denormalisation to original mV scale; if None,
                        returns output in z-score space (clipped to ±4)
        guidance_scale: CFG scale (e.g. 3.0). None = old single-pass behavior.

    Returns:
        np.ndarray of shape (n_samples, 1000, 12)
    """
    torch.manual_seed(seed)
    sig_len    = int(cfg.ptbxl.signal_length) if cfg is not None else 1000
    ddim_steps = int(cfg.diffusion.ddim_steps) if cfg is not None else 50
    n_leads    = 12

    model.eval()
    label_t = torch.full((n_samples,), fill_value=class_label, dtype=torch.long, device=device)

    with torch.no_grad():
        samples = diffusion.ddim_sample(
            model=model,
            shape=(n_samples, n_leads, sig_len),
            class_label=label_t,
            n_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )  # (n_samples, 12, 1000)

    # Transpose to (N, 1000, 12) for downstream consistency
    out = samples.cpu().float().numpy().transpose(0, 2, 1)

    if stats is not None:
        mu    = np.array(stats["per_lead_mean"], dtype=np.float32)   # (12,)
        sigma = np.array(stats["per_lead_std"],  dtype=np.float32)   # (12,)
        out   = out * sigma[np.newaxis, np.newaxis, :] + mu[np.newaxis, np.newaxis, :]

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Validation plotting
# ──────────────────────────────────────────────────────────────────────────────

def _validation_plot(
    model:       nn.Module,
    diffusion:   GaussianDiffusion,
    ema:         EMA,
    X_val:       np.ndarray,
    val_labels:  np.ndarray,
    class_names: list[str],
    cfg,
    epoch:       int,
    results_dir: Path,
    device:      str,
) -> float:
    """
    Generate 5 samples per class and plot 2 generated vs 2 real ECGs (Lead II).
    Returns average nearest-neighbour MSE (generated vs val set, in z-score space).
    """
    sig_len   = int(cfg.ptbxl.signal_length)
    fs        = float(cfg.ptbxl.sampling_rate)
    lead_idx  = 1   # Lead II
    n_gen     = 5
    n_cols    = min(len(class_names), 4)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 6), constrained_layout=True)
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    nn_mse_all: list[float] = []
    t_axis = np.arange(sig_len) / fs

    with ema.ema_scope(model):
        model.eval()
        for col, cls_name in enumerate(class_names[:n_cols]):
            cls_idx = class_names.index(cls_name)
            label_t = torch.full((n_gen,), cls_idx, dtype=torch.long, device=device)

            with torch.no_grad():
                gen = diffusion.ddim_sample(
                    model, (n_gen, 12, sig_len), label_t,
                    n_steps=int(cfg.diffusion.ddim_steps),
                )  # (5, 12, 1000)
            gen_np = gen.cpu().numpy()  # (5, 12, 1000)

            # Nearest-neighbour MSE for this class in the val set
            val_mask = (val_labels == cls_idx)
            if val_mask.sum() > 0:
                X_cls = X_val[val_mask][:, :, lead_idx]  # (M, 1000)
                for g_lead in gen_np[:, lead_idx, :]:
                    nn_mse_all.append(float(np.min(np.mean((X_cls - g_lead) ** 2, axis=1))))

            # Row 0: 2 generated examples
            ax_gen = axes[0, col]
            for k in range(min(2, n_gen)):
                ax_gen.plot(t_axis, gen_np[k, lead_idx, :], alpha=0.85,
                            linewidth=0.8, color=f"C{k}")
            ax_gen.set_title(f"{cls_name}\n(generated)", fontsize=9)
            ax_gen.set_ylabel("Lead II (z)", fontsize=8)
            ax_gen.spines["top"].set_visible(False)
            ax_gen.spines["right"].set_visible(False)
            ax_gen.set_xticks([])

            # Row 1: 2 real examples
            ax_real = axes[1, col]
            if val_mask.sum() > 0:
                real_idx = np.where(val_mask)[0]
                for k in range(min(2, len(real_idx))):
                    ax_real.plot(t_axis, X_val[real_idx[k], :, lead_idx],
                                 alpha=0.85, linewidth=0.8, color=f"C{k}")
            ax_real.set_title("(real)", fontsize=9)
            ax_real.set_xlabel("Time (s)", fontsize=8)
            ax_real.set_ylabel("Lead II (z)", fontsize=8)
            ax_real.spines["top"].set_visible(False)
            ax_real.spines["right"].set_visible(False)

    avg_nn_mse = float(np.mean(nn_mse_all)) if nn_mse_all else float("nan")
    fig.suptitle(
        f"Epoch {epoch:04d} — Lead II: generated vs real  |  "
        f"avg NN-MSE = {avg_nn_mse:.4f}",
        fontsize=10,
    )
    out_path = results_dir / f"diffusion_val_ep{epoch:04d}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return avg_nn_mse


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_device(cfg) -> str:
    if str(cfg.device) != "auto":
        return str(cfg.device)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(cfg, log) -> float:
    device = _resolve_device(cfg)
    log.info(f"Device: {device}")

    # ── Paths ─────────────────────────────────────────────────────────────────
    processed_dir = Path(cfg.paths.outputs.processed)
    models_dir    = Path(cfg.paths.outputs.models)
    results_dir   = Path(cfg.paths.outputs.results)
    logs_dir      = Path(cfg.paths.logs)
    generated_dir = Path(cfg.paths.outputs.generated) / "baseline_samples"
    for d_path in (models_dir, results_dir, logs_dir, generated_dir):
        d_path.mkdir(parents=True, exist_ok=True)

    # ── Class info ────────────────────────────────────────────────────────────
    cls_names_path = processed_dir / "class_names.json"
    cls_map_path   = processed_dir / "class_mapping.json"

    if cls_names_path.exists() and cls_map_path.exists():
        with open(cls_names_path)   as f: class_names   = json.load(f)
        with open(cls_map_path) as f: class_mapping = json.load(f)
        log.info(f"Loaded class_names.json: {class_names}")
    else:
        log.warning(
            "class_names.json / class_mapping.json not found — "
            "falling back to config.ptbxl.classes. Run step03 first for best results."
        )
        class_names   = list(cfg.ptbxl.classes)
        class_mapping = {c: c for c in class_names}

    n_classes = len(class_names)

    # ── Load signals ──────────────────────────────────────────────────────────
    for p in (
        processed_dir / "X_train.npy", processed_dir / "X_val.npy",
        processed_dir / "record_ids_train.npy", processed_dir / "record_ids_val.npy",
    ):
        if not p.exists():
            log.error(f"Missing: {p}. Run step02_preprocessing.py first.")
            raise FileNotFoundError(p)

    log.info("Loading signal arrays …")
    X_train       = np.load(str(processed_dir / "X_train.npy"))   # (N, 1000, 12)
    X_val         = np.load(str(processed_dir / "X_val.npy"))
    rec_ids_train = np.load(str(processed_dir / "record_ids_train.npy"))
    rec_ids_val   = np.load(str(processed_dir / "record_ids_val.npy"))
    log.info(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

    # ── Map record IDs → class labels ─────────────────────────────────────────
    db_path = Path(cfg.paths.data.ptbxl) / "ptbxl_database.csv"
    if not db_path.exists():
        log.error(f"ptbxl_database.csv not found at {db_path}. Run step01 first.")
        raise FileNotFoundError(db_path)

    log.info("Assigning class labels from PTB-XL metadata …")
    ptbxl_db = pd.read_csv(str(db_path), index_col="ecg_id")

    vi_train, train_labels = _load_class_labels(rec_ids_train, ptbxl_db, class_mapping, class_names, log)
    vi_val,   val_labels   = _load_class_labels(rec_ids_val,   ptbxl_db, class_mapping, class_names, log)
    X_train, X_val = X_train[vi_train], X_val[vi_val]
    log.info(f"After mapping — train: {len(X_train):,}  val: {len(X_val):,}")
    log.info(f"Train distribution: {dict(Counter(class_names[i] for i in train_labels.tolist()))}")

    # ── Optional preprocessing stats for denorm ────────────────────────────────
    stats_path = processed_dir / "preprocessing_stats.json"
    prep_stats: Optional[dict] = None
    if stats_path.exists():
        with open(stats_path) as f:
            prep_stats = json.load(f)

    # ── Datasets / DataLoaders ─────────────────────────────────────────────────
    train_ds = ECGDataset(X_train, train_labels)
    val_ds   = ECGDataset(X_val,   val_labels)
    sampler  = _make_weighted_sampler(train_labels)

    d = cfg.diffusion
    train_loader = DataLoader(
        train_ds, batch_size=int(d.batch_size), sampler=sampler,
        num_workers=0, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(d.batch_size), shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model    = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params / 1e6:.2f}M")
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    diffusion = GaussianDiffusion(T=int(d.T), beta_schedule=str(d.beta_schedule), device=device)
    ema       = EMA(model, decay=float(d.ema_decay))

    # ── CFG training hyperparameters ──────────────────────────────────────────
    p_uncond = float(getattr(d, "p_uncond", 0.10))
    log.info(f"CFG training: p_uncond={p_uncond}")

    # ── Optimiser and scheduler ───────────────────────────────────────────────
    _lr = float(d.lr)
    _wd = float(d.weight_decay)
    _expected_emb_shape = (n_classes + 1, int(d.model_dim))  # (7, 256) after null-class resize
    _class_emb_params = {n for n, _ in model.named_parameters() if n == "class_emb.weight"}
    assert _class_emb_params == {"class_emb.weight"}, f"Unexpected class_emb param names: {_class_emb_params}"
    _decay_params   = [p for n, p in model.named_parameters() if n != "class_emb.weight"]
    _nodecay_params = [p for n, p in model.named_parameters() if n == "class_emb.weight"]
    assert (len(_nodecay_params) == 1
            and tuple(_nodecay_params[0].shape) == _expected_emb_shape), (
        f"class_emb shape mismatch: expected {_expected_emb_shape}, "
        f"got {tuple(_nodecay_params[0].shape) if _nodecay_params else 'no tensor'}"
    )
    log.info(f"Optimizer groups: decay({len(_decay_params)} params, wd={_wd}), "
             f"no-decay({len(_nodecay_params)} params [{model.class_emb.weight.shape}], wd=0.0)")
    optimiser = torch.optim.AdamW(
        [
            {"params": _decay_params,   "weight_decay": _wd},
            {"params": _nodecay_params, "weight_decay": 0.0},
        ],
        lr=_lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=int(d.n_epochs))
    use_amp   = (device == "cuda")
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path  = logs_dir / "diffusion_training_log.csv"
    log_fh    = open(log_path, "w", newline="")
    writer    = csv.writer(log_fh)
    writer.writerow(["epoch", "step", "train_loss", "val_loss", "lr", "gpu_mem_gb"])
    log_fh.flush()

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    global_step   = 0
    n_epochs      = int(d.n_epochs)
    save_every    = int(d.save_every)
    log_interval  = int(cfg.logging.log_interval)

    log.info(f"Training: {n_epochs} epochs × {len(train_loader)} steps")

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0         = time.time()

        for batch_x, batch_cls in train_loader:
            batch_x   = batch_x.to(device)    # (B, 12, 1000)
            batch_cls = batch_cls.to(device)   # (B,)
            B         = batch_x.shape[0]

            # Per-sample CFG dropout: each sample independently replaced with
            # null_class_index with probability p_uncond. Per-sample (not per-batch)
            # so the model sees both conditional and unconditional in every batch.
            if p_uncond > 0.0:
                null_mask = torch.bernoulli(torch.full((B,), p_uncond, device=device)).bool()
                batch_cls = batch_cls.clone()
                batch_cls[null_mask] = model.null_class_index

            t_diff = torch.randint(0, int(d.T), (B,), device=device)
            noise  = torch.randn_like(batch_x)
            x_t, _ = diffusion.q_sample(batch_x, t_diff, noise)

            with torch.cuda.amp.autocast(enabled=use_amp):
                eps_pred = model(x_t, t_diff, batch_cls)
                loss     = F.mse_loss(eps_pred, noise)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(d.grad_clip))
            scaler.step(optimiser)
            scaler.update()
            ema.update(model)

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % log_interval == 0:
                gpu_mem = torch.cuda.memory_allocated(device) / 1e9 if device == "cuda" else 0.0
                writer.writerow([
                    epoch, global_step, f"{loss.item():.6f}", "",
                    f"{scheduler.get_last_lr()[0]:.2e}", f"{gpu_mem:.3f}",
                ])
                log_fh.flush()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)

        # ── Validation block ──────────────────────────────────────────────────
        if epoch % save_every == 0:
            model.eval()
            val_total, val_steps = 0.0, 0
            with ema.ema_scope(model), torch.no_grad():
                for batch_x, batch_cls in val_loader:
                    batch_x, batch_cls = batch_x.to(device), batch_cls.to(device)
                    # Validation always uses real labels — no CFG dropout here (intentional).
                    B      = batch_x.shape[0]
                    t_diff = torch.randint(0, int(d.T), (B,), device=device)
                    noise  = torch.randn_like(batch_x)
                    x_t, _ = diffusion.q_sample(batch_x, t_diff, noise)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        eps_pred = model(x_t, t_diff, batch_cls)
                        v_loss   = F.mse_loss(eps_pred, noise)
                    val_total += v_loss.item()
                    val_steps += 1
                    if val_steps >= 100:
                        break

            val_loss = val_total / max(val_steps, 1)

            nn_mse = _validation_plot(
                model, diffusion, ema, X_val, val_labels,
                class_names, cfg, epoch, results_dir, device,
            )

            log.info(
                f"Epoch {epoch:04d}/{n_epochs} | "
                f"train={avg_train:.5f} | val={val_loss:.5f} | "
                f"nn_mse={nn_mse:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"elapsed={time.time() - t0:.1f}s"
            )

            ckpt = {
                "epoch":       epoch,
                "model":       model.state_dict(),
                "ema_shadow":  ema.shadow,
                "optimiser":   optimiser.state_dict(),
                "val_loss":    val_loss,
                "class_names": class_names,
                "n_classes":   n_classes,
            }
            torch.save(ckpt, str(models_dir / f"diffusion_ckpt_ep{epoch:04d}.pt"))
            log.info(f"Checkpoint → diffusion_ckpt_ep{epoch:04d}.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(ckpt, str(models_dir / "diffusion_best.pt"))
                log.info(f"★ Best val loss: {best_val_loss:.5f}")

            writer.writerow([
                epoch, global_step, f"{avg_train:.6f}", f"{val_loss:.6f}",
                f"{scheduler.get_last_lr()[0]:.2e}", "",
            ])
            log_fh.flush()

        elif epoch % 10 == 0:
            log.info(
                f"Epoch {epoch:04d}/{n_epochs} | train={avg_train:.5f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | elapsed={time.time() - t0:.1f}s"
            )

    log_fh.close()

    # ── Architecture JSON ─────────────────────────────────────────────────────
    arch = {
        "model_class":          "ECGTransformerDiffusion",
        "model_dim":            int(d.model_dim),
        "patch_size":           int(d.patch_size),
        "n_leads":              12,
        "n_patches_per_lead":   12 * (int(cfg.ptbxl.signal_length) // int(d.patch_size)),
        "n_tokens":             12 * (int(cfg.ptbxl.signal_length) // int(d.patch_size)),
        "n_heads":              int(d.n_heads),
        "n_transformer_layers": int(d.n_transformer_layers),
        "d_ff":                 int(d.d_ff),
        "dropout":              float(d.dropout),
        "T":                    int(d.T),
        "beta_schedule":        str(d.beta_schedule),
        "ddim_steps":           int(d.ddim_steps),
        "n_classes":            n_classes,
        "class_names":          class_names,
        "n_params":             n_params,
        "best_val_loss":        best_val_loss,
    }
    with open(models_dir / "diffusion_architecture.json", "w") as f:
        json.dump(arch, f, indent=2)
    log.info("Saved diffusion_architecture.json")

    return best_val_loss


# ──────────────────────────────────────────────────────────────────────────────
# Post-training sample generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_final_samples(cfg, log) -> None:
    """Load best checkpoint and generate 20 samples per class."""
    device        = _resolve_device(cfg)
    processed_dir = Path(cfg.paths.outputs.processed)
    models_dir    = Path(cfg.paths.outputs.models)
    generated_dir = Path(cfg.paths.outputs.generated) / "baseline_samples"
    generated_dir.mkdir(parents=True, exist_ok=True)

    best_path = models_dir / "diffusion_best.pt"
    if not best_path.exists():
        log.warning("diffusion_best.pt not found — skipping final sample generation.")
        return

    log.info("Loading best checkpoint for sample generation …")
    ckpt        = torch.load(str(best_path), map_location=device)
    class_names = ckpt["class_names"]
    n_classes   = ckpt["n_classes"]

    model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model"])

    ema = EMA(model, decay=float(cfg.diffusion.ema_decay))
    ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}

    diffusion = GaussianDiffusion(
        T=int(cfg.diffusion.T), beta_schedule=str(cfg.diffusion.beta_schedule), device=device
    )

    stats_path = processed_dir / "preprocessing_stats.json"
    prep_stats = None
    if stats_path.exists():
        with open(stats_path) as f:
            prep_stats = json.load(f)

    n_per_class = 20
    with ema.ema_scope(model):
        for cls_idx, cls_name in enumerate(class_names):
            log.info(f"Generating {n_per_class} × {cls_name} …")
            samples = generate_ecg(
                model, diffusion,
                class_label=cls_idx,
                n_samples=n_per_class,
                device=device,
                cfg=cfg,
                seed=42 + cls_idx,
                stats=prep_stats,
            )  # (20, 1000, 12)
            for i, sample in enumerate(samples):
                np.save(str(generated_dir / f"class_{cls_name}_sample_{i:04d}.npy"), sample)

    log.info(f"Saved {n_per_class} × {len(class_names)} samples → {generated_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    log = get_logger("step04_transformer_diffusion", cfg=cfg)
    set_seed(cfg.seeds[0])
    snapshot_before_write(Path(cfg.paths.outputs.models))

    best_val_loss = train(cfg, log)
    _generate_final_samples(cfg, log)

    log.info("=" * 60)
    print(f"✓ Diffusion training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
