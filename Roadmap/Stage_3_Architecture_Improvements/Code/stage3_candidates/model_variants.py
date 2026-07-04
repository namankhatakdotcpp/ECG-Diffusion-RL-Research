"""
Stage 3 / Phase 1 -- candidate architecture variants (S3-001..S3-005).

Motivated by Stage 2's verified findings and the Phase 0 addendum to
`Stage2_Decision_Report.md`: conditioning's proportional influence
declines across blocks 1-6 (dilution, Task 0.1 -- High confidence),
and `final_norm`/`unproj` disproportionately suppresses the
conditioning-specific component relative to the whole-tensor signal
(Task 0.2 -- borderline, moderate confidence). Both are present
simultaneously, so the gain-focused candidates below (S3-002..S3-005)
are not exclusive alternatives to a `final_norm`/`unproj` fix -- they
are Phase 1's first five candidates; a 6th (`final_norm`/`unproj`
modification) is tracked separately per the Decision Report addendum
and is lower initial priority given Task 0.2's borderline margin.

Reuses `step04_transformer_diffusion.py`'s `PatchEmbed1D`, `modulate`,
`_sinusoidal_time_emb` unmodified (imported, not copied) -- only the
per-block residual-scaling mechanism and block-construction wiring
differ between variants. `TransformerBlock` (unmodified) is reused
directly for S3-001 (baseline) and for any block a variant does not
modify.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from step04_transformer_diffusion import (  # noqa: E402
    PatchEmbed1D,
    TransformerBlock,
    modulate,
    _sinusoidal_time_emb,
)


class TransformerBlockLayerScale(nn.Module):
    """S3-002 (LayerScale) building block: identical to TransformerBlock,
    with one learnable per-channel gain applied to each residual branch
    before it is added back -- `x = x + gamma1 * attn_out`, `x = x +
    gamma2 * ffn_out`. Initialized to 1.0 (identity at init, unlike the
    near-zero init used in very deep networks) since this architecture
    is only 6 blocks deep and the goal is to let training discover
    whether late-block contribution should grow, not stabilize a much
    deeper stack."""

    def __init__(self, model_dim: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, model_dim),
            nn.Dropout(dropout),
        )
        self.adaLN = nn.Linear(2 * model_dim, 4 * model_dim)
        self.gamma1 = nn.Parameter(torch.ones(model_dim))
        self.gamma2 = nn.Parameter(torch.ones(model_dim))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift1, scale1, shift2, scale2 = self.adaLN(cond).chunk(4, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        h, _ = self.attn(h, h, h)
        x = x + self.gamma1 * h
        x = x + self.gamma2 * self.ff(modulate(self.norm2(x), shift2, scale2))
        return x


class TransformerBlockResidualScale(nn.Module):
    """S3-004 (residual scaling) building block: identical to
    TransformerBlock, with one learnable SCALAR (not per-channel) gain
    per residual branch -- coarser than LayerScale, directly testing
    whether a single per-block scaling factor (matching Item 3's own
    block-level, not channel-level, granularity) is sufficient."""

    def __init__(self, model_dim: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, model_dim),
            nn.Dropout(dropout),
        )
        self.adaLN = nn.Linear(2 * model_dim, 4 * model_dim)
        self.gamma1 = nn.Parameter(torch.tensor(1.0))
        self.gamma2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift1, scale1, shift2, scale2 = self.adaLN(cond).chunk(4, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        h, _ = self.attn(h, h, h)
        x = x + self.gamma1 * h
        x = x + self.gamma2 * self.ff(modulate(self.norm2(x), shift2, scale2))
        return x


class TransformerBlockLayerScaleBoosted(TransformerBlockLayerScale):
    """S3-005 (hybrid) building block: TransformerBlockLayerScale plus
    one extra learnable scalar `boost` multiplying the block's TOTAL
    output (both residual adds combined) -- used only on blocks 5-6 in
    the hybrid variant, combining per-channel LayerScale (all blocks)
    with extra late-block emphasis (Items 1/3/5's blocks-5/6 finding),
    rather than either mechanism alone."""

    def __init__(self, model_dim: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__(model_dim, n_heads, d_ff, dropout)
        self.boost = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x_in = x
        x_out = super().forward(x, cond)
        return x_in + self.boost * (x_out - x_in)


class ECGTransformerDiffusionVariant(nn.Module):
    """Same architecture as ECGTransformerDiffusion
    (step04_transformer_diffusion.py:180-282), with the per-block class
    configurable so each Stage 3 candidate can swap in a different
    residual-scaling mechanism without duplicating patch-embedding,
    condition-embedding, or unpatchify logic. `block_classes` is a list
    of length `n_transformer_layers`, one class per block index (all
    must accept the same (model_dim, n_heads, d_ff, dropout)
    constructor signature as TransformerBlock)."""

    def __init__(self, cfg, n_classes: int, block_classes: list):
        super().__init__()
        d = cfg.diffusion
        model_dim = int(d.model_dim)
        n_leads = 12
        sig_len = int(cfg.ptbxl.signal_length)
        patch_sz = int(d.patch_size)
        n_layers = int(d.n_transformer_layers)
        assert len(block_classes) == n_layers, (
            f"block_classes must have exactly {n_layers} entries, got {len(block_classes)}"
        )

        self.patch_embed = PatchEmbed1D(
            n_leads=n_leads, signal_len=sig_len, patch_size=patch_sz, model_dim=model_dim,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim),
        )
        self.null_class_index = n_classes
        self.class_emb = nn.Embedding(n_classes + 1, model_dim)

        self.blocks = nn.ModuleList([
            cls(model_dim=model_dim, n_heads=int(d.n_heads), d_ff=int(d.d_ff), dropout=float(d.dropout))
            for cls in block_classes
        ])
        self.final_norm = nn.LayerNorm(model_dim)
        self.unproj = nn.Linear(model_dim, patch_sz)
        self.n_leads = n_leads
        self.n_patches = sig_len // patch_sz
        self.patch_size = patch_sz
        self._dim = model_dim

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                if m is self.class_emb:
                    nn.init.normal_(m.weight, std=1.0)
                else:
                    nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.unproj.weight)
        nn.init.zeros_(self.unproj.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN.weight)
            nn.init.zeros_(block.adaLN.bias)
        # gamma1/gamma2/boost (if present) are left at their class-level
        # init (1.0) -- zeroing adaLN already makes every block an
        # identity map at init (shift=scale=0 -> modulate is a no-op),
        # so these gains start as true no-ops regardless of their own
        # init value; only their GRADIENT behavior differs from 1.0.

    def forward(self, x_t: Tensor, t: Tensor, class_label: Tensor) -> Tensor:
        B = x_t.shape[0]
        tokens = self.patch_embed(x_t)
        t_emb = _sinusoidal_time_emb(t, self._dim)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_emb(class_label)
        cond = t_emb + c_emb
        cond_film = torch.cat([t_emb, c_emb], dim=-1)
        tokens = tokens + cond.unsqueeze(1)
        for block in self.blocks:
            tokens = block(tokens, cond_film)
        tokens = self.final_norm(tokens)
        out = self.unproj(tokens)
        out = out.reshape(B, self.n_leads, self.n_patches, self.patch_size)
        out = out.reshape(B, self.n_leads, -1)
        return out


def build_variant_model(cfg, n_classes: int, variant: str) -> ECGTransformerDiffusionVariant:
    """variant in {"baseline", "layerscale", "late_gain", "residual_scaling", "hybrid"}."""
    n_layers = int(cfg.diffusion.n_transformer_layers)

    if variant == "baseline":
        block_classes = [TransformerBlock] * n_layers
    elif variant == "layerscale":
        block_classes = [TransformerBlockLayerScale] * n_layers
    elif variant == "late_gain":
        # Blocks 5-6 (last two, 0-indexed n_layers-2, n_layers-1) get
        # LayerScale; blocks 1-4 stay as plain TransformerBlock.
        block_classes = [TransformerBlock] * (n_layers - 2) + [TransformerBlockLayerScale] * 2
    elif variant == "residual_scaling":
        block_classes = [TransformerBlockResidualScale] * n_layers
    elif variant == "hybrid":
        # LayerScale everywhere, PLUS extra boost on the last two blocks.
        block_classes = [TransformerBlockLayerScale] * (n_layers - 2) + [TransformerBlockLayerScaleBoosted] * 2
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return ECGTransformerDiffusionVariant(cfg, n_classes=n_classes, block_classes=block_classes)
