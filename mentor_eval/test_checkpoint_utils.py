"""
mentor_eval/test_checkpoint_utils.py -- regression test for load_checkpoint()
correctly detecting adaln_cross_attn checkpoints from their state_dict keys
instead of always falling through to the AdaLN-only Stage 3 variant path.

Run:
    pytest mentor_eval/test_checkpoint_utils.py -v
"""

from __future__ import annotations

from pathlib import Path

import torch

from mentor_eval.checkpoint_utils import load_checkpoint
from step04_transformer_diffusion import ECGTransformerDiffusion, EMA
from utils import load_config


def _save_fake_checkpoint(tmp_path: Path, conditioning: str, n_classes: int,
                           class_names: list[str]) -> Path:
    cfg = load_config()
    cfg.diffusion.conditioning = conditioning
    model = ECGTransformerDiffusion(cfg, n_classes=n_classes)
    ema = EMA(model, decay=float(cfg.diffusion.ema_decay))
    ckpt = {
        "epoch": 1,
        "model": model.state_dict(),
        "ema_shadow": ema.shadow,
        "optimiser": {},
        "val_loss": 0.0,
        "class_names": class_names,
        "n_classes": n_classes,
    }
    path = tmp_path / f"fake_{conditioning}_ckpt.pt"
    torch.save(ckpt, str(path))
    return path


def test_cross_attn_checkpoint_loads_via_cross_attn_path(tmp_path):
    """A checkpoint with disease_token_emb/cross_attn/cross_gate keys must be
    reconstructed as ECGTransformerDiffusion(conditioning='adaln_cross_attn'),
    not the AdaLN-only Stage 3 variant wrapper -- this is the bug that made
    strict=True state_dict loading fail with 'Unexpected key(s)' for real
    adaln_cross_attn training runs."""
    ckpt_path = _save_fake_checkpoint(tmp_path, "adaln_cross_attn", n_classes=2,
                                       class_names=["NORM", "MI"])
    loaded = load_checkpoint(ckpt_path, load_config())

    assert loaded is not None
    assert hasattr(loaded.model, "disease_token_emb")
    assert loaded.n_classes == 2
    assert loaded.class_names == ["NORM", "MI"]


def test_plain_adaln_checkpoint_unaffected(tmp_path):
    """Regression guard: plain adaln checkpoints (the common case) must
    keep using the existing build_variant_model path, unchanged."""
    ckpt_path = _save_fake_checkpoint(tmp_path, "adaln", n_classes=6,
                                       class_names=["NORM", "MI", "STTC", "CD", "HYP", "OTHER"])
    loaded = load_checkpoint(ckpt_path, load_config())

    assert loaded is not None
    assert not hasattr(loaded.model, "disease_token_emb")
    assert loaded.n_classes == 6
