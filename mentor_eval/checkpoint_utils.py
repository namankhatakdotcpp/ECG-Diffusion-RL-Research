"""
mentor_eval/checkpoint_utils.py — shared diffusion-checkpoint loading.

Thin wrapper around step04_transformer_diffusion's model classes so the
mentor_eval scripts don't duplicate model-construction logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step04_transformer_diffusion import (
    ECGTransformerDiffusion, GaussianDiffusion, EMA, generate_ecg, _resolve_device,
)


class LoadedCheckpoint:
    def __init__(self, model, diffusion, ema, class_names, n_classes, epoch, val_loss, device):
        self.model = model
        self.diffusion = diffusion
        self.ema = ema
        self.class_names = class_names
        self.n_classes = n_classes
        self.epoch = epoch
        self.val_loss = val_loss
        self.device = device


def load_checkpoint(ckpt_path: Path, cfg) -> Optional[LoadedCheckpoint]:
    """Load a diffusion_*.pt checkpoint. Returns None if the file doesn't exist
    (callers must handle this and flag it rather than fabricating output).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None

    device = _resolve_device(cfg)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    class_names = ckpt["class_names"]
    n_classes = ckpt["n_classes"]

    model = ECGTransformerDiffusion(cfg, n_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ema = EMA(model, decay=float(cfg.diffusion.ema_decay))
    ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}

    diffusion = GaussianDiffusion(
        T=int(cfg.diffusion.T), beta_schedule=str(cfg.diffusion.beta_schedule), device=device,
    )

    return LoadedCheckpoint(
        model=model, diffusion=diffusion, ema=ema,
        class_names=class_names, n_classes=n_classes,
        epoch=ckpt.get("epoch"), val_loss=ckpt.get("val_loss"),
        device=device,
    )


def generate_for_class(
    loaded: LoadedCheckpoint, class_name: str, n_samples: int, cfg, seed: int,
    stats: Optional[dict] = None, use_ema: bool = False,
    guidance_scale: Optional[float] = None,
    # use_ema defaulted to False: diagnosed 2026-06-25 that EMA shadow weights are
    # severely under-trained relative to live model weights (unproj.weight
    # std 0.0043 vs 0.024 live) — sampling with EMA produced pure noise
    # across all classes/leads. Revisit if EMA tracking/update frequency
    # is fixed in training.
):
    """Generate n_samples ECGs for class_name using the model's live (or EMA) weights.

    Returns (samples, error_message). If class_name isn't in the trained
    model's class_names, returns (None, "<reason>") instead of guessing.

    guidance_scale: CFG scale (e.g. 3.0). None = original single-pass behavior.
    """
    if class_name not in loaded.class_names:
        return None, (
            f"'{class_name}' is not one of the trained model's classes "
            f"{loaded.class_names} — cannot generate this class."
        )
    class_idx = loaded.class_names.index(class_name)

    _kwargs = dict(
        model=loaded.model, diffusion=loaded.diffusion, class_label=class_idx,
        n_samples=n_samples, device=loaded.device, cfg=cfg, seed=seed, stats=stats,
        guidance_scale=guidance_scale,
    )
    if use_ema:
        with loaded.ema.ema_scope(loaded.model):
            samples = generate_ecg(**_kwargs)
    else:
        samples = generate_ecg(**_kwargs)
    return samples, None
