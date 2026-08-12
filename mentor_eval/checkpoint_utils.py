"""
mentor_eval/checkpoint_utils.py — shared diffusion-checkpoint loading.

Thin wrapper around step04_transformer_diffusion's model classes so the
mentor_eval scripts don't duplicate model-construction logic.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step04_transformer_diffusion import (
    ECGTransformerDiffusion, GaussianDiffusion, EMA, generate_ecg, _resolve_device,
)

STAGE3_CANDIDATES_DIR = (
    Path(__file__).resolve().parents[1]
    / "Roadmap" / "Stage_3_Architecture_Improvements" / "Code" / "stage3_candidates"
)
if str(STAGE3_CANDIDATES_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_CANDIDATES_DIR))
from model_variants import build_variant_model  # noqa: E402


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
    variant = ckpt.get("variant", "baseline")

    # step04_transformer_diffusion.py never writes a "variant" key, so any
    # checkpoint trained there (adaln or adaln_cross_attn) falls through to
    # variant="baseline" above, which build_variant_model constructs as
    # ECGTransformerDiffusionVariant with use_cross_attn=False -- missing
    # the disease_token_emb/cross_attn/cross_norm/cross_gate parameters an
    # adaln_cross_attn checkpoint's state_dict actually has. Detect that
    # case directly from the state_dict's own keys (source of truth, not
    # a config guess) and construct the real ECGTransformerDiffusion class
    # with conditioning="adaln_cross_attn" instead. Everything else
    # (Stage 3 candidate checkpoints, plain adaln step04 checkpoints) is
    # unaffected -- unchanged build_variant_model path, still strict=True.
    state_dict = ckpt["model"]
    is_cross_attn = any(
        "cross_attn" in k or "cross_gate" in k or "disease_token" in k
        for k in state_dict.keys()
    )

    if is_cross_attn:
        cfg_cross = copy.deepcopy(cfg)
        cfg_cross.diffusion.conditioning = "adaln_cross_attn"
        model = ECGTransformerDiffusion(cfg_cross, n_classes=n_classes).to(device)
    else:
        model = build_variant_model(cfg, n_classes=n_classes, variant=variant).to(device)
    model.load_state_dict(state_dict, strict=True)
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
