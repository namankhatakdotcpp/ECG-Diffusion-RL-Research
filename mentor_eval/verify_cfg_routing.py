"""
mentor_eval/verify_cfg_routing.py — post-training CFG label-routing check.

Verifies that after training, the model's predicted noise actually differs
between a real class label and the null/unconditional label. At init, both
forward passes return identical (zero) output because unproj.weight/bias are
zero-initialized — the check is only meaningful once those weights have moved.

Usage:
    # Default: most-recent diffusion_ckpt_ep*.pt found by glob
    python -m mentor_eval.verify_cfg_routing

    # Explicit path
    python -m mentor_eval.verify_cfg_routing --ckpt outputs/models/diffusion_ckpt_ep0025.pt

Verdicts:
    LABEL ROUTING LIVE         — unproj is off zero AND eps_cond != eps_uncond
    OUTPUT HEAD STILL ZERO     — unproj ~0, check after more training
    WARNING: ROUTING BUG       — unproj is live but eps diff ~0, investigate cond path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, get_logger
from mentor_eval.checkpoint_utils import load_checkpoint

_UNPROJ_ZERO_THRESH = 1e-6   # unproj max-abs below this → still zero-initialized
_EPS_DIFF_THRESH    = 1e-4   # eps diff norm below this → routing not live


def _find_latest_checkpoint(models_dir: Path) -> Path | None:
    """Return the epoch checkpoint with the highest epoch number, or None."""
    candidates = sorted(models_dir.glob("diffusion_ckpt_ep*.pt"))
    if not candidates:
        return None
    # Sort by epoch number embedded in filename: diffusion_ckpt_ep0025.pt → 25
    def _epoch(p: Path) -> int:
        try:
            return int(p.stem.split("ep")[-1])
        except ValueError:
            return -1
    return max(candidates, key=_epoch)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify CFG label routing is live after training."
    )
    parser.add_argument(
        "--ckpt", type=str, default=None,
        help="Path to checkpoint. Default: most recent diffusion_ckpt_ep*.pt.",
    )
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("verify_cfg_routing", cfg=cfg)

    models_dir = Path(cfg.paths.outputs.models)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = _find_latest_checkpoint(models_dir)
        if ckpt_path is None:
            # Also try diffusion_best.pt as a fallback
            best = models_dir / "diffusion_best.pt"
            if best.exists():
                ckpt_path = best
            else:
                print(
                    f"\n[BLOCKED] No checkpoint found in {models_dir}.\n"
                    f"  Train step04 on the GPU server first, then re-run.\n"
                    f"  Expected: diffusion_ckpt_ep*.pt or diffusion_best.pt\n"
                )
                return

    if not ckpt_path.exists():
        print(
            f"\n[BLOCKED] Checkpoint not found: {ckpt_path}\n"
            f"  Train step04 on the GPU server first.\n"
        )
        return

    log.info(f"Loading checkpoint: {ckpt_path}")
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(f"\n[BLOCKED] load_checkpoint returned None for {ckpt_path}.\n")
        return

    model  = loaded.model   # already eval mode from load_checkpoint
    device = loaded.device
    log.info(f"Checkpoint: epoch={loaded.epoch}  val_loss={loaded.val_loss}  "
             f"classes={loaded.class_names}  device={device}")

    # ── Probe inputs ─────────────────────────────────────────────────────────
    # Use the device the model's parameters live on, not a hardcoded string
    param_device = next(model.parameters()).device
    dtype        = next(model.parameters()).dtype

    real_label = torch.zeros(2, dtype=torch.long, device=param_device)
    null_label = torch.full((2,), model.null_class_index, dtype=torch.long, device=param_device)
    x          = torch.randn(2, 12, int(cfg.ptbxl.signal_length),
                              dtype=dtype, device=param_device)
    t          = torch.zeros(2, dtype=torch.long, device=param_device)

    # ── Forward passes ────────────────────────────────────────────────────────
    with torch.no_grad():
        eps_cond   = model(x, t, real_label)
        eps_uncond = model(x, t, null_label)

    # ── Metrics ───────────────────────────────────────────────────────────────
    unproj_w_max = model.unproj.weight.abs().max().item()
    unproj_b_max = model.unproj.bias.abs().max().item()
    eps_diff_norm = (eps_cond - eps_uncond).norm().item()

    # Embedding diff — sanity-check label routing into class_emb is intact
    c_real = model.class_emb(real_label)
    c_null = model.class_emb(null_label)
    emb_diff_norm = (c_real - c_null).norm().item()

    print()
    print("=" * 60)
    print(f"CFG label-routing verification — {ckpt_path.name}")
    print("=" * 60)
    print(f"  epoch                    : {loaded.epoch}")
    print(f"  null_class_index         : {model.null_class_index}")
    print(f"  class_emb.weight.shape   : {tuple(model.class_emb.weight.shape)}")
    print()
    print(f"  unproj.weight max abs    : {unproj_w_max:.6f}")
    print(f"  unproj.bias   max abs    : {unproj_b_max:.6f}")
    print(f"  emb diff norm (real-null): {emb_diff_norm:.4f}  (expect ~33+ from random init)")
    print(f"  eps diff norm (cond-unc) : {eps_diff_norm:.6f}")
    print()

    unproj_live = (unproj_w_max > _UNPROJ_ZERO_THRESH
                   or unproj_b_max > _UNPROJ_ZERO_THRESH)
    diff_live   = eps_diff_norm > _EPS_DIFF_THRESH

    if unproj_live and diff_live:
        verdict = "LABEL ROUTING LIVE — CFG plumbing is functioning correctly."
    elif not unproj_live:
        verdict = (
            "OUTPUT HEAD STILL ZERO — unproj not yet off zero-init. "
            "Recheck after more training (first checkpoint is usually epoch 25)."
        )
    else:
        verdict = (
            "WARNING: ROUTING BUG — unproj is live but eps diff is ~0. "
            "Label signal is lost somewhere in cond construction or forward(). "
            "Investigate cond = t_emb + c_emb path and adaLN weight norms."
        )

    print(f"  VERDICT: {verdict}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
