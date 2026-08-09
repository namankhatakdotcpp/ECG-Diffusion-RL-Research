"""
test_cross_attn_conditioning.py — smoke test for adaln_cross_attn conditioning.

Verifies, with a single dummy forward pass and no GPU/dataset required:
  1. adaln_cross_attn output shape matches the adaln baseline exactly.
  2. Every cross-attention gate is zero-initialized.
  3. At initialization, adaln_cross_attn is bit-identical to adaln given the
     same shared weights (the zero-init gate must make the extra pathway a
     true no-op until training moves it away from zero).

Run before spending any GPU time on the adaln_cross_attn variant:
    python test_cross_attn_conditioning.py
"""

from __future__ import annotations

import copy

import torch

from step04_transformer_diffusion import ECGTransformerDiffusion
from utils import load_config


def main() -> None:
    cfg = load_config()
    n_classes = 2  # NORM, MI — matches the planned 2-class experiment

    cfg_adaln = copy.deepcopy(cfg)
    cfg_adaln.diffusion.conditioning = "adaln"
    cfg_cross = copy.deepcopy(cfg)
    cfg_cross.diffusion.conditioning = "adaln_cross_attn"

    model_adaln = ECGTransformerDiffusion(cfg_adaln, n_classes=n_classes).eval()
    model_cross = ECGTransformerDiffusion(cfg_cross, n_classes=n_classes).eval()

    # Gate check: every cross-attn block must start at exactly zero.
    gated_blocks = [b for b in model_cross.blocks if b.use_cross_attn]
    assert gated_blocks, "Expected at least one block with use_cross_attn=True"
    for i, block in enumerate(gated_blocks):
        assert block.cross_gate.item() == 0.0, f"cross_gate for block {i} is not zero-initialized"
    print(f"✓ {len(gated_blocks)}/{len(model_cross.blocks)} blocks have use_cross_attn=True, all gates zero-initialized")

    # Copy every shared-name, shared-shape parameter from model_cross into
    # model_adaln, so the only difference between the two models is the
    # extra (gated-to-zero) cross-attention machinery.
    adaln_state = model_adaln.state_dict()
    cross_state = model_cross.state_dict()
    copied = 0
    for k in adaln_state:
        if k in cross_state and adaln_state[k].shape == cross_state[k].shape:
            adaln_state[k] = cross_state[k]
            copied += 1
    model_adaln.load_state_dict(adaln_state)
    print(f"✓ Copied {copied}/{len(adaln_state)} shared parameters from adaln_cross_attn model into adaln model")

    # Dummy forward pass.
    torch.manual_seed(0)
    B, n_leads, sig_len = 4, 12, int(cfg.ptbxl.signal_length)
    x_t   = torch.randn(B, n_leads, sig_len)
    t     = torch.randint(0, int(cfg.diffusion.T), (B,))
    label = torch.randint(0, n_classes, (B,))

    with torch.no_grad():
        out_adaln = model_adaln(x_t, t, label)
        out_cross = model_cross(x_t, t, label)

    assert out_cross.shape == out_adaln.shape == (B, n_leads, sig_len), (
        f"Shape mismatch: adaln={out_adaln.shape}, cross_attn={out_cross.shape}"
    )
    print(f"✓ Output shapes match: {tuple(out_cross.shape)}")

    assert torch.equal(out_adaln, out_cross), (
        "adaln_cross_attn output is NOT bit-identical to adaln at initialization — "
        "the zero-init gate is not fully suppressing the cross-attention pathway."
    )
    print("✓ adaln_cross_attn is bit-identical to adaln at initialization (gate confirmed zero-effect)")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
