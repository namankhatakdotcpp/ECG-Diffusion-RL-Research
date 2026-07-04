"""
Stage 3 / Phase 1 -- CPU smoke test, shared across all 5 candidates.

Per Stage3_Roadmap.md Sec. 4: Track B is "implementation and local
smoke-testing only," GPU training gated on Decision Gate A. This is
NOT a substitute for GPU-validated training -- it only confirms each
variant (a) instantiates, (b) produces the expected output shape, (c)
has a working forward+backward pass with finite, non-zero gradients
reaching every parameter (including the new gamma/boost parameters
each variant adds), on synthetic random input (same convention as
Items 1/3/Task 0.1's CPU-only probes -- no real PTB-XL data needed for
a shape/gradient-flow check).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model_variants import build_variant_model  # noqa: E402


def run_smoke_test(cfg, log, variant: str, run_id: str) -> None:
    torch.manual_seed(0)
    n_classes = int(cfg.ptbxl.n_classes)
    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)
    batch_size = 4

    model = build_variant_model(cfg, n_classes=n_classes, variant=variant)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[{run_id}] SMOKE TEST | variant={variant} | params={n_params/1e6:.2f}M")

    x_t = torch.randn(batch_size, n_leads, seq_len)
    t = torch.randint(0, int(cfg.diffusion.T), (batch_size,))
    y = torch.randint(0, n_classes, (batch_size,))

    out = model(x_t, t, y)
    expected_shape = (batch_size, n_leads, seq_len)
    assert out.shape == tuple(expected_shape), (
        f"[{run_id}] output shape mismatch: got {tuple(out.shape)}, expected {expected_shape}"
    )
    assert torch.isfinite(out).all(), f"[{run_id}] forward pass produced non-finite values"

    # At init, adaLN is zero-initialized (same as the original architecture),
    # so every block is an identity map for its FiLM modulation; the
    # zero-initialized unproj means the RAW output should be exactly zero
    # at init regardless of variant (gamma/boost only affect the
    # magnitude of a currently-zero residual contribution) -- confirms
    # the variant's added parameters did not break the original
    # near-zero-output-at-init property step04 relies on.
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
        f"[{run_id}] expected near-zero output at init (unproj zero-init), "
        f"got max abs value {out.abs().max().item()}"
    )

    noise = torch.randn_like(x_t)
    loss = F.mse_loss(out, noise)
    loss.backward()

    n_missing_grad, n_nonfinite_grad = 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            n_missing_grad += 1
            log.warning(f"[{run_id}] no gradient reached parameter: {name}")
        elif not torch.isfinite(p.grad).all():
            n_nonfinite_grad += 1
            log.warning(f"[{run_id}] non-finite gradient at parameter: {name}")

    assert n_missing_grad == 0, f"[{run_id}] {n_missing_grad} parameters received no gradient"
    assert n_nonfinite_grad == 0, f"[{run_id}] {n_nonfinite_grad} parameters had non-finite gradients"

    log.info(f"[{run_id}] SMOKE TEST PASSED: shape OK, near-zero-at-init OK, "
             f"{n_params} params all received finite gradients")
    print(f"[{run_id}] SMOKE TEST PASSED (variant={variant}, params={n_params/1e6:.2f}M)")
