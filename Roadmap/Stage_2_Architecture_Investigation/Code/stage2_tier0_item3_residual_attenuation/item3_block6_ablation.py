"""
Stage 2 / Tier 0 Item 3 -- causal ablation check, final validation before
locking (no further investigation after this, regardless of outcome).

Corrects a specific overclaim flagged in review: the earlier
block6_investigation.py compared pre-FinalNorm residual/output ratios
between block 6 and block 3, and concluded block 6's contribution
"survives" FinalNorm's compression. That claim is not supported by
pre-normalization ratios alone -- FinalNorm (LayerNorm) normalizes the
WHOLE tensor's variance, not any one block's specific contribution, so a
pre-norm ratio disparity says nothing about what's still distinguishable
after normalization. This script runs the actual causal test: ablate
block 6's residual update (override its output with its own input,
skipping its contribution entirely), pass the result through
`final_norm`, and measure how much the final_norm output changes versus
doing the identical ablation at block 3 (the valley). Whichever
ablation produces a larger change is the block whose contribution
genuinely survives normalization -- this is a direct measurement, not
an inference from pre-norm ratios.

Reuses the same 5x3x20 design, same checkpoint, same hook infrastructure
(a simple identity-ablation hook, structurally the same idea as
common/hooks.py's OverrideHook/CorrectionHook -- returns the block's own
input in place of its output, i.e. override = input, the degenerate
gain=0 case of "replace with a fixed tensor"). No new investigation
questions opened; this either confirms or downgrades the one claim
already drafted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]  # Roadmap/.../Code/
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.io import REPO_ROOT, load_config, get_logger, load_model_checkpoint  # noqa: E402
from common.utils import class_pairs, K_DRAWS, TIMESTEPS  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item3_residual_attenuation"
)

BLOCK6_IDX = 5  # 0-indexed
VALLEY_IDX = 2  # block 3, 0-indexed


class IdentityAblationHook:
    """Returns the block's own INPUT in place of its output -- skips the
    block's residual update entirely (both sub-layers, attention+FFN),
    as if this block were not there. `inp[0]` is the tensor arg PyTorch
    forward hooks receive as the module's actual input."""

    def __call__(self, module, inp, out):
        return inp[0]


def register_final_norm_capture(model):
    captured = {}

    def _hook(module, inp, out):
        captured["final_norm_output"] = out.detach().mean(dim=1).cpu()

    handle = model.final_norm.register_forward_hook(_hook)
    return handle, captured


def run_pass(model, x_t, t, y, ablate_block_idx: int | None):
    """One forward pass. If ablate_block_idx is not None, that block's
    residual update is skipped (identity ablation). Returns final_norm's
    mean-pooled output."""
    handles = []
    if ablate_block_idx is not None:
        handles.append(model.blocks[ablate_block_idx].register_forward_hook(IdentityAblationHook()))
    fn_handle, fn_captured = register_final_norm_capture(model)
    try:
        with torch.no_grad():
            model(x_t, t, y)
        final_norm_output = fn_captured["final_norm_output"].clone()
    finally:
        fn_handle.remove()
        for h in handles:
            h.remove()
    return final_norm_output


def main() -> None:
    cfg = load_config()
    log = get_logger("item3_block6_ablation", cfg=cfg)
    torch.manual_seed(0)

    loaded = load_model_checkpoint(cfg)
    if loaded is None:
        print("[BLOCKED] Checkpoint not found. Run Experiment 1 first.")
        return

    model = loaded.model
    device = loaded.device
    n_classes = loaded.n_classes
    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)

    pairs = class_pairs(n_classes)
    log.info(f"Ablation check: pairs={pairs}, timesteps={TIMESTEPS}, k_draws={K_DRAWS} "
             f"(same design as main sweep, reused for consistency)")

    delta_6_list, delta_3_list = [], []
    baseline_norms, ablated6_norms, ablated3_norms = [], [], []

    for (y_a_val, y_b_val) in pairs:
        for t_val in TIMESTEPS:
            for draw in range(K_DRAWS):
                torch.manual_seed(1000 + draw)
                x_t = torch.randn(1, n_leads, seq_len, device=device)
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                y_a = torch.full((1,), y_a_val, device=device, dtype=torch.long)
                y_b = torch.full((1,), y_b_val, device=device, dtype=torch.long)

                for y in (y_a, y_b):
                    baseline = run_pass(model, x_t, t, y, ablate_block_idx=None)
                    ablated_6 = run_pass(model, x_t, t, y, ablate_block_idx=BLOCK6_IDX)
                    ablated_3 = run_pass(model, x_t, t, y, ablate_block_idx=VALLEY_IDX)

                    b = baseline[0].numpy()
                    a6 = ablated_6[0].numpy()
                    a3 = ablated_3[0].numpy()

                    delta_6_list.append(float(np.linalg.norm(b - a6)))
                    delta_3_list.append(float(np.linalg.norm(b - a3)))
                    baseline_norms.append(float(np.linalg.norm(b)))
                    ablated6_norms.append(float(np.linalg.norm(a6)))
                    ablated3_norms.append(float(np.linalg.norm(a3)))

    mean_delta_6 = float(np.mean(delta_6_list))
    mean_delta_3 = float(np.mean(delta_3_list))
    mean_baseline_norm = float(np.mean(baseline_norms))

    result = {
        "mean_baseline_final_norm_output_norm": mean_baseline_norm,
        "mean_delta_block6_ablation": mean_delta_6,
        "mean_delta_block3_valley_ablation": mean_delta_3,
        "ratio_delta6_to_delta3": mean_delta_6 / mean_delta_3 if mean_delta_3 > 1e-12 else None,
        "n_observations": len(delta_6_list),
    }

    with open(OUT_DIR / "block6_ablation.json", "w") as f:
        json.dump(result, f, indent=2)

    log.info(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    ratio = result["ratio_delta6_to_delta3"]
    if ratio is not None and ratio > 1.2:
        verdict = ("CONFIRMED: block 6's contribution produces a larger change in final_norm's "
                    "output than block 3's -- the pre-normalization disparity DOES translate to "
                    "greater post-normalization influence.")
    elif ratio is not None and ratio < (1 / 1.2):
        verdict = ("REVERSED: block 3's contribution produces a larger post-normalization change "
                    "than block 6's, despite block 6 having the larger pre-normalization residual -- "
                    "downgrade the claim.")
    else:
        verdict = ("ROUGHLY EQUAL: block 6's larger pre-normalization residual does NOT translate "
                    "to proportionally greater post-normalization influence -- downgrade the claim, "
                    "state as an open question.")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
