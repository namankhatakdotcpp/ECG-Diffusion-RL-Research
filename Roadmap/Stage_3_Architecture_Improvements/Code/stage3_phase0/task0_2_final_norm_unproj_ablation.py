"""
Stage 3 / Phase 0 / Task 0.2 -- final_norm/unproj causal ablation check.

Implements the locked pre-registration
(Roadmap/Stage_3_Architecture_Improvements/Stage3_Phase0_PreRegistration.md).
Tests whether `final_norm` -> `unproj` (step04_transformer_diffusion.py:226,229)
disproportionately suppresses the conditioning-specific component of the
signal relative to the whole-tensor signal, as it passes through.

Method: directly extends Item 3's own causal-ablation pattern
(item3_block6_ablation.py's IdentityAblationHook -- override a block's
output with its own input, i.e. skip its contribution) one stage
further downstream than Item 3's own final_norm-only measurement, and
adds the "before" side (which Item 3's block6_ablation.py did not need,
since it measured an absolute post-ablation change, not a retention
RATIO across a layer pair).

Two ratios, each pool-first-then-ratio (Item 3 Finding 3's convention,
not a per-observation ratio averaged afterward):

    retention_ratio_conditioning = delta_after_conditioning / delta_before_conditioning
      delta_before_conditioning = ||block6_output(class B) - block6_output(class A)||   (pre-final_norm, cross-class)
      delta_after_conditioning  = ||model_output(class B) - model_output(class A)||     (post-unproj, cross-class)

    retention_ratio_whole = delta_after_whole / delta_before_whole
      delta_before_whole = ||block6_output(class A) - block6_input(class A)||           (= block 6's own residual
                                                                                            update magnitude, unnormalized)
      delta_after_whole  = ||model_output(class A, no ablation) - model_output(class A, block-6-ablated)||

Same 5 class-pairs x 3 timesteps x 20 draws design as Items 1/3/Task 0.1,
CPU-only, class A held fixed as reference (=0) matching class_pairs().
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

STAGE2_CODE_DIR = (
    Path(__file__).resolve().parents[3]  # Roadmap/
    / "Stage_2_Architecture_Investigation" / "Code"
)
if str(STAGE2_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_CODE_DIR))

from common.io import REPO_ROOT, load_config, get_logger, load_model_checkpoint  # noqa: E402
from common.utils import class_pairs, K_DRAWS, TIMESTEPS  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Outputs"
    / "stage3_phase0_task0_2_final_norm_unproj_ablation"
)

# Locked decision threshold, Stage3_Phase0_PreRegistration.md Task 0.2
IMPLICATES_THRESHOLD = 0.5


def run_pass(model, x_t, t, y, ablate: bool):
    """One forward pass. Captures block 6's raw per-token output signal
    (or, if ablate=True, block 6's own INPUT -- the identity-ablation
    value, same mechanism as Item 3's IdentityAblationHook: skip block
    6's residual update entirely) and the model's full final output
    (post final_norm -> unproj -> reshape, i.e. model.forward()'s own
    return value -- no separate final_norm/unproj hook needed since
    forward() already includes both)."""
    captured = {}

    def _hook(module, inp, out):
        if ablate:
            captured["block6_signal"] = inp[0].detach().cpu().clone()
            return inp[0]
        captured["block6_signal"] = out.detach().cpu().clone()
        return out

    handle = model.blocks[-1].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model_output = model(x_t, t, y)
    finally:
        handle.remove()
    return captured["block6_signal"], model_output.detach().cpu().clone()


def main() -> None:
    cfg = load_config()
    log = get_logger("task0_2_final_norm_unproj_ablation", cfg=cfg)
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
    log.info(f"Task 0.2 ablation: pairs={pairs}, timesteps={TIMESTEPS}, k_draws={K_DRAWS} "
             f"(same design as Items 1/3/Task 0.1, reused for consistency)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    delta_before_cond_list, delta_after_cond_list = [], []
    delta_before_whole_list, delta_after_whole_list = [], []

    for (y_a_val, y_b_val) in pairs:
        for t_val in TIMESTEPS:
            for draw in range(K_DRAWS):
                torch.manual_seed(1000 + draw)
                x_t = torch.randn(1, n_leads, seq_len, device=device)
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                y_a = torch.full((1,), y_a_val, device=device, dtype=torch.long)
                y_b = torch.full((1,), y_b_val, device=device, dtype=torch.long)

                sig_a_base, out_a_base = run_pass(model, x_t, t, y_a, ablate=False)
                sig_b_base, out_b_base = run_pass(model, x_t, t, y_b, ablate=False)
                sig_a_abl, out_a_abl = run_pass(model, x_t, t, y_a, ablate=True)

                sig_a_b, sig_b_b = sig_a_base.numpy(), sig_b_base.numpy()
                out_a_b, out_b_b = out_a_base.numpy(), out_b_base.numpy()
                sig_a_a, out_a_a = sig_a_abl.numpy(), out_a_abl.numpy()

                delta_before_cond_list.append(float(np.linalg.norm(sig_b_b - sig_a_b)))
                delta_after_cond_list.append(float(np.linalg.norm(out_b_b - out_a_b)))
                delta_before_whole_list.append(float(np.linalg.norm(sig_a_b - sig_a_a)))
                delta_after_whole_list.append(float(np.linalg.norm(out_a_b - out_a_a)))

    mean_delta_before_cond = float(np.mean(delta_before_cond_list))
    mean_delta_after_cond = float(np.mean(delta_after_cond_list))
    mean_delta_before_whole = float(np.mean(delta_before_whole_list))
    mean_delta_after_whole = float(np.mean(delta_after_whole_list))

    retention_ratio_conditioning = mean_delta_after_cond / mean_delta_before_cond
    retention_ratio_whole = mean_delta_after_whole / mean_delta_before_whole

    implicates = retention_ratio_conditioning < IMPLICATES_THRESHOLD * retention_ratio_whole

    result = {
        "mean_delta_before_conditioning": mean_delta_before_cond,
        "mean_delta_after_conditioning": mean_delta_after_cond,
        "retention_ratio_conditioning": retention_ratio_conditioning,
        "mean_delta_before_whole": mean_delta_before_whole,
        "mean_delta_after_whole": mean_delta_after_whole,
        "retention_ratio_whole": retention_ratio_whole,
        "ratio_of_ratios_conditioning_over_whole": retention_ratio_conditioning / retention_ratio_whole,
        "implicates_threshold": IMPLICATES_THRESHOLD,
        "n_observations": len(delta_before_cond_list),
        "ratio_formula": ("Pool-first-then-ratio (Item 3 Finding 3 convention): each retention_ratio "
                           "is mean(delta_after across all observations) / mean(delta_before across all "
                           "observations), not a mean of per-observation ratios."),
        "verdict": (
            "IMPLICATES final_norm/unproj as a fix target"
            if implicates else
            "RULES OUT final_norm/unproj as a fix target"
        ),
    }

    with open(OUT_DIR / "task0_2_raw.json", "w") as f:
        json.dump(result, f, indent=2)

    log.info(json.dumps(result, indent=2))
    log.info(f"retention_ratio_conditioning={retention_ratio_conditioning:.4f}, "
             f"retention_ratio_whole={retention_ratio_whole:.4f}, "
             f"threshold={IMPLICATES_THRESHOLD} * whole = {IMPLICATES_THRESHOLD * retention_ratio_whole:.4f}")
    log.info(f"VERDICT: {result['verdict']}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
