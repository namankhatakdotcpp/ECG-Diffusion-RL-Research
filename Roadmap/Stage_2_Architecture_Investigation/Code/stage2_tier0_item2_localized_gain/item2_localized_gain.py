"""
Stage 2 / Tier 0 Item 2 -- Localized Gain, Phase A (per Tier0_Findings.md's
Item 2 v3 pre-registration, commit e84c54c -- locked before this file was
written).

INVESTIGATE-BEFORE-CODE, confirmed prior to writing this script (read-only):
- Hook target tensor: step04_transformer_diffusion.py:177, the return value
  of TransformerBlock.forward() -- shape (1, 600, 256) float32
  (config.model.model_dim=256; 600 tokens per Item 1's own docstring), on
  whatever device _resolve_device(cfg) selects (CPU on this machine, per
  every Item 1 run so far).
- Delta/capture mechanism reuses Item 1's own `_register_layer_hooks` and
  `cosine_sim` (imported directly from the Stage 2 copy of
  layerwise_direction_probe.py) rather than reimplemented independently, per
  the pre-registration's explicit instruction.
- Checkpoint: outputs/models/diffusion_best.pt -- the same checkpoint every
  Item 1 run used (exp1_baseline_reproduction's checkpoint). Item 2 does not
  introduce a different one.

THIS FILE IMPLEMENTS PHASE A1-A2 ONLY:
  A1. The localized-gain substitution hook (Item 2 v3, Section 3).
  A2. The mandatory g=1.0 identity-regression test (Item 2 v3, Section 6).
Per the agreed strict Phase A/B/C ordering, this script stops after A2 and
reports the result -- it does NOT run the real gain sweep (Phase A3), which
is gated on this test passing and lives in a separate script/run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

ITEM1_CODE_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Code"
    / "stage2_tier0_item1_layerwise_magnitude_direction"
)
sys.path.insert(0, str(ITEM1_CODE_DIR))

from utils import load_config, get_logger  # noqa: E402
from mentor_eval.checkpoint_utils import load_checkpoint  # noqa: E402
from layerwise_direction_probe import _register_layer_hooks  # noqa: E402  (reused, not reimplemented)

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item2_localized_gain"
)

# block index 0 = "block 1" in Item 1's 1-indexed reporting; its OUTPUT is
# what feeds "block 2" -- this is the block1->2 transition Item 1 identified
# as the dominant attenuation point, and the transition this localized
# variant intervenes on (Item 2 v3, Section 3).
TARGET_BLOCK_IDX = 0


class LocalizedGainHook:
    """Forward hook on model.blocks[TARGET_BLOCK_IDX].

    Item 2 v3, Section 2-3: during the class-A pass, caches the raw,
    unmodified full per-token output H_A(i) (never substituted -- class A
    remains the fixed reference throughout, exactly as in Item 1). During
    the class-B pass, computes Delta(i) = H_B_raw(i) - H_A(i) and returns
    H_A(i) + gain * Delta(i) in its place, so blocks downstream of this one
    consume the corrected tensor instead of the original H_B(i).
    """

    def __init__(self, gain: float):
        self.gain = gain
        self.mode: str | None = None  # "A" or "B", set by the caller before each forward()
        self.cached_A: torch.Tensor | None = None

    def __call__(self, module, inp, out):
        if self.mode == "A":
            self.cached_A = out.detach().clone()
            return out  # unmodified -- class A is never perturbed
        elif self.mode == "B":
            if self.cached_A is None:
                raise RuntimeError("Class-A pass must run before class-B pass.")
            delta = out - self.cached_A
            corrected = self.cached_A + self.gain * delta
            return corrected
        else:
            raise RuntimeError("LocalizedGainHook.mode not set before forward().")


def run_baseline_pass(model, x_t, t, y_a, y_b):
    """The original, unmodified paired forward pass -- identical in method to
    Item 1's own script, no substitution hook attached anywhere."""
    handles, captured = _register_layer_hooks(model)
    try:
        with torch.no_grad():
            model(x_t, t, y_a)
        captured_a = {k: v.clone() for k, v in captured.items()}
        with torch.no_grad():
            model(x_t, t, y_b)
        captured_b = {k: v.clone() for k, v in captured.items()}
    finally:
        for h in handles:
            h.remove()
    return captured_a, captured_b


def run_paired_pass_with_substitution(model, x_t, t, y_a, y_b, gain_hook: LocalizedGainHook,
                                       target_block_idx: int = TARGET_BLOCK_IDX):
    """Registers the standard multi-layer capture hooks FIRST (so they observe
    every block's RAW output, including the target block's own raw,
    unmodified output -- Item 1's per-layer metric semantics for the
    intervened block itself are preserved), THEN attaches the substitution
    hook onto the target block SECOND, so it fires after capture and
    overrides the value seen by the downstream blocks during the class-B
    pass. PyTorch calls a module's forward hooks in registration order, and
    each hook's returned value (if any) becomes the output seen by the next
    hook -- this ordering is what makes layer-(target+1)'s captured metric
    reflect the raw B value while blocks after the target consume the
    corrected one."""
    handles, captured = _register_layer_hooks(model)
    sub_handle = model.blocks[target_block_idx].register_forward_hook(gain_hook)
    try:
        gain_hook.mode = "A"
        with torch.no_grad():
            model(x_t, t, y_a)
        captured_a = {k: v.clone() for k, v in captured.items()}

        gain_hook.mode = "B"
        with torch.no_grad():
            model(x_t, t, y_b)
        captured_b = {k: v.clone() for k, v in captured.items()}
    finally:
        sub_handle.remove()
        for h in handles:
            h.remove()
    return captured_a, captured_b


def identity_regression_test(model, device, n_leads, seq_len, t_val, n_layers):
    """Item 2 v3, Section 6 -- mandatory before evaluating any gain != 1.0.
    Confirms the hook-substitution pipeline at gain=1.0 reproduces the
    unmodified forward pass to floating-point tolerance, for the localized
    (block-index-0) hook specifically. Uses class pair 0->1 (NORM vs MI),
    the same first pair and draw-seeding convention (torch.manual_seed(1000
    + draw)) as Item 1's own script, at t=500 (Item 1's baseline timestep)."""
    torch.manual_seed(1000)
    x_t = torch.randn(1, n_leads, seq_len, device=device)
    t = torch.full((1,), t_val, device=device, dtype=torch.long)
    y_a = torch.full((1,), 0, device=device, dtype=torch.long)
    y_b = torch.full((1,), 1, device=device, dtype=torch.long)

    baseline_a, baseline_b = run_baseline_pass(model, x_t, t, y_a, y_b)

    gain_hook = LocalizedGainHook(gain=1.0)
    corrected_a, corrected_b = run_paired_pass_with_substitution(model, x_t, t, y_a, y_b, gain_hook)

    results = {}
    all_pass = True
    for layer in range(n_layers):
        max_abs_diff_a = (baseline_a[layer] - corrected_a[layer]).abs().max().item()
        max_abs_diff_b = (baseline_b[layer] - corrected_b[layer]).abs().max().item()
        close_a = torch.allclose(baseline_a[layer], corrected_a[layer], atol=1e-5, rtol=1e-5)
        close_b = torch.allclose(baseline_b[layer], corrected_b[layer], atol=1e-5, rtol=1e-5)
        results[layer + 1] = {  # 1-indexed to match Item 1's reporting convention
            "max_abs_diff_class_A": max_abs_diff_a,
            "max_abs_diff_class_B": max_abs_diff_b,
            "allclose_class_A": close_a,
            "allclose_class_B": close_b,
        }
        all_pass = all_pass and close_a and close_b
    return all_pass, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("item2_localized_gain", cfg=cfg)
    torch.manual_seed(0)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(f"[BLOCKED] Checkpoint not found at {ckpt_path}. Run Experiment 1 first.")
        return

    model = loaded.model
    device = loaded.device
    n_layers = len(model.blocks)
    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)
    t_val = int(int(cfg.diffusion.T) * 0.5)  # t=500, matching Item 1's baseline timestep

    log.info(f"Item 2 Phase A2: identity-regression test (gain=1.0), localized hook on "
             f"block index {TARGET_BLOCK_IDX} (block {TARGET_BLOCK_IDX + 1} in 1-indexed terms).")
    all_pass, results = identity_regression_test(model, device, n_leads, seq_len, t_val, n_layers)

    for layer, r in results.items():
        log.info(f"Layer {layer}: allclose(A)={r['allclose_class_A']} allclose(B)={r['allclose_class_B']} "
                 f"max|diff| A={r['max_abs_diff_class_A']:.2e} B={r['max_abs_diff_class_B']:.2e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "identity_regression_test.json", "w") as f:
        json.dump({"all_pass": all_pass, "per_layer": results}, f, indent=2)

    if all_pass:
        print("IDENTITY REGRESSION TEST: PASSED -- gain=1.0 reproduces the unmodified forward "
              "pass to floating-point tolerance at every layer, for both class-A and class-B "
              "passes. Per Item 2 v3 Section 6, safe to proceed to Phase A3 (real gain sweep) "
              "pending sign-off.")
    else:
        print("IDENTITY REGRESSION TEST: FAILED -- see per-layer diffs above. "
              "ABORTING per Item 2 v3 Section 6 -- do not proceed to Phase A3.")


if __name__ == "__main__":
    main()
