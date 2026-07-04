"""
Stage 2 / Tier 0 Item 2, Phase A/B -- Uniform Gain variant
(`stage2_tier0_item2_uniform_gain`), per Item 2 v3's locked pre-registration
(Tier0_Findings.md, commit e84c54c) and the budget-matching fairness rule
carried forward unchanged from v2 (quoted in full in
common/utils.py::uniform_per_block_gain's docstring).

INVESTIGATE-BEFORE-CODE, confirmed prior to writing this file (chat,
this session):
- Hook targets: model.blocks[0..4] (blocks 1-5, 1-indexed) -- cumulative
  substitution, block 6 (model.blocks[5]) never hooked (recovery is
  measured there, Sec. 4).
- Budget-matching formula, quoted from the pre-registration: uniform
  variant applies g_k = g_L^(1/sqrt(5)) at each of the 5 transitions, so
  that 5*(ln g_k)^2 = (ln g_L)^2 -- equal total squared log-gain to the
  localized variant's single injection at that same nominal grid value.
  Implemented as common/utils.py::uniform_per_block_gain. At
  nominal_gain=1.0, g_k=1.0 exactly (1.0**x == 1.0) -- required for this
  file's identity test to be a true no-op.
- common/metrics.py's magnitude_and_consistency and common/statistics.py's
  POOLED_BLOCK1_TO_2_DROP denominator are confirmed location-agnostic by
  direct code read (chat, this session) -- neither takes the hook
  location as an argument or assumes a single injection point. Reusable
  unchanged for the uniform variant's recovery-fraction computation.

THIS FILE IMPLEMENTS PHASE A (hook) AND PHASE B (mandatory g=1.0
identity-regression test) ONLY -- INDEPENDENT of Item 2A's own identity
test, per Item 2 v3 Sec. 6's explicit instruction that the uniform hook's
identity test is not assumed to carry over from the localized hook's,
since the cumulative bookkeeping is structurally different (5 chained
substitutions vs. 1). Stops after Phase B and reports the result --
does NOT run the real gain sweep, which is gated on this test passing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1]  # Roadmap/.../Code/
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.io import REPO_ROOT, load_config, get_logger, load_model_checkpoint  # noqa: E402
from common.hooks import register_layer_hooks, RawCaptureHook, CorrectionHook  # noqa: E402
from common.utils import uniform_per_block_gain, N_UNIFORM_BLOCKS  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item2_uniform_gain"
)

TARGET_BLOCK_IDXS = list(range(N_UNIFORM_BLOCKS))  # [0,1,2,3,4] -- blocks 1-5, 1-indexed


def run_baseline_pass(model, x_t, t, y_a, y_b):
    """Original, unmodified paired forward pass -- no hooks besides the
    standard multi-layer mean-pooled capture. Identical in method to Item
    1's and Item 2A's own baseline pass."""
    handles, captured = register_layer_hooks(model)
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


def run_paired_pass_with_uniform_substitution(model, x_t, t, y_a, y_b, nominal_gain: float):
    """Registers the standard multi-layer capture hooks FIRST (so they
    observe every block's RAW output, matching Item 1/2A's per-layer
    metric semantics), THEN attaches one CorrectionHook per target block
    (0-4) SECOND, each with its own per-block gain g_k = nominal_gain^
    (1/sqrt(5)). During the class-A pass, RawCaptureHook instances on the
    same 5 blocks cache each block's own never-modified reference tensor
    H_k^A(i). During the class-B pass, each CorrectionHook fires in block
    order and returns cached_A + g_k*(out - cached_A) -- block k+1 then
    receives the already-corrected trajectory from block k as its input,
    exactly the cumulative computation graph in Item 2 v3 Sec. 3."""
    g_k = uniform_per_block_gain(nominal_gain)

    handles, captured = register_layer_hooks(model)
    raw_hooks = [RawCaptureHook() for _ in TARGET_BLOCK_IDXS]
    raw_handles = [model.blocks[i].register_forward_hook(raw_hooks[j])
                   for j, i in enumerate(TARGET_BLOCK_IDXS)]
    try:
        with torch.no_grad():
            model(x_t, t, y_a)
        captured_a = {k: v.clone() for k, v in captured.items()}
        cached_A_raw = [h.tensor.clone() for h in raw_hooks]
    finally:
        for h in raw_handles:
            h.remove()
        for h in handles:
            h.remove()

    handles, captured = register_layer_hooks(model)
    correction_hooks = [CorrectionHook(cached_A=cached_A_raw[j], gain=g_k)
                        for j in range(len(TARGET_BLOCK_IDXS))]
    correction_handles = [model.blocks[i].register_forward_hook(correction_hooks[j])
                          for j, i in enumerate(TARGET_BLOCK_IDXS)]
    try:
        with torch.no_grad():
            model(x_t, t, y_b)
        captured_b = {k: v.clone() for k, v in captured.items()}
    finally:
        for h in correction_handles:
            h.remove()
        for h in handles:
            h.remove()

    return captured_a, captured_b, g_k


def identity_regression_test(model, device, n_leads, seq_len, t_val, n_layers):
    """Item 2 v3 Sec. 6, run INDEPENDENTLY for the uniform hook (not
    assumed to carry over from Item 2A's localized-hook test). Same
    class pair (0->1, t=500) and draw-seeding convention (torch.
    manual_seed(1000)) as Item 2A's own identity test, for direct
    comparability."""
    torch.manual_seed(1000)
    x_t = torch.randn(1, n_leads, seq_len, device=device)
    t = torch.full((1,), t_val, device=device, dtype=torch.long)
    y_a = torch.full((1,), 0, device=device, dtype=torch.long)
    y_b = torch.full((1,), 1, device=device, dtype=torch.long)

    baseline_a, baseline_b = run_baseline_pass(model, x_t, t, y_a, y_b)
    corrected_a, corrected_b, g_k = run_paired_pass_with_uniform_substitution(
        model, x_t, t, y_a, y_b, nominal_gain=1.0
    )

    results = {}
    all_pass = True
    for layer in range(n_layers):
        max_abs_diff_a = (baseline_a[layer] - corrected_a[layer]).abs().max().item()
        max_abs_diff_b = (baseline_b[layer] - corrected_b[layer]).abs().max().item()
        close_a = torch.allclose(baseline_a[layer], corrected_a[layer], atol=1e-5, rtol=1e-5)
        close_b = torch.allclose(baseline_b[layer], corrected_b[layer], atol=1e-5, rtol=1e-5)
        results[layer + 1] = {
            "max_abs_diff_class_A": max_abs_diff_a,
            "max_abs_diff_class_B": max_abs_diff_b,
            "allclose_class_A": close_a,
            "allclose_class_B": close_b,
        }
        all_pass = all_pass and close_a and close_b
    return all_pass, results, g_k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("item2_uniform_gain", cfg=cfg)
    torch.manual_seed(0)

    ckpt_path = Path(args.ckpt) if args.ckpt else None
    loaded = load_model_checkpoint(cfg, ckpt_path)
    if loaded is None:
        print("[BLOCKED] Checkpoint not found. Run Experiment 1 first.")
        return

    model = loaded.model
    device = loaded.device
    n_layers = len(model.blocks)
    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)
    t_val = int(int(cfg.diffusion.T) * 0.5)  # t=500, matching Item 1/2A's baseline timestep

    log.info(f"Item 2B Phase A/B: identity-regression test (nominal_gain=1.0), uniform hook on "
             f"blocks {[i + 1 for i in TARGET_BLOCK_IDXS]} (1-indexed), independent of Item 2A's test.")
    all_pass, results, g_k = identity_regression_test(model, device, n_leads, seq_len, t_val, n_layers)

    for layer, r in results.items():
        log.info(f"Layer {layer}: allclose(A)={r['allclose_class_A']} allclose(B)={r['allclose_class_B']} "
                 f"max|diff| A={r['max_abs_diff_class_A']:.2e} B={r['max_abs_diff_class_B']:.2e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "identity_regression_test.json", "w") as f:
        json.dump({
            "all_pass": all_pass,
            "nominal_gain": 1.0,
            "per_block_gain": g_k,
            "per_layer": results,
        }, f, indent=2)

    if all_pass:
        print(f"IDENTITY REGRESSION TEST (UNIFORM HOOK): PASSED -- nominal_gain=1.0 (per_block_gain="
              f"{g_k}) reproduces the unmodified forward pass to floating-point tolerance at every "
              f"layer, for both class-A and class-B passes. Independent of Item 2A's own test, per "
              f"Item 2 v3 Sec. 6. Safe to proceed to the full uniform-variant sweep pending sign-off.")
    else:
        print("IDENTITY REGRESSION TEST (UNIFORM HOOK): FAILED -- see per-layer diffs above. "
              "ABORTING per Item 2 v3 Sec. 6 -- do not proceed to the sweep.")


if __name__ == "__main__":
    main()
