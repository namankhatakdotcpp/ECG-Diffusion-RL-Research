"""
Stage 3 / Phase 0 / Task 0.1 -- Dilution-ratio test.

Implements the locked pre-registration
(Roadmap/Stage_3_Architecture_Improvements/Stage3_Phase0_PreRegistration.md).
Tests whether conditioning's proportional influence on the residual
stream (conditioning_delta(block_k) / total_output_norm(block_k))
declines from block 1 to block 6 -- the "dilution mechanism" flagged as
a hypothesis (not yet measured as a unified ratio) in
Stage2_Decision_Report.md Conclusion 5b.

Reuses Item 1's own hook points (common/hooks.py::register_layer_hooks)
and Item 1's own pooling convention (common/metrics.py::
magnitude_and_consistency) unmodified -- dilution_ratio(block_k, cell)
IS that function's existing `magnitude` return value, not a new
per-draw ratio reimplemented here. Same 5 class-pairs x 3 timesteps x
20 draws design as Items 1/3, CPU-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

STAGE2_CODE_DIR = (
    Path(__file__).resolve().parents[3]  # Roadmap/
    / "Stage_2_Architecture_Investigation" / "Code"
)
if str(STAGE2_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_CODE_DIR))

from common.io import REPO_ROOT, load_config, get_logger, load_model_checkpoint  # noqa: E402
from common.hooks import register_layer_hooks  # noqa: E402
from common.metrics import magnitude_and_consistency  # noqa: E402
from common.utils import class_pairs, K_DRAWS, TIMESTEPS  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_3_Architecture_Improvements" / "Outputs"
    / "stage3_phase0_task0_1_dilution_ratio"
)

# Locked decision thresholds, Stage3_Phase0_PreRegistration.md Task 0.1
ALPHA = 0.05
SUPPORTED_DECLINE_THRESHOLD = 0.30
NOT_SUPPORTED_DECLINE_THRESHOLD = 0.10


def run_single_pass(model, x_t, t, y):
    """One forward pass, mean-pooled per-block outputs, identical hook
    usage to Item 1/Item 3's own probes."""
    handles, captured = register_layer_hooks(model)
    try:
        with torch.no_grad():
            model(x_t, t, y)
        block_outputs = {k: v.clone() for k, v in captured.items()}
    finally:
        for h in handles:
            h.remove()
    return block_outputs


def main() -> None:
    cfg = load_config()
    log = get_logger("task0_1_dilution_ratio", cfg=cfg)
    torch.manual_seed(0)

    loaded = load_model_checkpoint(cfg)
    if loaded is None:
        print("[BLOCKED] Checkpoint not found. Run Experiment 1 first.")
        return

    model = loaded.model
    device = loaded.device
    n_classes = loaded.n_classes
    n_layers = len(model.blocks)
    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)

    pairs = class_pairs(n_classes)
    log.info(f"Task 0.1 sweep: pairs={pairs}, timesteps={TIMESTEPS}, k_draws={K_DRAWS}, "
             f"n_layers={n_layers}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = {"n_layers": n_layers, "k_draws": K_DRAWS, "timesteps": TIMESTEPS, "pairs": {}}
    # per-cell (pair, timestep) dilution_ratio per block -- n=15 cells
    dilution_cells = [[] for _ in range(n_layers)]

    for (y_a_val, y_b_val) in pairs:
        for t_val in TIMESTEPS:
            feat_a_by_layer = [[] for _ in range(n_layers)]
            feat_b_by_layer = [[] for _ in range(n_layers)]

            for draw in range(K_DRAWS):
                torch.manual_seed(1000 + draw)
                x_t = torch.randn(1, n_leads, seq_len, device=device)
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                y_a = torch.full((1,), y_a_val, device=device, dtype=torch.long)
                y_b = torch.full((1,), y_b_val, device=device, dtype=torch.long)

                out_a = run_single_pass(model, x_t, t, y_a)
                out_b = run_single_pass(model, x_t, t, y_b)

                for layer in range(n_layers):
                    feat_a_by_layer[layer].append(out_a[layer][0].numpy())
                    feat_b_by_layer[layer].append(out_b[layer][0].numpy())

            cell_ratios = []
            for layer in range(n_layers):
                magnitude, _consistency = magnitude_and_consistency(
                    feat_a_by_layer[layer], feat_b_by_layer[layer]
                )
                dilution_cells[layer].append(magnitude)
                cell_ratios.append(magnitude)

            raw["pairs"].setdefault(f"0->{y_b_val}", {})[str(t_val)] = {
                "dilution_ratio": cell_ratios,
            }
            log.info(f"Pair (0->{y_b_val}), t={t_val}: dilution_ratio="
                     f"{[round(x, 4) for x in cell_ratios]}")

    dilution_pooled = [float(np.mean(v)) for v in dilution_cells]

    block1_vals = np.array(dilution_cells[0])
    block6_vals = np.array(dilution_cells[n_layers - 1])
    try:
        stat, p_value = wilcoxon(block1_vals, block6_vals)
    except ValueError as e:
        stat, p_value = None, None
        log.info(f"Wilcoxon test could not run: {e}")

    net_decline = (dilution_pooled[0] - dilution_pooled[n_layers - 1]) / dilution_pooled[0]

    # Non-monotonicity tolerance check (descriptive, per pre-registration)
    diffs = np.diff(dilution_pooled)
    n_non_monotonic_steps = int(np.sum(diffs > 0))  # count of blocks where ratio increased vs prior

    if p_value is not None and p_value < ALPHA and net_decline >= SUPPORTED_DECLINE_THRESHOLD:
        verdict = "SUPPORTED"
    elif (
        net_decline < NOT_SUPPORTED_DECLINE_THRESHOLD
        or (p_value is not None and p_value >= ALPHA)
        or net_decline < 0
    ):
        verdict = "NOT SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"

    result = {
        "dilution_ratio_pooled_by_block": dilution_pooled,
        "wilcoxon_block1_vs_block6": {
            "statistic": float(stat) if stat is not None else None,
            "p_value": float(p_value) if p_value is not None else None,
            "n": len(block1_vals),
        },
        "net_relative_decline_block1_to_block6": net_decline,
        "n_non_monotonic_steps": n_non_monotonic_steps,
        "decision_thresholds": {
            "alpha": ALPHA,
            "supported_decline_threshold": SUPPORTED_DECLINE_THRESHOLD,
            "not_supported_decline_threshold": NOT_SUPPORTED_DECLINE_THRESHOLD,
        },
        "verdict": verdict,
    }
    raw["result"] = result
    with open(OUT_DIR / "task0_1_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    log.info(f"Pooled dilution_ratio by block: {[round(x, 4) for x in dilution_pooled]}")
    log.info(f"Wilcoxon (block1 vs block6, n={len(block1_vals)}): stat={stat}, p={p_value}")
    log.info(f"Net relative decline block1->block6: {net_decline:.4f}")
    log.info(f"VERDICT: {verdict}")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
