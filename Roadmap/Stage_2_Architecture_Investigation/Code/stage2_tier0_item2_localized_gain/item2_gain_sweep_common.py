"""
Stage 2 / Tier 0 Item 2A -- refactored copy of item2_gain_sweep.py that
imports its hooks/metrics/decision-table logic from Code/common/ instead
of reimplementing them inline.

This file exists ONLY to verify the common/ extraction is bit-identical
to Item 2A's own already-committed, already-verified result before Item
2B (and later items) start depending on common/ by default. It writes to
a separate `_refactor_verify/` output subdirectory -- it does NOT
overwrite item2_gain_sweep.py's canonical outputs/stage2_tier0_item2_
localized_gain/sweep_summary.json, which remains Item 2A's historical
record.

item2_gain_sweep.py itself is left untouched on disk, per the
consolidation instructions -- this is a new file, not an edit to it.
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
from common.hooks import register_layer_hooks, RawCaptureHook, OverrideHook  # noqa: E402
from common.metrics import magnitude_and_consistency  # noqa: E402
from common.statistics import (  # noqa: E402
    decision_table_verdict, monotonicity_check, POOLED_BLOCK1_TO_2_DROP,
)
from common.utils import GAIN_GRID, TIMESTEPS, K_DRAWS, class_pairs  # noqa: E402

CANONICAL_OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item2_localized_gain"
)
VERIFY_OUT_DIR = CANONICAL_OUT_DIR / "_refactor_verify"

TARGET_BLOCK_IDX = 0  # block 1 (1-indexed) -- localized variant, Item 2 v3 Sec. 3


def run_cell(model, device, n_leads, seq_len, t_val, y_a_val, y_b_val, gains):
    n_layers = len(model.blocks)
    baseline_a = [[] for _ in range(n_layers)]
    baseline_b = [[] for _ in range(n_layers)]
    corrected_b = {g: [[] for _ in range(n_layers)] for g in gains}
    g1_max_abs_diff = []

    raw_hook = RawCaptureHook()
    override_hook = OverrideHook()

    for draw in range(K_DRAWS):
        torch.manual_seed(1000 + draw)
        x_t = torch.randn(1, n_leads, seq_len, device=device)
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        y_a = torch.full((1,), y_a_val, device=device, dtype=torch.long)
        y_b = torch.full((1,), y_b_val, device=device, dtype=torch.long)

        handles, captured = register_layer_hooks(model)
        raw_handle = model.blocks[TARGET_BLOCK_IDX].register_forward_hook(raw_hook)
        with torch.no_grad():
            model(x_t, t, y_a)
        feat_a = {k: v.clone() for k, v in captured.items()}
        h1_a_full = raw_hook.tensor.clone()
        raw_handle.remove()
        for h in handles:
            h.remove()

        handles, captured = register_layer_hooks(model)
        raw_handle = model.blocks[TARGET_BLOCK_IDX].register_forward_hook(raw_hook)
        with torch.no_grad():
            model(x_t, t, y_b)
        feat_b = {k: v.clone() for k, v in captured.items()}
        h1_b_full = raw_hook.tensor.clone()
        raw_handle.remove()
        for h in handles:
            h.remove()

        delta1_full = h1_b_full - h1_a_full

        for layer in range(n_layers):
            baseline_a[layer].append(feat_a[layer][0].numpy())
            baseline_b[layer].append(feat_b[layer][0].numpy())

        for g in gains:
            override_hook.override = h1_a_full + g * delta1_full
            handles, captured = register_layer_hooks(model)
            sub_handle = model.blocks[TARGET_BLOCK_IDX].register_forward_hook(override_hook)
            with torch.no_grad():
                model(x_t, t, y_b)
            feat_b_corr = {k: v.clone() for k, v in captured.items()}
            sub_handle.remove()
            for h in handles:
                h.remove()
            override_hook.override = None

            for layer in range(n_layers):
                corrected_b[g][layer].append(feat_b_corr[layer][0].numpy())

            if g == 1.0:
                for layer in range(n_layers):
                    diff = float(np.abs(feat_b_corr[layer][0].numpy() - feat_b[layer][0].numpy()).max())
                    g1_max_abs_diff.append(diff)

    return baseline_a, baseline_b, corrected_b, g1_max_abs_diff


def main() -> None:
    cfg = load_config()
    log = get_logger("item2_gain_sweep_common", cfg=cfg)
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

    VERIFY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = class_pairs(n_classes)
    log.info(f"Refactor-verification sweep: gains={GAIN_GRID}, pairs={pairs}, "
             f"timesteps={TIMESTEPS}, k_draws={K_DRAWS}, n_layers={n_layers}")

    per_gain_recovered = {g: [] for g in GAIN_GRID}
    per_gain_injected = {g: [] for g in GAIN_GRID}
    per_gain_efficiency = {g: [] for g in GAIN_GRID}
    per_gain_layer_consistency = {g: [[] for _ in range(n_layers)] for g in GAIN_GRID}
    per_gain_layer_magnitude = {g: [[] for _ in range(n_layers)] for g in GAIN_GRID}
    baseline_layer_magnitude_all = [[] for _ in range(n_layers)]
    g1_identity_diffs_all = []

    for (y_a_val, y_b_val) in pairs:
        for t_val in TIMESTEPS:
            log.info(f"Cell: pair (0->{y_b_val}), t={t_val}")
            baseline_a, baseline_b, corrected_b, g1_diffs = run_cell(
                model, device, n_leads, seq_len, t_val, y_a_val, y_b_val, GAIN_GRID
            )
            g1_identity_diffs_all.extend(g1_diffs)

            cell_baseline_mag = []
            for layer in range(n_layers):
                mag, _ = magnitude_and_consistency(baseline_a[layer], baseline_b[layer])
                cell_baseline_mag.append(mag)
                baseline_layer_magnitude_all[layer].append(mag)

            mag6_baseline = cell_baseline_mag[n_layers - 1]
            mag1_baseline = cell_baseline_mag[0]

            for g in GAIN_GRID:
                cell_mag = []
                for layer in range(n_layers):
                    mag, cons = magnitude_and_consistency(baseline_a[layer], corrected_b[g][layer])
                    cell_mag.append(mag)
                    per_gain_layer_consistency[g][layer].append(cons)
                    per_gain_layer_magnitude[g][layer].append(mag)

                mag6_corrected = cell_mag[n_layers - 1]
                recovered = mag6_corrected - mag6_baseline
                injected = (g - 1.0) * mag1_baseline
                efficiency = (recovered / injected) if abs(injected) > 1e-12 else None

                per_gain_recovered[g].append(recovered)
                per_gain_injected[g].append(injected)
                if efficiency is not None:
                    per_gain_efficiency[g].append(efficiency)

    max_g1_diff = max(g1_identity_diffs_all) if g1_identity_diffs_all else None

    summary_rows = []
    for g in GAIN_GRID:
        recovered_avg = float(np.mean(per_gain_recovered[g]))
        efficiency_avg = float(np.mean(per_gain_efficiency[g])) if per_gain_efficiency[g] else None
        layer_cons_avg = [float(np.mean(v)) for v in per_gain_layer_consistency[g]]
        min_direction_consistency = min(layer_cons_avg[1:])
        recovery_fraction = recovered_avg / POOLED_BLOCK1_TO_2_DROP
        verdict = decision_table_verdict(recovery_fraction, min_direction_consistency)
        summary_rows.append({
            "gain": g,
            "recovered_magnitude_block6_avg": recovered_avg,
            "propagation_efficiency_avg": efficiency_avg,
            "recovery_fraction": recovery_fraction,
            "min_direction_consistency_layers_2_6": min_direction_consistency,
            "decision_table_verdict": verdict,
        })

    recoveries = [row["recovery_fraction"] for row in summary_rows]
    directions = [row["min_direction_consistency_layers_2_6"] for row in summary_rows]
    efficiencies = [row["propagation_efficiency_avg"] for row in summary_rows]
    monotonicity_flags = monotonicity_check(GAIN_GRID, recoveries, directions, efficiencies)

    with open(VERIFY_OUT_DIR / "sweep_summary.json", "w") as f:
        json.dump({
            "gain_grid": GAIN_GRID,
            "summary_rows": summary_rows,
            "monotonicity_flags": monotonicity_flags,
            "g1_builtin_identity_recheck_max_abs_diff": max_g1_diff,
        }, f, indent=2)

    print(f"Wrote refactor-verification output: {VERIFY_OUT_DIR / 'sweep_summary.json'}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
