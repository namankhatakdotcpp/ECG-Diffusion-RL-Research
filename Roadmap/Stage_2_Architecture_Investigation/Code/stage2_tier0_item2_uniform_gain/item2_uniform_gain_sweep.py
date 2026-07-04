"""
Stage 2 / Tier 0 Item 2B, Phase C -- uniform-gain full sweep.

Runs AFTER item2_uniform_gain.py's independent identity-regression test
has PASSED (outputs/identity_regression_test.json, all_pass=true).
Implements Item 2 v3's locked grid for the uniform variant (blocks 1-5,
cumulative substitution, budget-matched per common/utils.py::
uniform_per_block_gain).

Per explicit instruction, this file stops after the numerical sweep --
no plots, no Item2B_Report.md, no STAGE2_STATUS.md update, no commit.

Efficiency note (same reasoning as Item 2A's item2_gain_sweep.py): the
class-A pass and each target block's raw class-A output are
gain-independent (class A is never modified), so per (pair, timestep,
draw) this script runs ONE class-A pass (caching all 5 blocks' raw
outputs) and ONE class-B baseline pass, then reruns ONLY the class-B
forward once per gain value with the 5 CorrectionHooks live -- blocks
2-5's actual computation still depends on gain (since they consume the
already-corrected trajectory from the block before them), so unlike
Item 2A's single-substitution case, this cannot be reduced further
without re-running the full forward pass per gain.
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
from common.hooks import register_layer_hooks, RawCaptureHook, CorrectionHook  # noqa: E402
from common.metrics import magnitude_and_consistency  # noqa: E402
from common.statistics import decision_table_verdict, monotonicity_check, POOLED_BLOCK1_TO_2_DROP  # noqa: E402
from common.utils import GAIN_GRID, TIMESTEPS, K_DRAWS, class_pairs, uniform_per_block_gain, N_UNIFORM_BLOCKS  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item2_uniform_gain"
)

TARGET_BLOCK_IDXS = list(range(N_UNIFORM_BLOCKS))  # [0,1,2,3,4]


def run_cell(model, device, n_leads, seq_len, t_val, y_a_val, y_b_val, gains):
    n_layers = len(model.blocks)
    baseline_a = [[] for _ in range(n_layers)]
    baseline_b = [[] for _ in range(n_layers)]
    corrected_b = {g: [[] for _ in range(n_layers)] for g in gains}
    g1_max_abs_diff = []

    for draw in range(K_DRAWS):
        torch.manual_seed(1000 + draw)
        x_t = torch.randn(1, n_leads, seq_len, device=device)
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        y_a = torch.full((1,), y_a_val, device=device, dtype=torch.long)
        y_b = torch.full((1,), y_b_val, device=device, dtype=torch.long)

        # Class-A pass: mean-pooled capture (all 6 layers) + raw capture on blocks 0-4.
        handles, captured = register_layer_hooks(model)
        raw_hooks = [RawCaptureHook() for _ in TARGET_BLOCK_IDXS]
        raw_handles = [model.blocks[i].register_forward_hook(raw_hooks[j])
                       for j, i in enumerate(TARGET_BLOCK_IDXS)]
        with torch.no_grad():
            model(x_t, t, y_a)
        feat_a = {k: v.clone() for k, v in captured.items()}
        cached_A_raw = [h.tensor.clone() for h in raw_hooks]
        for h in raw_handles:
            h.remove()
        for h in handles:
            h.remove()

        # Class-B raw baseline pass (no correction).
        handles, captured = register_layer_hooks(model)
        with torch.no_grad():
            model(x_t, t, y_b)
        feat_b = {k: v.clone() for k, v in captured.items()}
        for h in handles:
            h.remove()

        for layer in range(n_layers):
            baseline_a[layer].append(feat_a[layer][0].numpy())
            baseline_b[layer].append(feat_b[layer][0].numpy())

        # Corrected class-B passes, one per gain, cumulative substitution at blocks 0-4.
        for g in gains:
            g_k = uniform_per_block_gain(g)
            handles, captured = register_layer_hooks(model)
            correction_hooks = [CorrectionHook(cached_A=cached_A_raw[j], gain=g_k)
                                for j in range(len(TARGET_BLOCK_IDXS))]
            correction_handles = [model.blocks[i].register_forward_hook(correction_hooks[j])
                                  for j, i in enumerate(TARGET_BLOCK_IDXS)]
            with torch.no_grad():
                model(x_t, t, y_b)
            feat_b_corr = {k: v.clone() for k, v in captured.items()}
            for h in correction_handles:
                h.remove()
            for h in handles:
                h.remove()

            for layer in range(n_layers):
                corrected_b[g][layer].append(feat_b_corr[layer][0].numpy())

            if g == 1.0:
                for layer in range(n_layers):
                    diff = float(np.abs(feat_b_corr[layer][0].numpy() - feat_b[layer][0].numpy()).max())
                    g1_max_abs_diff.append(diff)

    return baseline_a, baseline_b, corrected_b, g1_max_abs_diff


def main() -> None:
    cfg = load_config()
    log = get_logger("item2_uniform_gain_sweep", cfg=cfg)
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

    identity_path = OUT_DIR / "identity_regression_test.json"
    if not identity_path.exists():
        print(f"[BLOCKED] {identity_path} not found -- run item2_uniform_gain.py (Phase A/B) first.")
        return
    with open(identity_path) as f:
        identity_result = json.load(f)
    if not identity_result.get("all_pass"):
        print("[BLOCKED] Uniform-hook identity-regression test did not pass -- aborting per Item 2 v3 Sec. 6.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = class_pairs(n_classes)
    log.info(f"Phase C sweep (uniform): gains={GAIN_GRID}, pairs={pairs}, timesteps={TIMESTEPS}, "
             f"k_draws={K_DRAWS}, n_layers={n_layers}, target_blocks(0-idx)={TARGET_BLOCK_IDXS}")

    per_gain_recovered = {g: [] for g in GAIN_GRID}
    per_gain_injected = {g: [] for g in GAIN_GRID}
    per_gain_efficiency = {g: [] for g in GAIN_GRID}
    per_gain_layer_consistency = {g: [[] for _ in range(n_layers)] for g in GAIN_GRID}
    per_gain_layer_magnitude = {g: [[] for _ in range(n_layers)] for g in GAIN_GRID}
    baseline_layer_magnitude_all = [[] for _ in range(n_layers)]
    baseline_layer_consistency_all = [[] for _ in range(n_layers)]
    g1_identity_diffs_all = []
    per_cell_detail = []

    for (y_a_val, y_b_val) in pairs:
        for t_val in TIMESTEPS:
            log.info(f"Cell: pair (0->{y_b_val}), t={t_val}")
            baseline_a, baseline_b, corrected_b, g1_diffs = run_cell(
                model, device, n_leads, seq_len, t_val, y_a_val, y_b_val, GAIN_GRID
            )
            g1_identity_diffs_all.extend(g1_diffs)

            cell_baseline_mag = []
            cell_baseline_cons = []
            for layer in range(n_layers):
                mag, cons = magnitude_and_consistency(baseline_a[layer], baseline_b[layer])
                cell_baseline_mag.append(mag)
                cell_baseline_cons.append(cons)
                baseline_layer_magnitude_all[layer].append(mag)
                baseline_layer_consistency_all[layer].append(cons)

            mag6_baseline = cell_baseline_mag[n_layers - 1]
            mag1_baseline = cell_baseline_mag[0]  # InjectedDelta base, Item 1 units, Sec. 5 -- unchanged, location-agnostic

            cell_detail = {
                "pair": f"0->{y_b_val}", "timestep": t_val,
                "baseline_magnitude_per_layer": cell_baseline_mag,
                "baseline_direction_consistency_per_layer": cell_baseline_cons,
                "gains": {},
            }

            for g in GAIN_GRID:
                g_k = uniform_per_block_gain(g)
                cell_mag = []
                cell_cons = []
                for layer in range(n_layers):
                    mag, cons = magnitude_and_consistency(baseline_a[layer], corrected_b[g][layer])
                    cell_mag.append(mag)
                    cell_cons.append(cons)
                    per_gain_layer_consistency[g][layer].append(cons)
                    per_gain_layer_magnitude[g][layer].append(mag)

                mag6_corrected = cell_mag[n_layers - 1]
                recovered = mag6_corrected - mag6_baseline
                injected = (g - 1.0) * mag1_baseline  # InjectedDelta uses nominal g, matching localized units
                efficiency = (recovered / injected) if abs(injected) > 1e-12 else None

                per_gain_recovered[g].append(recovered)
                per_gain_injected[g].append(injected)
                if efficiency is not None:
                    per_gain_efficiency[g].append(efficiency)

                cell_detail["gains"][str(g)] = {
                    "nominal_gain": g,
                    "per_block_gain": g_k,
                    "magnitude_per_layer": cell_mag,
                    "direction_consistency_per_layer": cell_cons,
                    "recovered_magnitude_block6": recovered,
                    "injected_delta": injected,
                    "propagation_efficiency": efficiency,
                }
            per_cell_detail.append(cell_detail)

    max_g1_diff = max(g1_identity_diffs_all) if g1_identity_diffs_all else None
    log.info(f"Built-in nominal_gain=1.0 identity re-check (sweep code path): max|diff| across all "
             f"cells/draws/layers = {max_g1_diff:.2e}" if max_g1_diff is not None else "no g=1 data")

    avg_baseline_magnitude = [float(np.mean(v)) for v in baseline_layer_magnitude_all]
    avg_baseline_consistency = [float(np.mean(v)) for v in baseline_layer_consistency_all]

    summary_rows = []
    for g in GAIN_GRID:
        g_k = uniform_per_block_gain(g)
        recovered_avg = float(np.mean(per_gain_recovered[g]))
        injected_avg = float(np.mean(per_gain_injected[g])) if per_gain_injected[g] else None
        efficiency_avg = float(np.mean(per_gain_efficiency[g])) if per_gain_efficiency[g] else None
        layer_cons_avg = [float(np.mean(v)) for v in per_gain_layer_consistency[g]]
        layer_mag_avg = [float(np.mean(v)) for v in per_gain_layer_magnitude[g]]
        min_direction_consistency = min(layer_cons_avg[1:])  # layers 2-6, Sec. 9

        recovery_fraction = recovered_avg / POOLED_BLOCK1_TO_2_DROP
        verdict = decision_table_verdict(recovery_fraction, min_direction_consistency)

        summary_rows.append({
            "nominal_gain": g,
            "per_block_gain": g_k,
            "recovered_magnitude_block6_avg": recovered_avg,
            "injected_delta_avg": injected_avg,
            "propagation_efficiency_avg": efficiency_avg,
            "recovery_fraction": recovery_fraction,
            "min_direction_consistency_layers_2_6": min_direction_consistency,
            "layer_direction_consistency_avg": layer_cons_avg,
            "layer_magnitude_avg": layer_mag_avg,
            "decision_table_verdict": verdict,
        })

        gain_json = {
            "nominal_gain": g,
            "per_block_gain": g_k,
            "n_pairs": len(pairs), "n_timesteps": len(TIMESTEPS), "k_draws": K_DRAWS,
            "target_blocks_0indexed": TARGET_BLOCK_IDXS,
            "baseline_magnitude_per_layer_avg": avg_baseline_magnitude,
            "baseline_direction_consistency_per_layer_avg": avg_baseline_consistency,
            "corrected_magnitude_per_layer_avg": layer_mag_avg,
            "corrected_direction_consistency_per_layer_avg": layer_cons_avg,
            "recovered_magnitude_block6_avg": recovered_avg,
            "injected_delta_avg": injected_avg,
            "propagation_efficiency_avg": efficiency_avg,
            "recovery_fraction": recovery_fraction,
            "min_direction_consistency_layers_2_6": min_direction_consistency,
            "decision_table_verdict": verdict,
            "per_cell_recovered_magnitude": per_gain_recovered[g],
            "per_cell_injected_delta": per_gain_injected[g],
            "per_cell_propagation_efficiency": per_gain_efficiency[g],
        }
        gpath = OUT_DIR / f"gain_{g:.2f}.json"
        with open(gpath, "w") as f:
            json.dump(gain_json, f, indent=2)
        log.info(f"nominal_gain={g} (per_block_gain={g_k:.4f}): recovery_fraction={recovery_fraction:.4f} "
                 f"min_dir_cons={min_direction_consistency:.4f} efficiency={efficiency_avg} verdict={verdict}")

    recoveries = [row["recovery_fraction"] for row in summary_rows]
    directions = [row["min_direction_consistency_layers_2_6"] for row in summary_rows]
    efficiencies = [row["propagation_efficiency_avg"] for row in summary_rows]
    monotonicity_flags = monotonicity_check(GAIN_GRID, recoveries, directions, efficiencies)
    bug_signature = any(f["classification"] == "BUG_SIGNATURE" for f in monotonicity_flags)

    efficiency_gate_fail = [row for row in summary_rows
                             if row["propagation_efficiency_avg"] is not None
                             and (row["propagation_efficiency_avg"] > 1.0 or row["propagation_efficiency_avg"] < 0.0)]
    nan_found = any(
        any(np.isnan(x) for x in row["layer_direction_consistency_avg"] + row["layer_magnitude_avg"])
        for row in summary_rows
    )

    with open(OUT_DIR / "sweep_summary.json", "w") as f:
        json.dump({
            "gain_grid": GAIN_GRID,
            "summary_rows": summary_rows,
            "monotonicity_flags": monotonicity_flags,
            "bug_signature_detected": bug_signature,
            "g1_builtin_identity_recheck_max_abs_diff": max_g1_diff,
            "per_cell_detail": per_cell_detail,
        }, f, indent=2)

    print(json.dumps({
        "summary_rows": summary_rows,
        "monotonicity_flags": monotonicity_flags,
        "bug_signature_detected": bug_signature,
        "efficiency_gate_fail_gt_100pct_or_negative": [row["nominal_gain"] for row in efficiency_gate_fail],
        "nan_found": nan_found,
        "g1_builtin_identity_recheck_max_abs_diff": max_g1_diff,
    }, indent=2, default=str))

    if bug_signature or nan_found:
        print("[STOP] A gate condition fired (bug-signature monotonicity or NaN). Investigate before "
              "trusting these numbers.")
    else:
        print("[OK] All gates clear.")

    print("STOP -- per explicit instruction: no plots, no Item2B_Report.md, no STAGE2_STATUS.md "
          "update, no commit. Numerical sweep complete, awaiting review.")


if __name__ == "__main__":
    main()
