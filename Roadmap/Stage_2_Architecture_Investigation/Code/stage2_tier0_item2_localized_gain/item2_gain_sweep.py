"""
Stage 2 / Tier 0 Item 2, Phase A3-A6 -- localized-gain full sweep.

Runs AFTER item2_localized_gain.py's A2 identity-regression test has
PASSED (confirmed: outputs/identity_regression_test.json, all_pass=true).
Implements Item 2 v3 (Tier0_Findings.md, commit e84c54c + the two closing
clarifications) Sections 2, 3 (k=0 only -- localized variant), 4, 5, 7, 9.

Efficiency note (not a methodological change): the class-A forward pass
and the raw block-1 tensors do not depend on gain (Section 3 -- the class-A
pass is never modified, and the substitution at block 1 is a direct
override of its already-computed output, not a recomputation of block 1
itself). So per (pair, timestep, draw) this script runs ONE class-A pass
and ONE raw class-B pass to obtain H_1^A(i) and Delta_1(i) = H_1^B(i) -
H_1^A(i) (both cached), then for each gain in the locked grid reruns only
the class-B forward with block 1's hook overriding its output to
H_1^A(i) + g*Delta_1(i) -- mathematically identical to Section 3's
computation graph, just avoiding redundant identical A-pass recomputation
across the 6 gain values. At g=1.0 this override reduces exactly (to
floating-point precision) to H_1^B(i), which the script cross-checks
against the plain baseline pass as a built-in second identity check
(alongside the standalone A2 test already run and passed).

Aggregation-across-cells, a choice the pre-registration left implicit
(documented explicitly here so it is not a silent decision, per this
project's standing rule): Section 9 quotes pooled n=15 (5 pairs x 3
timesteps) baseline statistics for the decision table. This script
computes RecoveredMagnitude(g), InjectedDelta(g), and per-layer direction
consistency independently for each of the 15 (pair, timestep) cells (each
from its own 20 draws, per Section 2's exact Delta/consistency
definitions), then AVERAGES RecoveredMagnitude(g) and PropagationEfficiency(g)
across the 15 cells, and takes the per-layer MEAN across cells before
taking the min over layers 2-6 for direction consistency -- both are
simple unweighted means across the 15 equally-sized (20-draw) cells.
InjectedDelta(g) is defined in Item 1's own normalized-magnitude units
(mean_i(||delta||)/mean_i(||feat_A||), matching Section 9's L1=0.1346
citation) so that PropagationEfficiency(g) = RecoveredMagnitude(g) /
InjectedDelta(g) is a same-units ratio, as Section 5 requires.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
from layerwise_direction_probe import _register_layer_hooks, cosine_sim  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item2_localized_gain"
)

TARGET_BLOCK_IDX = 0  # block 1 (1-indexed) -- the block1->2 transition, per Item 2 v3 Sec. 3
GAIN_GRID = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]  # locked, Item 2 v3 Sec. 7
TIMESTEPS = [100, 500, 900]  # matches Item 1's own baseline + sensitivity reruns
K_DRAWS = 20
DIRECTION_FLOOR = 0.989
RECOVERY_SUPPORTED = 0.70
RECOVERY_PARTIAL = 0.30
POOLED_BLOCK1_TO_2_DROP = 0.0635  # Item 1 pooled n=15 stat, Tier0_Findings.md Sec. 9


class RawCaptureHook:
    """Captures block 1's raw, full per-token output tensor (1,600,D) --
    not mean-pooled -- so it can be reused to build every gain's override
    without recomputing block 1's forward for each gain."""
    def __init__(self):
        self.tensor: torch.Tensor | None = None

    def __call__(self, module, inp, out):
        self.tensor = out.detach().clone()
        return out


class OverrideHook:
    """Forward hook on model.blocks[TARGET_BLOCK_IDX]. When `.override` is
    set (a full (1,600,D) tensor), replaces block 1's output with it for
    that single forward call; when None, passes the block's real output
    through unmodified (used for the class-A / raw-capture passes)."""
    def __init__(self):
        self.override: torch.Tensor | None = None

    def __call__(self, module, inp, out):
        if self.override is not None:
            return self.override
        return out


def run_cell(model, device, n_leads, seq_len, t_val, y_a_val, y_b_val, gains):
    """One (pair, timestep) cell: K_DRAWS draws, baseline pass + one
    override pass per gain. Returns per-layer per-draw mean-pooled feature
    arrays for baseline A/B and, per gain, corrected B."""
    n_layers = len(model.blocks)
    baseline_a = [[] for _ in range(n_layers)]
    baseline_b = [[] for _ in range(n_layers)]
    corrected_b = {g: [[] for _ in range(n_layers)] for g in gains}
    g1_max_abs_diff = []  # built-in second identity check at g=1.0, all layers, all draws

    raw_hook = RawCaptureHook()
    override_hook = OverrideHook()

    for draw in range(K_DRAWS):
        torch.manual_seed(1000 + draw)
        x_t = torch.randn(1, n_leads, seq_len, device=device)
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        y_a = torch.full((1,), y_a_val, device=device, dtype=torch.long)
        y_b = torch.full((1,), y_b_val, device=device, dtype=torch.long)

        # Class-A pass: plain capture hooks + raw-capture hook on block 1 (never overridden).
        handles, captured = _register_layer_hooks(model)
        raw_handle = model.blocks[TARGET_BLOCK_IDX].register_forward_hook(raw_hook)
        with torch.no_grad():
            model(x_t, t, y_a)
        feat_a = {k: v.clone() for k, v in captured.items()}
        h1_a_full = raw_hook.tensor.clone()
        raw_handle.remove()
        for h in handles:
            h.remove()

        # Class-B raw pass (unmodified -- gives H_1^B(i) and hence Delta_1(i)).
        handles, captured = _register_layer_hooks(model)
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

        # Corrected class-B passes, one per gain, overriding block 1's output.
        for g in gains:
            override_hook.override = h1_a_full + g * delta1_full
            handles, captured = _register_layer_hooks(model)
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


def magnitude_and_consistency(feat_a_draws, feat_x_draws):
    """Item 1's exact per-layer formulas (Section 2 of layerwise_direction_probe.py's
    docstring): normalized magnitude and direction consistency of delta = feat_x - feat_a
    across draws, for one layer."""
    fa = np.stack(feat_a_draws)
    fx = np.stack(feat_x_draws)
    deltas = fx - fa
    mean_delta = deltas.mean(axis=0)
    mean_base_norm = float(np.mean(np.linalg.norm(fa, axis=1)))
    magnitude = float(np.mean(np.linalg.norm(deltas, axis=1))) / (mean_base_norm + 1e-8)
    consistency = float(np.mean([cosine_sim(d, mean_delta) for d in deltas]))
    return magnitude, consistency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    log = get_logger("item2_gain_sweep", cfg=cfg)
    torch.manual_seed(0)

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(cfg.paths.outputs.models) / "diffusion_best.pt"
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(f"[BLOCKED] Checkpoint not found at {ckpt_path}. Run Experiment 1 first.")
        return

    model = loaded.model
    device = loaded.device
    n_classes = loaded.n_classes
    n_layers = len(model.blocks)
    n_leads = 12
    seq_len = int(cfg.ptbxl.signal_length)

    identity_path = OUT_DIR / "identity_regression_test.json"
    if not identity_path.exists():
        print(f"[BLOCKED] {identity_path} not found -- run item2_localized_gain.py (A2) first.")
        return
    with open(identity_path) as f:
        identity_result = json.load(f)
    if not identity_result.get("all_pass"):
        print("[BLOCKED] A2 identity-regression test did not pass -- aborting per Item 2 v3 Sec. 6.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    class_pairs = [(0, cls_b) for cls_b in range(1, n_classes)]
    log.info(f"A3 sweep: gains={GAIN_GRID}, pairs={class_pairs}, timesteps={TIMESTEPS}, "
             f"k_draws={K_DRAWS}, n_layers={n_layers}")

    # per-gain accumulators across the 15 (pair,timestep) cells
    per_gain_recovered = {g: [] for g in GAIN_GRID}
    per_gain_injected = {g: [] for g in GAIN_GRID}
    per_gain_efficiency = {g: [] for g in GAIN_GRID}
    per_gain_layer_consistency = {g: [[] for _ in range(n_layers)] for g in GAIN_GRID}
    per_gain_layer_magnitude = {g: [[] for _ in range(n_layers)] for g in GAIN_GRID}
    baseline_layer_magnitude_all = [[] for _ in range(n_layers)]
    baseline_layer_consistency_all = [[] for _ in range(n_layers)]
    g1_identity_diffs_all = []
    per_cell_detail = []  # raw per-(pair,timestep) numbers, for the JSON dump

    for (y_a_val, y_b_val) in class_pairs:
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
            mag1_baseline = cell_baseline_mag[0]  # InjectedDelta base, Item 1 units, Sec. 5

            cell_detail = {
                "pair": f"0->{y_b_val}", "timestep": t_val,
                "baseline_magnitude_per_layer": cell_baseline_mag,
                "baseline_direction_consistency_per_layer": cell_baseline_cons,
                "gains": {},
            }

            for g in GAIN_GRID:
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
                injected = (g - 1.0) * mag1_baseline
                efficiency = (recovered / injected) if abs(injected) > 1e-12 else None

                per_gain_recovered[g].append(recovered)
                per_gain_injected[g].append(injected)
                if efficiency is not None:
                    per_gain_efficiency[g].append(efficiency)

                cell_detail["gains"][str(g)] = {
                    "magnitude_per_layer": cell_mag,
                    "direction_consistency_per_layer": cell_cons,
                    "recovered_magnitude_block6": recovered,
                    "injected_delta": injected,
                    "propagation_efficiency": efficiency,
                }
            per_cell_detail.append(cell_detail)

    max_g1_diff = max(g1_identity_diffs_all) if g1_identity_diffs_all else None
    log.info(f"Built-in g=1.0 identity re-check (override-hook path): max|diff| across all "
             f"cells/draws/layers = {max_g1_diff:.2e}" if max_g1_diff is not None else "no g=1 data")

    avg_baseline_magnitude = [float(np.mean(v)) for v in baseline_layer_magnitude_all]
    avg_baseline_consistency = [float(np.mean(v)) for v in baseline_layer_consistency_all]

    summary_rows = []
    per_gain_json_paths = []
    for g in GAIN_GRID:
        recovered_avg = float(np.mean(per_gain_recovered[g]))
        injected_avg = float(np.mean(per_gain_injected[g])) if per_gain_injected[g] else None
        efficiency_avg = float(np.mean(per_gain_efficiency[g])) if per_gain_efficiency[g] else None
        layer_cons_avg = [float(np.mean(v)) for v in per_gain_layer_consistency[g]]
        layer_mag_avg = [float(np.mean(v)) for v in per_gain_layer_magnitude[g]]
        min_direction_consistency = min(layer_cons_avg[1:])  # layers 2-6, Sec. 9

        recovery_fraction = recovered_avg / POOLED_BLOCK1_TO_2_DROP

        if min_direction_consistency < DIRECTION_FLOOR:
            verdict = "MAGNITUDE-AT-EXPENSE-OF-INTEGRITY"
        elif recovery_fraction >= RECOVERY_SUPPORTED:
            verdict = "SUPPORTED"
        elif recovery_fraction >= RECOVERY_PARTIAL:
            verdict = "PARTIAL SUPPORT"
        else:
            verdict = "REJECTED"

        summary_rows.append({
            "gain": g,
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
            "gain": g,
            "n_pairs": len(class_pairs), "n_timesteps": len(TIMESTEPS), "k_draws": K_DRAWS,
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
        per_gain_json_paths.append(gpath)
        log.info(f"gain={g}: recovery_fraction={recovery_fraction:.4f} "
                 f"min_dir_cons={min_direction_consistency:.4f} "
                 f"efficiency={efficiency_avg} verdict={verdict}")

    # Monotonicity check on Block 6 recovery% across the sweep (authorized gate addition).
    recoveries = [row["recovery_fraction"] for row in summary_rows]
    directions = [row["min_direction_consistency_layers_2_6"] for row in summary_rows]
    efficiencies = [row["propagation_efficiency_avg"] for row in summary_rows]
    monotonic = all(recoveries[i] <= recoveries[i + 1] for i in range(len(recoveries) - 1))
    monotonicity_flags = []
    if not monotonic:
        for i in range(1, len(recoveries)):
            if recoveries[i] < recoveries[i - 1]:
                dir_dips = directions[i] < directions[i - 1]
                eff_dips = (efficiencies[i] is not None and efficiencies[i - 1] is not None
                            and efficiencies[i] < efficiencies[i - 1])
                correlated = dir_dips and eff_dips
                monotonicity_flags.append({
                    "gain_from": GAIN_GRID[i - 1], "gain_to": GAIN_GRID[i],
                    "recovery_drop": recoveries[i - 1] - recoveries[i],
                    "direction_also_dipped": dir_dips, "efficiency_also_dipped": eff_dips,
                    "classification": "plausible_nonlinear_interaction" if correlated else "BUG_SIGNATURE",
                })

    bug_signature = any(f["classification"] == "BUG_SIGNATURE" for f in monotonicity_flags)

    # Other sanity gates (per standing authorization).
    direction_gate_fail = [row for row in summary_rows if row["min_direction_consistency_layers_2_6"] < 0.0]  # sanity, not the 0.989 pass/fail line
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
        "efficiency_gate_fail_gt_100pct_or_negative": [row["gain"] for row in efficiency_gate_fail],
        "nan_found": nan_found,
        "g1_builtin_identity_recheck_max_abs_diff": max_g1_diff,
    }, indent=2, default=str))

    if bug_signature or nan_found:
        print("[STOP] A gate condition fired (bug-signature monotonicity or NaN). "
              "Do not proceed to plots/report -- investigate first.")
    else:
        print("[OK] All gates clear. Proceed to plots + report generation.")


if __name__ == "__main__":
    main()
