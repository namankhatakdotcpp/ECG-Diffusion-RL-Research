"""
Stage 2 / Tier 0 Item 3 -- Residual-Path Attenuation.

Implements the locked pre-registration
(Roadmap/Stage_2_Architecture_Investigation/Reports/Item3_PreRegistration.md,
commit 3101f28). No intervention, no hook substitution -- pure measurement
on unmodified forward passes, same in spirit as Item 1's own probe.

Formal definition (pre-registration, "Formal definition" section):
    DeltaH_k(i) = H_k^out(i) - H_k^in(i)
    R_k(i)      = ||DeltaH_k(i)|| / ||H_k^in(i)||    (mean-pooled norms)
R_k measures the COMBINED effect of both residual adds within block k
(attention-sublayer + FFN-sublayer) -- sub-block decomposition is out of
scope, per the pre-registration's explicit scope statement.

Capture mechanism (dependency audit, restated): blocks 2-6's inputs are
free -- block k's output IS block k+1's input, bit-identical, since
cond_film is held constant across the block loop (source-verified,
step04_transformer_diffusion.py:257-282) -- captured via
common/hooks.py::register_layer_hooks (unchanged from Item 1). Block 1's
true input requires the ONE new addition,
common/hooks.py::register_block0_input_hook (a forward_pre_hook on
model.blocks[0]).

Experimental design inherited from Item 1 (stated explicitly, per the
pre-registration's Runtime section, not independently chosen): 5 class
pairs (0 vs 1..5) x 3 timesteps (100, 500, 900) x 20 draws
(torch.manual_seed(1000+draw)), same-seed/same-timestep/class-label-only-
differs pairs, per the master prompt's own Item 3 definition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

CODE_DIR = Path(__file__).resolve().parents[1]  # Roadmap/.../Code/
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.io import REPO_ROOT, load_config, get_logger, load_model_checkpoint  # noqa: E402
from common.hooks import register_layer_hooks, register_block0_input_hook  # noqa: E402
from common.metrics import residual_update_ratio  # noqa: E402
from common.utils import class_pairs, K_DRAWS, TIMESTEPS  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item3_residual_attenuation"
)
FIG_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Figures"
    / "stage2_tier0_item3_residual_attenuation"
)


def run_single_pass(model, x_t, t, y):
    """One forward pass for one class label. Captures H_k^in and H_k^out
    (mean-pooled) at every block 1-6. Block 1's input comes from the new
    pre-hook; blocks 2-6's inputs are the previous block's already-hooked
    output (free, per the dependency audit)."""
    handles, out_captured = register_layer_hooks(model)
    pre_handle, in0_captured = register_block0_input_hook(model)
    try:
        with torch.no_grad():
            model(x_t, t, y)
        block_outputs = {k: v.clone() for k, v in out_captured.items()}  # 0..5 (blocks 1-6)
        block0_input = in0_captured["block0_input"].clone()
    finally:
        pre_handle.remove()
        for h in handles:
            h.remove()

    n_layers = len(block_outputs)
    block_inputs = {0: block0_input}
    for k in range(1, n_layers):
        block_inputs[k] = block_outputs[k - 1]  # block k's input == block k-1's output
    return block_inputs, block_outputs


def main() -> None:
    cfg = load_config()
    log = get_logger("item3_residual_probe", cfg=cfg)
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
    log.info(f"Item 3 sweep: pairs={pairs}, timesteps={TIMESTEPS}, k_draws={K_DRAWS}, "
             f"n_layers={n_layers}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw = {"n_layers": n_layers, "k_draws": K_DRAWS, "timesteps": TIMESTEPS, "pairs": {}}
    # per-cell (pair, timestep) pooled R_k, per class, for the headline pooled curve
    r_k_A_cells = [[] for _ in range(n_layers)]  # class A (=0) -- same reference every pair
    r_k_B_cells = [[] for _ in range(n_layers)]  # class B (=cls_b) -- varies per pair

    for (y_a_val, y_b_val) in pairs:
        for t_val in TIMESTEPS:
            in_a_by_layer = [[] for _ in range(n_layers)]
            out_a_by_layer = [[] for _ in range(n_layers)]
            in_b_by_layer = [[] for _ in range(n_layers)]
            out_b_by_layer = [[] for _ in range(n_layers)]

            for draw in range(K_DRAWS):
                torch.manual_seed(1000 + draw)
                x_t = torch.randn(1, n_leads, seq_len, device=device)
                t = torch.full((1,), t_val, device=device, dtype=torch.long)
                y_a = torch.full((1,), y_a_val, device=device, dtype=torch.long)
                y_b = torch.full((1,), y_b_val, device=device, dtype=torch.long)

                in_a, out_a = run_single_pass(model, x_t, t, y_a)
                in_b, out_b = run_single_pass(model, x_t, t, y_b)

                for layer in range(n_layers):
                    in_a_by_layer[layer].append(in_a[layer][0].numpy())
                    out_a_by_layer[layer].append(out_a[layer][0].numpy())
                    in_b_by_layer[layer].append(in_b[layer][0].numpy())
                    out_b_by_layer[layer].append(out_b[layer][0].numpy())

            r_k_A = [residual_update_ratio(in_a_by_layer[layer], out_a_by_layer[layer])
                     for layer in range(n_layers)]
            r_k_B = [residual_update_ratio(in_b_by_layer[layer], out_b_by_layer[layer])
                     for layer in range(n_layers)]

            for layer in range(n_layers):
                r_k_A_cells[layer].append(r_k_A[layer])
                r_k_B_cells[layer].append(r_k_B[layer])

            raw["pairs"].setdefault(f"0->{y_b_val}", {})[str(t_val)] = {
                "R_k_class_A": r_k_A, "R_k_class_B": r_k_B,
            }
            log.info(f"Pair (0->{y_b_val}), t={t_val}: R_k(A)={[round(x, 4) for x in r_k_A]} "
                     f"R_k(B)={[round(x, 4) for x in r_k_B]}")

    # Pooled curves (n=15 cells, matching Item 1's own pooling convention)
    r_k_A_pooled = [float(np.mean(v)) for v in r_k_A_cells]
    r_k_B_pooled = [float(np.mean(v)) for v in r_k_B_cells]
    # Combined (both classes treated as independent observations of the same block-level quantity)
    r_k_combined_cells = [r_k_A_cells[layer] + r_k_B_cells[layer] for layer in range(n_layers)]
    r_k_combined_pooled = [float(np.mean(v)) for v in r_k_combined_cells]

    # Candidate statistical test (Wilcoxon signed-rank, matching Item 1's a089496 precedent):
    # paired comparison of block 1 vs block 6's pooled-per-cell R_k (n=15 pairs x timesteps,
    # each cell's class-A and class-B values kept separate as independent paired observations
    # -> n=30), testing whether the endpoints differ significantly (systematic variation vs. flat/noise).
    block1_vals = np.array(r_k_combined_cells[0])
    block6_vals = np.array(r_k_combined_cells[n_layers - 1])
    try:
        stat, p_value = wilcoxon(block1_vals, block6_vals)
    except ValueError as e:
        stat, p_value = None, None
        log.info(f"Wilcoxon test could not run: {e}")

    df = pd.DataFrame({
        "block": list(range(1, n_layers + 1)),
        "R_k_class_A_pooled": r_k_A_pooled,
        "R_k_class_B_pooled": r_k_B_pooled,
        "R_k_combined_pooled": r_k_combined_pooled,
    })
    df.to_csv(OUT_DIR / "residual_probe.csv", index=False)

    raw["pooled"] = {
        "R_k_class_A_pooled": r_k_A_pooled,
        "R_k_class_B_pooled": r_k_B_pooled,
        "R_k_combined_pooled": r_k_combined_pooled,
        "wilcoxon_block1_vs_block6": {
            "statistic": float(stat) if stat is not None else None,
            "p_value": float(p_value) if p_value is not None else None,
            "n": len(block1_vals),
        },
    }
    with open(OUT_DIR / "residual_probe_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    # Monotonicity / shape classification (descriptive, not a locked criterion -- Decision
    # criteria section explicitly defers thresholds to this real data).
    diffs = np.diff(r_k_combined_pooled)
    if np.all(diffs <= 0):
        shape = "monotonically declining (attenuation)"
    elif np.all(diffs >= 0):
        shape = "monotonically increasing (amplification)"
    else:
        shape = "non-monotonic"

    log.info(f"Pooled R_k (combined): {[round(x, 4) for x in r_k_combined_pooled]}")
    log.info(f"Shape classification: {shape}")
    log.info(f"Wilcoxon (block1 vs block6, n={len(block1_vals)}): "
             f"stat={stat}, p={p_value}")

    print(json.dumps({
        "R_k_class_A_pooled": r_k_A_pooled,
        "R_k_class_B_pooled": r_k_B_pooled,
        "R_k_combined_pooled": r_k_combined_pooled,
        "shape_classification": shape,
        "wilcoxon_block1_vs_block6": {"statistic": stat, "p_value": p_value, "n": len(block1_vals)},
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
