"""
Stage 2 / Tier 0 Item 5 -- AdaLN/FiLM Parameter Statistics.

Implements the locked pre-registration
(Roadmap/Stage_2_Architecture_Investigation/Reports/Item5_PreRegistration.md).
Pure weight inspection -- no forward pass, no data, no draws. Confirmed
by direct source read (step04_transformer_diffusion.py:169,171-172):
adaLN is nn.Linear(2*model_dim, 4*model_dim), and torch.chunk(4, dim=-1)
on its output splits into 4 contiguous row-blocks of adaLN.weight, in
order: shift1, scale1, shift2, scale2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common.io import load_config, get_logger, load_model_checkpoint  # noqa: E402
from common.plotting import plot_scale_shift_fraction_vs_block  # noqa: E402

OUT_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Outputs"
    / "stage2_tier0_item5_adaln_statistics"
)
FIG_DIR = (
    REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Figures"
    / "stage2_tier0_item5_adaln_statistics"
)
REPORT_DIR = REPO_ROOT / "Roadmap" / "Stage_2_Architecture_Investigation" / "Reports"

CHUNK_NAMES = ["shift1", "scale1", "shift2", "scale2"]


def main() -> None:
    cfg = load_config()
    log = get_logger("item5_adaln_statistics", cfg=cfg)

    loaded = load_model_checkpoint(cfg)
    if loaded is None:
        print("[BLOCKED] Checkpoint not found. Run Experiment 1 first.")
        return

    model = loaded.model
    n_blocks = len(model.blocks)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Internal consistency audit (Phase E), before analysis ---
    model_dim = model.blocks[0].adaLN.weight.shape[1] // 2
    audit_ok = True
    for k, block in enumerate(model.blocks):
        w = block.adaLN.weight
        b = block.adaLN.bias
        expected_w_shape = (4 * model_dim, 2 * model_dim)
        expected_b_shape = (4 * model_dim,)
        if tuple(w.shape) != expected_w_shape or tuple(b.shape) != expected_b_shape:
            log.error(f"Block {k}: adaLN shape mismatch -- W={tuple(w.shape)} "
                      f"(expected {expected_w_shape}), b={tuple(b.shape)} "
                      f"(expected {expected_b_shape})")
            audit_ok = False
    if not audit_ok:
        print("[STOP] Internal consistency audit FAILED -- adaLN weight shapes did not match "
              "the expected (4*model_dim, 2*model_dim) pattern confirmed by source read. "
              "Aborting -- do not trust any downstream numbers until this is resolved.")
        return
    log.info(f"Internal consistency audit: PASS -- all {n_blocks} blocks' adaLN weight/bias "
             f"shapes match the expected (4*{model_dim}, 2*{model_dim}) / (4*{model_dim},) pattern.")

    rows = []
    per_block_detail = {}
    for k, block in enumerate(model.blocks):
        w = block.adaLN.weight.detach()
        b = block.adaLN.bias.detach()
        chunk_size = w.shape[0] // 4

        w_chunks = {name: w[i * chunk_size:(i + 1) * chunk_size, :] for i, name in enumerate(CHUNK_NAMES)}
        b_chunks = {name: b[i * chunk_size:(i + 1) * chunk_size] for i, name in enumerate(CHUNK_NAMES)}

        w_full_norm = float(w.norm().item())
        w_chunk_norms = {name: float(t.norm().item()) for name, t in w_chunks.items()}
        b_chunk_norms = {name: float(t.norm().item()) for name, t in b_chunks.items()}

        # Sum-of-squares check: disjoint row-blocks' squared Frobenius norms must sum to the
        # full matrix's squared Frobenius norm exactly (Phase D verification, not assumed).
        sum_sq_chunks = sum(v ** 2 for v in w_chunk_norms.values())
        full_sq = w_full_norm ** 2
        recon_diff = abs(sum_sq_chunks - full_sq)
        recon_rel_diff = recon_diff / full_sq if full_sq > 0 else 0.0

        scale_sq = w_chunk_norms["scale1"] ** 2 + w_chunk_norms["scale2"] ** 2
        shift_sq = w_chunk_norms["shift1"] ** 2 + w_chunk_norms["shift2"] ** 2
        scale_fraction = scale_sq / full_sq
        shift_fraction = shift_sq / full_sq

        rows.append({
            "block": k + 1,
            "full_frobenius_norm": w_full_norm,
            "scale_fraction": scale_fraction,
            "shift_fraction": shift_fraction,
            "shift1_norm": w_chunk_norms["shift1"], "scale1_norm": w_chunk_norms["scale1"],
            "shift2_norm": w_chunk_norms["shift2"], "scale2_norm": w_chunk_norms["scale2"],
            "bias_shift1_norm": b_chunk_norms["shift1"], "bias_scale1_norm": b_chunk_norms["scale1"],
            "bias_shift2_norm": b_chunk_norms["shift2"], "bias_scale2_norm": b_chunk_norms["scale2"],
        })
        per_block_detail[k + 1] = {
            "reconstruction_check_abs_diff": recon_diff,
            "reconstruction_check_rel_diff": recon_rel_diff,
        }
        log.info(f"Block {k+1}: ||W||_F={w_full_norm:.4f} scale_fraction={scale_fraction:.4f} "
                 f"shift_fraction={shift_fraction:.4f} (reconstruction rel diff={recon_rel_diff:.2e})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "adaln_statistics.csv", index=False)

    max_recon_rel_diff = max(v["reconstruction_check_rel_diff"] for v in per_block_detail.values())
    log.info(f"Max reconstruction relative diff across all blocks: {max_recon_rel_diff:.2e} "
             f"(verifies disjoint row-block squared-norms sum exactly to the full matrix's "
             f"squared Frobenius norm, per the pre-registration's quadrature-sum claim)")

    with open(OUT_DIR / "adaln_statistics.json", "w") as f:
        json.dump({
            "n_blocks": n_blocks,
            "model_dim": model_dim,
            "per_block": rows,
            "reconstruction_check": per_block_detail,
            "max_reconstruction_rel_diff": max_recon_rel_diff,
        }, f, indent=2)

    fig_path = plot_scale_shift_fraction_vs_block(df, FIG_DIR)

    scale_fracs = df["scale_fraction"].tolist()
    frac_range = max(scale_fracs) - min(scale_fracs)
    verdict = ("VERIFIED -- non-uniform scale/shift allocation across blocks"
               if frac_range > 0.02 else
               "FALSIFIED -- allocation is essentially uniform across blocks (range <= 2pp)")

    print(json.dumps({"per_block": rows, "verdict": verdict,
                      "max_reconstruction_rel_diff": max_recon_rel_diff}, indent=2, default=str))
    print(f"\nScale-fraction range across blocks: {frac_range:.4f} -- {verdict}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
