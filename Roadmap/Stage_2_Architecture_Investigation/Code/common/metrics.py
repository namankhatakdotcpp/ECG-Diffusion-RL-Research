"""
Stage 2 Tier 0 -- shared magnitude/direction-consistency metrics.

LIFTED (copied, not moved) from
stage2_tier0_item1_layerwise_magnitude_direction/layerwise_direction_probe.py
(`cosine_sim`) and stage2_tier0_item2_localized_gain/item2_gain_sweep.py
(`magnitude_and_consistency`) -- both originals untouched.
"""

from __future__ import annotations

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def magnitude_and_consistency(feat_a_draws, feat_x_draws) -> tuple[float, float]:
    """Item 1's exact per-layer formulas: normalized magnitude and direction
    consistency of delta = feat_x - feat_a across draws, for one layer.

    magnitude    = mean_i(||delta_i||) / mean_i(||feat_a_i||)
    consistency  = mean_i(cosine_sim(delta_i, mean_j(delta_j))), in [-1, 1]
    """
    fa = np.stack(feat_a_draws)
    fx = np.stack(feat_x_draws)
    deltas = fx - fa
    mean_delta = deltas.mean(axis=0)
    mean_base_norm = float(np.mean(np.linalg.norm(fa, axis=1)))
    magnitude = float(np.mean(np.linalg.norm(deltas, axis=1))) / (mean_base_norm + 1e-8)
    consistency = float(np.mean([cosine_sim(d, mean_delta) for d in deltas]))
    return magnitude, consistency


def residual_update_ratio(block_input_draws, block_output_draws) -> float:
    """Item 3's R_k, per Item3_PreRegistration.md's Formal Definition
    section: the WITHIN-PASS residual update ratio at one block, for one
    class, across draws --

        DeltaH_k(i) = H_k^out(i) - H_k^in(i)
        R_k(i)      = ||DeltaH_k(i)|| / ||H_k^in(i)||
        R_k         = mean_i( R_k(i) )   (this function's return value)

    Deliberately a SIBLING to magnitude_and_consistency, not a reuse of
    it: that function computes a CROSS-CLASS delta (feat_x - feat_a,
    two different class labels, same block); this one computes a
    WITHIN-PASS delta (block_output - block_input, one class label, two
    different points in the same forward pass). Combines both residual
    adds within block k (attention-sublayer and FFN-sublayer) as a
    single quantity, per the pre-registration's explicit scope statement
    -- sub-block decomposition is out of scope for Item 3.
    """
    h_in = np.stack(block_input_draws)
    h_out = np.stack(block_output_draws)
    delta = h_out - h_in
    per_draw_ratio = np.linalg.norm(delta, axis=1) / (np.linalg.norm(h_in, axis=1) + 1e-8)
    return float(np.mean(per_draw_ratio))
