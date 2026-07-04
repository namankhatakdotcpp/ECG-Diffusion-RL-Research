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
