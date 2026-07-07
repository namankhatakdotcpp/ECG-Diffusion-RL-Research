"""
mentor_eval/hausdorff_distance.py — amplitude-only Hausdorff distance
between real and generated ECGs, per lead, mean-aggregated across leads.

Mathematical definition (see 2026-07-07 pre-registration discussion):
each lead's signal is treated as an UNORDERED 1D point set of amplitude
values (order-blind -- NOT a (time, amplitude) 2D trajectory). For two
sets A, B:

    h(A, B) = sup_{a in A} inf_{b in B} |a - b|
    H(A, B) = max(h(A, B), h(B, A))

Deliberately amplitude-only, not (time, amplitude): a 2D framing would
mix time-index units (range ~1000) with amplitude units (range 8, post
z-score-and-clip per config.yaml's [-4.0, 4.0]) and would need an
arbitrary, unjustified time/amplitude relative-scaling constant that has
no precedent anywhere else in this pipeline (Mahalanobis/Bhattacharyya
operate on summary statistics with no time axis; cosine similarity is a
fixed-index dot product with no explicit temporal-tolerance mechanism
either). Amplitude-only avoids that problem and is unit-consistent by
construction. This complements (does not duplicate) the existing
similarity_metrics.py trio: it captures worst-case per-lead amplitude-
range/outlier mismatch, which none of cosine/Mahalanobis/Bhattacharyya
directly measure.

Explicit limitation, stated rather than hidden: this metric is ORDER-
BLIND -- it does not capture morphology or timing (where peaks occur).
Two signals with identical amplitude range but completely different
shape would score as similar. Report this alongside any number.

Bounds: given the project's fixed z-score-and-clip-to-[-4,4] preprocessing
convention (config.yaml: preprocessing.clip_range), every per-lead value
is bounded in [0, 8] (worst case: one signal pinned at -4, the other at
+4 throughout). For a well-matched real/generated pair of the same
class, most per-lead values should be well under 1.0; values approaching
2-4 indicate a real amplitude-range mismatch worth flagging, not just
reporting as a bare number.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import directed_hausdorff

N_LEADS = 12


def _hausdorff_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two 1D point sets (amplitude
    values from one lead). directed_hausdorff only computes one direction
    (u -> v); Hausdorff distance requires the max of both directions."""
    a2 = a.reshape(-1, 1)
    b2 = b.reshape(-1, 1)
    d_ab = directed_hausdorff(a2, b2)[0]
    d_ba = directed_hausdorff(b2, a2)[0]
    return float(max(d_ab, d_ba))


def compute_hausdorff_per_lead(real_signal: np.ndarray, generated_signal: np.ndarray) -> np.ndarray:
    """real_signal, generated_signal: (1000, 12) arrays (this project's
    fixed signal-length convention -- see config.yaml's signal_length).
    Returns a (12,) array, one amplitude-only Hausdorff distance per lead.

    Raises ValueError on shape mismatch or non-finite input rather than
    silently skipping/imputing -- a silently-dropped sample changes the
    point SET being compared without telling the caller, and this
    project's convention elsewhere (subband_similarity_metrics.py,
    checkpoint_utils.py) is to filter/report non-finite data explicitly
    upstream of any metric, not inside it. Unequal-length signals are not
    supported -- every signal produced by this pipeline is fixed at
    1000 samples (ECGDataset's own convention); if that ever changes,
    this should fail loudly rather than silently resample."""
    real_signal = np.asarray(real_signal)
    generated_signal = np.asarray(generated_signal)

    if real_signal.shape != generated_signal.shape:
        raise ValueError(
            f"real_signal and generated_signal must have the same shape, "
            f"got {real_signal.shape} vs {generated_signal.shape}. "
            f"Unequal-length signals are not supported by this function."
        )
    if real_signal.shape[-1] != N_LEADS:
        raise ValueError(f"expected {N_LEADS} leads (last dim), got shape {real_signal.shape}")
    if not np.isfinite(real_signal).all() or not np.isfinite(generated_signal).all():
        raise ValueError(
            "real_signal/generated_signal contains non-finite values (NaN/Inf) -- "
            "filter these upstream before calling compute_hausdorff, same "
            "convention as subband_similarity_metrics.py's np.isfinite(sig).all() check."
        )

    return np.array([
        _hausdorff_1d(real_signal[:, lead], generated_signal[:, lead])
        for lead in range(N_LEADS)
    ])


def compute_hausdorff(real_signal: np.ndarray, generated_signal: np.ndarray) -> float:
    """Mean across the 12 per-lead Hausdorff distances -- single-number
    summary for a comparison table. Use compute_hausdorff_per_lead()
    directly for the per-lead breakdown."""
    return float(compute_hausdorff_per_lead(real_signal, generated_signal).mean())
