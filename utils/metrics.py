"""
utils/metrics.py — evaluation metric skeletons for ECG generation and classification.

Metrics implemented:
    dtw_distance   — Dynamic Time Warping distance between two 1-D signals
    mmd_rbf        — Maximum Mean Discrepancy (RBF kernel) between sample sets
    per_class_f1   — Per-class F1 scores for multi-label classification
    tstr_score     — Train-on-Synthetic Test-on-Real aggregate score
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Dynamic Time Warping
# ---------------------------------------------------------------------------

def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the DTW distance between two 1-D time series.

    Uses the ``tslearn`` library for an efficient implementation.  Falls back
    to a pure-NumPy O(n²) implementation when tslearn is unavailable.

    Args:
        x: 1-D array of shape (T,).
        y: 1-D array of shape (T,).

    Returns:
        Scalar DTW distance.
    """
    try:
        from tslearn.metrics import dtw  # type: ignore
        return float(dtw(x, y))
    except ImportError:
        return _dtw_numpy(x, y)


def _dtw_numpy(x: np.ndarray, y: np.ndarray) -> float:
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (RBF kernel)
# ---------------------------------------------------------------------------

def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: Optional[float] = None,
) -> float:
    """Estimate MMD² between two sample sets using an RBF kernel.

    Args:
        X: Real samples, shape (N, D).
        Y: Generated samples, shape (M, D).
        sigma: RBF bandwidth. Defaults to the median heuristic.

    Returns:
        Scalar MMD² estimate (≥ 0; lower is better, 0 means identical distributions).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if sigma is None:
        # Median heuristic over pairwise distances in the combined set
        Z = np.vstack([X, Y])
        dists = np.sum((Z[:, None] - Z[None, :]) ** 2, axis=-1)
        sigma = float(np.sqrt(np.median(dists[dists > 0]) / 2))
        sigma = max(sigma, 1e-8)

    gamma = 1.0 / (2.0 * sigma ** 2)

    def rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq_dists = np.sum((A[:, None] - B[None, :]) ** 2, axis=-1)
        return np.exp(-gamma * sq_dists)

    K_XX = rbf(X, X)
    K_YY = rbf(Y, Y)
    K_XY = rbf(X, Y)

    n, m = len(X), len(Y)
    mmd2 = (
        (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
        + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
        - 2.0 * K_XY.mean()
    )
    return float(mmd2)


# ---------------------------------------------------------------------------
# Per-class F1
# ---------------------------------------------------------------------------

def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute per-class F1 for multi-label (sigmoid) predictions.

    Args:
        y_true: Ground-truth binary labels, shape (N, C).
        y_pred: Model output probabilities, shape (N, C).
        class_names: Optional list of class name strings, length C.
        threshold: Decision threshold for converting probabilities → labels.

    Returns:
        Dict mapping class name → F1 score.
    """
    y_pred_bin = (np.asarray(y_pred) >= threshold).astype(int)
    y_true = np.asarray(y_true, dtype=int)

    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    scores = f1_score(y_true, y_pred_bin, average=None, zero_division=0)
    return {name: float(s) for name, s in zip(class_names, scores)}


# ---------------------------------------------------------------------------
# TSTR Score
# ---------------------------------------------------------------------------

def tstr_score(
    real_f1: dict[str, float],
    synthetic_f1: dict[str, float],
) -> float:
    """Compute the TSTR (Train-on-Synthetic, Test-on-Real) macro-F1 ratio.

    A ratio close to 1.0 means the synthetic data is as useful as real data
    for training a downstream classifier.

    Args:
        real_f1:      Per-class F1 when training on *real* data.
        synthetic_f1: Per-class F1 when training on *synthetic* data.

    Returns:
        Scalar ratio: mean(synthetic_f1) / mean(real_f1).
        Values > 1 are possible if synthetic augmentation improves results.
    """
    real_macro = float(np.mean(list(real_f1.values())))
    synth_macro = float(np.mean(list(synthetic_f1.values())))
    if real_macro == 0:
        return 0.0
    return synth_macro / real_macro
