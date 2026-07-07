"""
mentor_eval/test_hausdorff_distance.py -- unit tests + the four
pre-registered verification cases for hausdorff_distance.py.

Run:
    pytest mentor_eval/test_hausdorff_distance.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from mentor_eval.hausdorff_distance import (
    compute_hausdorff, compute_hausdorff_per_lead, matched_hausdorff, N_LEADS,
)

SIG_LEN = 1000


def _real_signal(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = rng.normal(0, 1, size=(SIG_LEN, N_LEADS)).astype(np.float32)
    return np.clip(sig, -4.0, 4.0)


# ── Pre-registered verification cases (Section 4, 2026-07-07) ──────────

def test_case1_identical_signals_gives_zero():
    real = _real_signal()
    assert compute_hausdorff(real, real) == 0.0


def test_case2_pure_time_shift_gives_zero_not_nonzero():
    """Documents a deliberate, verified consequence of the amplitude-only
    (order-blind) framing chosen in Section 2 -- NOT a bug. A circular
    time-shift does not change the SET of amplitude values a signal
    takes, only when they occur; since this metric ignores order by
    design, it cannot distinguish a time-shifted signal from an
    identical one. This is the metric's stated limitation (does not
    capture temporal/morphological misalignment), made concrete."""
    real = _real_signal()
    shifted = np.roll(real, shift=50, axis=0)
    assert compute_hausdorff(real, shifted) == 0.0


def test_case3_amplitude_scale_gives_nonzero():
    real = _real_signal()
    scaled = np.clip(real * 1.5, -4.0, 4.0)
    result = compute_hausdorff(real, scaled)
    assert result > 0.0
    assert result < 8.0  # within the theoretical [0,8] bound


def test_case4_real_vs_matched_amplitude_noise_gives_nonzero_and_discriminates():
    real = _real_signal()
    rng = np.random.default_rng(1)
    noise = np.clip(rng.normal(0, real.std(), size=(SIG_LEN, N_LEADS)).astype(np.float32), -4.0, 4.0)
    result = compute_hausdorff(real, noise)
    assert result > 0.0


# ── Bounds and error handling ────────────────────────────────────────────

def test_bounded_by_clip_range():
    """Worst case: one signal pinned at -4, the other at +4 -- must not
    exceed 8.0 (the theoretical bound given config.yaml's clip_range)."""
    low = np.full((SIG_LEN, N_LEADS), -4.0)
    high = np.full((SIG_LEN, N_LEADS), 4.0)
    assert compute_hausdorff(low, high) == pytest.approx(8.0)


def test_raises_on_shape_mismatch():
    real = _real_signal()
    wrong_shape = _real_signal()[:500]
    with pytest.raises(ValueError, match="same shape"):
        compute_hausdorff(real, wrong_shape)


def test_raises_on_wrong_lead_count():
    wrong_leads = np.zeros((SIG_LEN, 8))
    with pytest.raises(ValueError, match="12 leads"):
        compute_hausdorff(wrong_leads, wrong_leads)


def test_raises_on_nan_input():
    real = _real_signal()
    with_nan = real.copy()
    with_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_hausdorff(real, with_nan)


def test_raises_on_inf_input():
    real = _real_signal()
    with_inf = real.copy()
    with_inf[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        compute_hausdorff(real, with_inf)


def test_per_lead_returns_12_values():
    real = _real_signal()
    scaled = np.clip(real * 1.5, -4.0, 4.0)
    per_lead = compute_hausdorff_per_lead(real, scaled)
    assert per_lead.shape == (N_LEADS,)
    assert compute_hausdorff(real, scaled) == pytest.approx(per_lead.mean())


def test_symmetric():
    real = _real_signal()
    scaled = np.clip(real * 1.5, -4.0, 4.0)
    assert compute_hausdorff(real, scaled) == pytest.approx(compute_hausdorff(scaled, real))


# ── matched_hausdorff (batch, nearest-neighbour matched) ────────────────

def test_matched_hausdorff_real_vs_itself_is_zero():
    rng = np.random.default_rng(0)
    real = np.clip(rng.normal(0, 1, (50, SIG_LEN, N_LEADS)).astype(np.float32), -4.0, 4.0)
    assert matched_hausdorff(real, real[:20]) == 0.0


def test_matched_hausdorff_discriminates_noise():
    rng = np.random.default_rng(0)
    real = np.clip(rng.normal(0, 1, (50, SIG_LEN, N_LEADS)).astype(np.float32), -4.0, 4.0)
    noise = np.clip(rng.normal(0, 2, (20, SIG_LEN, N_LEADS)).astype(np.float32), -4.0, 4.0)
    assert matched_hausdorff(real, noise) > 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
