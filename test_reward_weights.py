"""
test_reward_weights.py -- proves cfg.reward.weights actually controls
ClinicalReward's combined output.

This is the specific regression this test exists to catch: get_reward()
was previously found to silently ignore cfg.reward.weights and always use
a hardcoded ABLATION_CONFIGS["full"] dict instead (see
Roadmap/Stage_4_Optimization/Decisions.md). Fixed once already -- this
test exists so it can't quietly break again.

Uses fake component objects (fixed .compute() return values) instead of
real MorphologyReward/HRVReward/RealismReward/DiagnosticUtilityReward/
A3Reward, so this only tests the weighted-combination arithmetic in
ClinicalReward.compute(), not any real reward computation -- no GPU, no
PTB-XL data, no trained classifier needed.

Run:
    pytest test_reward_weights.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from step06_reward_function import ClinicalReward


class _FakeComponent:
    """Returns a fixed scalar regardless of input -- stands in for any of
    the five real reward components (all share a `.compute(...)` interface
    that returns a float in [0, 1])."""

    def __init__(self, value: float):
        self.value = value

    def compute(self, *args, **kwargs) -> float:
        return self.value


DUMMY_ECG = np.zeros((1000, 12), dtype=np.float32)


def _build_reward(weights: dict[str, float], values: dict[str, float]) -> ClinicalReward:
    return ClinicalReward(
        morph_reward=_FakeComponent(values.get("morph", 0.0)),
        hrv_reward=_FakeComponent(values.get("hrv", 0.0)),
        real_reward=_FakeComponent(values.get("real", 0.0)),
        diag_reward=_FakeComponent(values.get("diag", 0.0)),
        a3_reward=_FakeComponent(values.get("a3", 0.0)),
        weights=weights,
        class_names=["NORM", "MI"],
    )


def test_diag_only_weight_isolates_diag_term():
    """weights={diag: 1, everything else: 0} -> total == r_diag alone."""
    values = {"morph": 0.9, "hrv": 0.8, "real": 0.7, "diag": 0.42, "a3": 0.6}
    weights = {"morph": 0.0, "hrv": 0.0, "real": 0.0, "diag": 1.0, "a3": 0.0}
    reward = _build_reward(weights, values)

    result = reward.compute(DUMMY_ECG, "NORM", 0)

    assert result["total"] == pytest.approx(0.42)
    assert result["r_diag"] == pytest.approx(0.42)


def test_a3_only_weight_isolates_a3_term():
    """weights={a3: 1, everything else: 0} -> total == r_a3 alone."""
    values = {"morph": 0.9, "hrv": 0.8, "real": 0.7, "diag": 0.5, "a3": 0.37}
    weights = {"morph": 0.0, "hrv": 0.0, "real": 0.0, "diag": 0.0, "a3": 1.0}
    reward = _build_reward(weights, values)

    result = reward.compute(DUMMY_ECG, "NORM", 0)

    assert result["total"] == pytest.approx(0.37)
    assert result["r_a3"] == pytest.approx(0.37)


def test_zero_weight_component_has_no_effect_on_total():
    """A component's own score should never leak into the total when its
    weight is 0 -- catches a get_reward()-style bug where weights are
    silently ignored / a hardcoded default sneaks back in."""
    values = {"morph": 1.0, "hrv": 1.0, "real": 1.0, "diag": 0.5, "a3": 1.0}
    weights = {"morph": 0.0, "hrv": 0.0, "real": 0.0, "diag": 1.0, "a3": 0.0}
    reward = _build_reward(weights, values)

    result = reward.compute(DUMMY_ECG, "NORM", 0)

    # If morph/hrv/real/a3's weight-0 values (all 1.0) leaked in at all,
    # total would be > 0.5.
    assert result["total"] == pytest.approx(0.5)


def test_weighted_sum_matches_manual_calculation():
    """A genuinely mixed weight vector — the arithmetic itself, not just
    the isolation cases."""
    values = {"morph": 0.2, "hrv": 0.4, "real": 0.6, "diag": 0.8, "a3": 0.5}
    weights = {"morph": 0.16, "hrv": 0.10, "real": 0.16, "diag": 0.37, "a3": 0.21}
    reward = _build_reward(weights, values)

    expected = sum(weights[k] * values[k] for k in weights)
    result = reward.compute(DUMMY_ECG, "NORM", 0)

    assert result["total"] == pytest.approx(expected)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
