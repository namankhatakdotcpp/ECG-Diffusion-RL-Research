"""
test_reliability_alpha.py -- proves DiagnosticUtilityReward's reliability_alpha
interpolation is correct at both endpoints and monotonic in between.

Context: reliability scaling used to be a boolean (use_reliability_scaling),
either the full per-class F1 vector or all-ones. It's now a continuous
`reliability_alpha` in [0, 1] so the reliability-weight validation study
(Roadmap/Stage_4_Optimization/Reward_Design_v2.md Section 4a) can sweep
100/75/50/25/0% settings instead of only the two extremes.

alpha=1.0 must be a true no-op relative to the old use_reliability=True
behavior -- if this drifts, every sweep arm becomes uninterpretable relative
to the current production baseline.

Uses a nonexistent classifier_path (classifier loading is irrelevant here --
the reliability array is computed unconditionally regardless of whether the
CNN itself loaded) and a temp eval JSON, so no GPU/trained model is needed.

Run:
    pytest test_reliability_alpha.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from step06_reward_function import DiagnosticUtilityReward

PER_CLASS_F1 = [0.792, 0.645, 0.612, 0.585, 0.376, 0.571]  # NORM/MI/STTC/CD/HYP/OTHER
N_CLASSES = len(PER_CLASS_F1)
HYP_IDX = 4


def _build(tmp_path, alpha: float) -> DiagnosticUtilityReward:
    eval_path = tmp_path / "trtr_classifier_eval.json"
    eval_path.write_text(json.dumps({"per_class_f1": PER_CLASS_F1}))
    return DiagnosticUtilityReward(
        classifier_path=str(tmp_path / "nonexistent_classifier.pt"),
        n_classes=N_CLASSES,
        eval_path=str(eval_path),
        reliability_alpha=alpha,
    )


def test_alpha_1_0_is_true_no_op(tmp_path):
    """alpha=1.0 must reproduce the old use_reliability=True branch's output
    bit-identically -- this is the load-bearing test for the whole sweep."""
    reward = _build(tmp_path, alpha=1.0)

    expected = np.asarray(PER_CLASS_F1, dtype=float)
    assert np.array_equal(reward.reliability, expected)


def test_alpha_0_0_gives_uniform_reliability(tmp_path):
    """alpha=0.0 must give reliability=1.0 for every class, exactly."""
    reward = _build(tmp_path, alpha=0.0)

    assert np.array_equal(reward.reliability, np.ones(N_CLASSES))


def test_monotonic_interpolation_across_sweep_points(tmp_path):
    """For HYP (per_class_f1=0.376 < 1.0), reliability must increase
    monotonically as alpha decreases across the 5 sweep points -- catches a
    sign error in the interpolation formula before it reaches GPU time."""
    sweep_points = [1.00, 0.75, 0.50, 0.25, 0.00]
    hyp_reliability = [
        _build(tmp_path, alpha=a).reliability[HYP_IDX] for a in sweep_points
    ]

    # sweep_points is descending alpha, so hyp_reliability must be strictly increasing
    assert all(
        earlier < later
        for earlier, later in zip(hyp_reliability, hyp_reliability[1:])
    )
    assert hyp_reliability[0] == pytest.approx(0.376)
    assert hyp_reliability[-1] == pytest.approx(1.0)


def test_out_of_range_alpha_raises(tmp_path):
    with pytest.raises(ValueError):
        _build(tmp_path, alpha=1.5)
    with pytest.raises(ValueError):
        _build(tmp_path, alpha=-0.1)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
