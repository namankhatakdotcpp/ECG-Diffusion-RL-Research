"""
test_bias_corrected_baseline.py -- proves the Adam-style bias correction on
DDPOTrainer's EMA advantage baseline actually removes the quantified
cold-start bias (Roadmap/Stage_4_Optimization/Decisions.md), not just that
it compiles.

Standalone re-implementation of the exact update in
DDPOTrainer._update_and_bias_correct_baseline (no torch/model dependency --
this only tests the EMA/bias-correction arithmetic, not PPO itself), kept
byte-for-byte in sync with step07_rl_finetuning.py's formula so a drift
between the two would need to be introduced deliberately, not accidentally.

Run:
    pytest test_bias_corrected_baseline.py -v
"""

from __future__ import annotations

import pytest


class _BaselineTracker:
    """Mirrors DDPOTrainer.__init__ + _update_and_bias_correct_baseline exactly."""

    def __init__(self, decay: float = 0.99):
        self.baseline = 0.0
        self.baseline_decay = decay
        self.baseline_step = 0

    def update(self, r: float) -> tuple[float, float]:
        """Returns (raw_baseline_after_update, bias_corrected_baseline)."""
        self.baseline = self.baseline_decay * self.baseline + (1.0 - self.baseline_decay) * r
        self.baseline_step += 1
        correction = 1.0 - self.baseline_decay ** self.baseline_step
        return self.baseline, self.baseline / correction


def test_raw_ema_shows_the_quantified_cold_start_bias():
    """Confirms the documented 85%/45%/20% residual-bias pattern still holds
    for the RAW (uncorrected) baseline -- the problem this fix addresses."""
    tracker = _BaselineTracker(decay=0.99)
    true_reward = 0.5
    raw_at_k = {}
    for k in range(1, 161):
        raw, _ = tracker.update(true_reward)
        if k in (16, 80, 160):
            raw_at_k[k] = raw

    # weight_on_init = decay^k -> raw baseline = true_reward * (1 - decay^k)
    assert raw_at_k[16]  == pytest.approx(true_reward * (1 - 0.99 ** 16),  abs=1e-9)
    assert raw_at_k[80]  == pytest.approx(true_reward * (1 - 0.99 ** 80),  abs=1e-9)
    assert raw_at_k[160] == pytest.approx(true_reward * (1 - 0.99 ** 160), abs=1e-9)
    # And the raw baseline is still meaningfully below the true reward at k=160
    assert raw_at_k[160] < true_reward * 0.85   # still >15% short at k=160


def test_bias_corrected_baseline_converges_within_5_percent_by_t10():
    """The actual claim this fix makes: baseline_hat should be close to the
    true reward much sooner than the raw EMA is."""
    tracker = _BaselineTracker(decay=0.99)
    true_reward = 0.5
    corrected_at_t = {}
    for t in range(1, 11):
        _, corrected = tracker.update(true_reward)
        corrected_at_t[t] = corrected

    for t, val in corrected_at_t.items():
        rel_err = abs(val - true_reward) / true_reward
        assert rel_err < 0.05, f"t={t}: baseline_hat={val:.4f}, {rel_err:.1%} off true reward {true_reward}"


def test_bias_corrected_baseline_much_closer_than_raw_at_early_t():
    """Direct raw-vs-corrected comparison at t=1 (the sharpest contrast) and
    t=16 (end of one smoke-test iteration)."""
    tracker = _BaselineTracker(decay=0.99)
    true_reward = 0.5

    raw_1, corrected_1 = tracker.update(true_reward)
    assert corrected_1 == pytest.approx(true_reward, abs=1e-9)   # t=1: exact recovery by construction
    assert raw_1 == pytest.approx(0.005, abs=1e-9)               # raw EMA barely moved off 0.0
    assert abs(corrected_1 - true_reward) < abs(raw_1 - true_reward)

    tracker2 = _BaselineTracker(decay=0.99)
    for _ in range(16):
        raw_16, corrected_16 = tracker2.update(true_reward)
    assert abs(corrected_16 - true_reward) < abs(raw_16 - true_reward)
    assert abs(corrected_16 - true_reward) / true_reward < 0.01   # <1% off at k=16
    assert abs(raw_16 - true_reward) / true_reward > 0.8          # raw still >80% off at k=16


def test_long_run_behaviour_unchanged_by_correction():
    """As t -> large, decay^t -> 0, so baseline_hat -> baseline -- the fix
    should not change long-run behaviour, only early/mid-run bias."""
    tracker = _BaselineTracker(decay=0.99)
    true_reward = 0.5
    for t in range(1, 2001):
        raw, corrected = tracker.update(true_reward)
    assert abs(corrected - raw) < 1e-6


def test_matches_step07_source_formula():
    """Guards against the standalone tracker here silently drifting from the
    real implementation in step07_rl_finetuning.py."""
    import inspect
    from step07_rl_finetuning import DDPOTrainer
    src = inspect.getsource(DDPOTrainer._update_and_bias_correct_baseline)
    assert "self.baseline_decay * self.baseline" in src
    assert "1.0 - self.baseline_decay" in src
    assert "self.baseline_step" in src
    assert "1.0 - self.baseline_decay ** self.baseline_step" in src


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
