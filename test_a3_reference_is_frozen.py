"""
test_a3_reference_is_frozen.py -- proves A3Reward.compute() never mutates
its own reference statistics.

Closes the loop on the earlier manual grep-based verification (see
Roadmap/Stage_4_Optimization/Decisions.md) that a3_subband_stats.json's
per-class (mean, inv_cov) is fixed for the whole PPO run: if `compute()`
ever wrote to `self._cache`, the reward target would start moving toward
whatever the policy happens to generate — a more severe version of the
TSTR circularity already ruled out for DiagnosticUtilityReward. This test
makes that guarantee permanent (a future refactor that reintroduces a
mutation would fail this immediately) rather than relying on a one-time
grep.

Needs outputs/processed/a3_subband_stats.json (run
step03_eda_and_class_mapping.py first). No GPU/checkpoint needed.

Run:
    pytest test_a3_reference_is_frozen.py -v
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from step06_reward_function import A3Reward

A3_STATS_PATH = Path("outputs/processed/a3_subband_stats.json")


@pytest.mark.skipif(not A3_STATS_PATH.exists(), reason="a3_subband_stats.json not found — run step03 first")
def test_cache_unchanged_after_compute():
    a3_stats_file = json.load(open(A3_STATS_PATH))
    reward = A3Reward(a3_stats_file)

    cache_before = copy.deepcopy(reward._cache)

    rng = np.random.default_rng(0)
    fake_ecg = rng.normal(0, 1, size=(1000, 12)).astype(np.float32)
    target_class = next(iter(reward._cache))
    reward.compute(fake_ecg, target_class)

    cache_after = reward._cache

    assert set(cache_before.keys()) == set(cache_after.keys())
    for cls in cache_before:
        mean_before, inv_cov_before = cache_before[cls]
        mean_after, inv_cov_after = cache_after[cls]
        np.testing.assert_array_equal(mean_before, mean_after)
        np.testing.assert_array_equal(inv_cov_before, inv_cov_after)


@pytest.mark.skipif(not A3_STATS_PATH.exists(), reason="a3_subband_stats.json not found — run step03 first")
def test_cache_unchanged_after_many_computes_across_classes():
    """Same guarantee, but stress it across every class and several calls —
    a mutation bug that only shows up after N calls (e.g. an accidental
    running-average update) wouldn't be caught by a single call above."""
    a3_stats_file = json.load(open(A3_STATS_PATH))
    reward = A3Reward(a3_stats_file)
    cache_before = copy.deepcopy(reward._cache)

    rng = np.random.default_rng(1)
    for _ in range(20):
        for cls in reward._cache:
            fake_ecg = rng.normal(0, 1, size=(1000, 12)).astype(np.float32)
            reward.compute(fake_ecg, cls)

    for cls in cache_before:
        mean_before, inv_cov_before = cache_before[cls]
        mean_after, inv_cov_after = reward._cache[cls]
        np.testing.assert_array_equal(mean_before, mean_after)
        np.testing.assert_array_equal(inv_cov_before, inv_cov_after)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
