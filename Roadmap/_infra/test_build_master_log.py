"""
test_build_master_log.py -- regression tests for build_master_log.py's
sanity-check logic, focused on the 2026-07-02 false-positive fix.

Run:
    pytest Roadmap/_infra/test_build_master_log.py -v

Two kinds of test here, deliberately both present:
  - False-positive regressions: confirm the specific bugs found in this
    repo's real usage no longer fire.
  - True-positive guards: confirm the fix didn't just widen a blind spot
    to make the false positives go away -- a genuine size mismatch,
    a genuinely-should-vary-but-doesn't metric, and a failed run must
    still be caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_master_log import _run_sanity_checks


def _record(experiment_id, status="success", params=None, metrics=None):
    return {
        "experiment_id": experiment_id,
        "status": status,
        "params": params or {},
        "metrics": metrics or {},
        "log_file": "fake.log",
    }


# ── False-positive regressions ──────────────────────────────────────────

def test_batch_size_not_confused_with_n_train_records_actual():
    """The exact bug found in run_experiment_1_for_real.py's real ledger
    shape: batch_size (params) and n_train_records_actual (metrics) in the
    same record, semantically unrelated, previously flagged because
    "batch_size" ends in "size" and "n_train_records_actual" contains
    "actual"/"n_train"/"record"."""
    records = [_record(
        "exp1_baseline_reproduction",
        params={"batch_size": 32, "n_epochs": 200},
        metrics={"n_train_records_actual": 17418, "best_val_loss": 0.05},
    )]
    flags = _run_sanity_checks(records)
    assert flags == [], f"expected no flags, got: {flags}"


def test_n_generated_constant_across_family_not_flagged():
    """n_generated is a fixed evaluation budget (n_gen_per_class x
    n_generatable_classes) in run_dataset_scaling.py -- constant BY DESIGN
    across every dataset size in a scaling sweep."""
    records = [
        _record(f"exp2_dataset_scaling_{size}",
                params={"dataset_size_requested": size},
                metrics={"n_generated": 300, "n_train_records_actual": size})
        for size in (380, 1000, 2500, 5000, 10000)
    ]
    flags = _run_sanity_checks(records)
    assert flags == [], f"expected no flags, got: {flags}"


def test_small_rounding_deviation_within_tolerance_not_flagged():
    """Stratified per-class rounding can legitimately produce a small
    (empirically measured <=2, tolerance set to 5) deviation between
    requested and actual size -- this is not the bug the check exists for."""
    records = [_record(
        "exp2_dataset_scaling_5000",
        params={"dataset_size_requested": 5000},
        metrics={"n_train_records_actual": 5002},
    )]
    flags = _run_sanity_checks(records)
    assert flags == [], f"expected no flags for a 2-record deviation, got: {flags}"


# ── True-positive guards (the fix must not go blind) ────────────────────

def test_large_size_mismatch_still_flagged():
    """The actual bug this check exists for: every requested size produces
    the same small actual count, far beyond the rounding tolerance."""
    records = [_record(
        "exp2_dataset_scaling_5000",
        params={"dataset_size_requested": 5000},
        metrics={"n_train_records_actual": 380},
    )]
    flags = _run_sanity_checks(records)
    assert len(flags) == 1
    assert "dataset_size_requested=5000" in flags[0]
    assert "n_train_records_actual=380" in flags[0]


def test_unexpected_constant_metric_still_flagged():
    """A metric that SHOULD vary with dataset size (e.g. final_train_loss)
    but doesn't must still be caught -- constant-by-design is an
    allowlisted exception, not a loosened check."""
    records = [
        _record(f"exp2_dataset_scaling_{size}",
                params={"dataset_size_requested": size},
                metrics={"final_train_loss": 0.12345, "n_train_records_actual": size})
        for size in (380, 1000, 2500)
    ]
    flags = _run_sanity_checks(records)
    assert any("final_train_loss" in f for f in flags), f"expected final_train_loss flagged, got: {flags}"


def test_failed_status_still_flagged():
    records = [_record("exp1_baseline_sanity_check", status="failed")]
    flags = _run_sanity_checks(records)
    assert len(flags) == 1
    assert "status='failed'" in flags[0]


def test_unregistered_pair_never_compared():
    """A param/metric pair that isn't in REQUESTED_ACTUAL_PAIRS is never
    compared, even if it superficially resembles the registered pattern --
    this is deliberate (see module docstring): new families must register
    pairs explicitly rather than rely on heuristic matching."""
    records = [_record(
        "exp3_some_other_experiment",
        params={"some_size_requested": 100},
        metrics={"some_actual_count": 999999},
    )]
    flags = _run_sanity_checks(records)
    assert flags == [], f"expected no flags for an unregistered pair, got: {flags}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
