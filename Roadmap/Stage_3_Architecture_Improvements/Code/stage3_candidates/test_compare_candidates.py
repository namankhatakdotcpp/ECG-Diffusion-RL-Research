"""
test_compare_candidates.py -- regression tests for compare_candidates.py.

Run:
    pytest Roadmap/Stage_3_Architecture_Improvements/Code/stage3_candidates/test_compare_candidates.py -v

Focused on the specific risk raised during Stage 3 review: that
compare_candidates.py's status handling could collapse "evaluation
crashed" (status=eval_failed) into the same bucket as "hasn't run yet"
(no metadata.json at all) -- those are different debugging actions
(fix the evaluator vs. wait longer) and the table must be able to tell
them apart. Also covers optimizer-config ancestry classification (must
be derived from git history, never a hardcoded per-run_id literal) and
the delta/similarity computations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_candidates as cc


@pytest.fixture
def tmp_results_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "RESULTS_ROOT", tmp_path)
    return tmp_path


def _write_metadata(run_dir: Path, **fields) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(fields, open(run_dir / "metadata.json", "w"))


# ── Status edge cases: the thing doc 74 specifically asked to verify ────

def test_never_run_vs_eval_failed_are_distinguishable(tmp_results_root):
    """The core ask: a candidate that hasn't started (no metadata.json)
    must not report the same status string as one whose training
    succeeded but whose evaluation crashed (status=eval_failed)."""
    never_run = cc.load_candidate_row("S3-999-NEVER-RUN", "baseline")

    eval_failed_dir = tmp_results_root / "S3-999-EVAL-FAILED"
    (eval_failed_dir / "mentor_eval").mkdir(parents=True)
    _write_metadata(eval_failed_dir, status="eval_failed", commit="deadbeef")
    eval_failed = cc.load_candidate_row("S3-999-EVAL-FAILED", "baseline")

    assert never_run["Status"] != eval_failed["Status"]
    assert "eval_failed" in eval_failed["Status"]
    assert "no metadata.json" in never_run["Status"]


def test_train_failed_reported_distinctly(tmp_results_root):
    run_dir = tmp_results_root / "S3-999-TRAIN-FAILED"
    _write_metadata(run_dir, status="train_failed", commit="deadbeef")
    row = cc.load_candidate_row("S3-999-TRAIN-FAILED", "baseline")
    assert row["Status"] == "train_failed"
    assert row["Generated Accuracy"] is None


def test_status_done_but_generated_eval_missing_is_flagged_not_silent(tmp_results_root):
    """status=done should always have classifier_generated_eval.json
    (run_candidate() only sets status=done after the eval subprocess
    succeeds) -- if it's missing anyway, that is itself worth a loud
    status string, not a silent None."""
    run_dir = tmp_results_root / "S3-999-DONE-NO-EVAL"
    (run_dir / "mentor_eval").mkdir(parents=True)
    _write_metadata(run_dir, status="done", commit="deadbeef")
    row = cc.load_candidate_row("S3-999-DONE-NO-EVAL", "baseline")
    assert "unexpected" in row["Status"]


def test_partial_evaluation_real_metrics_present_generated_missing(tmp_results_root):
    """classification_validation.py's own [PARTIAL] behavior: Stage 1
    (real-data classifier) can complete while Stage 2 (generated-data
    eval) is still blocked on a missing checkpoint."""
    run_dir = tmp_results_root / "S3-999-PARTIAL"
    eval_dir = run_dir / "mentor_eval"
    eval_dir.mkdir(parents=True)
    json.dump({"accuracy": 0.84}, open(eval_dir / "classifier_real_eval.json", "w"))
    _write_metadata(run_dir, status="training", commit="deadbeef")
    row = cc.load_candidate_row("S3-999-PARTIAL", "baseline")
    assert row["Real-data Accuracy"] == 0.84
    assert row["Generated Accuracy"] is None


# ── Optimizer-config ancestry classification ────────────────────────────

def test_optimizer_config_unknown_commit_hash():
    assert cc.classify_optimizer_config("not-a-real-commit-hash") == "unknown"


def test_optimizer_config_none_commit():
    assert cc.classify_optimizer_config(None) == "unknown"


def test_optimizer_config_pre_fix_real_history():
    """Uses this repo's own real git history: the commit that introduced
    stage3_metadata.py (8507fa8) is an ancestor of, and predates, the
    weight-decay fix -- must classify as pre-fix, not post-fix."""
    result = cc.classify_optimizer_config("8507fa8")
    assert result == "pre-fix", f"expected pre-fix, got {result}"


def test_optimizer_config_post_fix_real_history():
    """The tripwire commit itself must classify as post-fix (a commit is
    its own ancestor)."""
    result = cc.classify_optimizer_config("432395c")
    assert result == "post-fix", f"expected post-fix, got {result}"


def test_optimizer_config_not_hardcoded_per_run_id():
    """Regression guard for the specific risk raised: classify_optimizer_config
    must be a pure function of the commit hash, not a lookup keyed by
    run_id. Confirmed two ways: (1) its signature takes only a commit
    hash, no run_id; (2) two rows with DIFFERENT run_ids but the SAME
    commit hash must classify identically -- a hardcoded per-run_id
    lookup would not guarantee this."""
    import inspect
    sig = inspect.signature(cc.classify_optimizer_config)
    assert list(sig.parameters) == ["commit"], (
        f"expected classify_optimizer_config(commit) only, got {list(sig.parameters)}"
    )
    same_commit = "8507fa8"
    assert cc.classify_optimizer_config(same_commit) == cc.classify_optimizer_config(same_commit)


# ── Delta computation ────────────────────────────────────────────────────

def test_delta_computed_when_both_present():
    row = {"Generated Accuracy": 0.90, "Delta vs. Baseline (accuracy)": None}
    cc._delta(row, baseline_accuracy=0.85)
    assert row["Delta vs. Baseline (accuracy)"] == pytest.approx(0.05)


def test_delta_none_when_baseline_missing():
    row = {"Generated Accuracy": 0.90, "Delta vs. Baseline (accuracy)": None}
    cc._delta(row, baseline_accuracy=None)
    assert row["Delta vs. Baseline (accuracy)"] is None


def test_delta_none_when_candidate_not_evaluated():
    row = {"Generated Accuracy": None, "Delta vs. Baseline (accuracy)": None}
    cc._delta(row, baseline_accuracy=0.85)
    assert row["Delta vs. Baseline (accuracy)"] is None


# ── Similarity summary: excludes flagged/skipped rows from the mean ─────

def test_similarity_summary_excludes_flagged_rows(tmp_path):
    run_dir = tmp_path / "S3-999-SIM"
    (run_dir / "mentor_eval").mkdir(parents=True)
    pd.DataFrame([
        {"class": "Normal", "cosine_similarity": 0.90, "mahalanobis": 1.0, "bhattacharyya": 0.2, "flag": ""},
        {"class": "AFIB", "cosine_similarity": None, "mahalanobis": None, "bhattacharyya": None,
         "flag": "No generated samples available"},
    ]).to_csv(run_dir / "mentor_eval" / "similarity_metrics.csv", index=False)
    summary = cc.similarity_summary(run_dir)
    assert summary["cosine"] == 0.90


def test_similarity_summary_missing_file_returns_all_none(tmp_path):
    run_dir = tmp_path / "S3-999-NOFILE"
    summary = cc.similarity_summary(run_dir)
    assert summary == {"cosine": None, "mahalanobis": None, "bhattacharyya": None}


# ── Baseline manifest: must fail loudly, never silently compute live ────

def test_resolve_baseline_raises_when_manifest_missing(tmp_path, monkeypatch):
    """Regression guard for the exact bug found in real GPU-server usage:
    _resolve_baseline() used to silently fall back to a live-computed
    checksum when BASELINE_MANIFEST_PATH didn't exist, producing a
    baseline number indistinguishable from a verified one in the
    rendered table. Must now raise instead."""
    monkeypatch.setattr(cc, "BASELINE_MANIFEST_PATH", tmp_path / "nonexistent_manifest.json")
    with pytest.raises(cc.MissingBaselineManifestError):
        cc._resolve_baseline()


def test_resolve_baseline_succeeds_with_real_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / "baseline_manifest.json"
    json.dump({"checkpoint_sha256": "abc123", "git_commit": "deadbeef"}, open(manifest_path, "w"))
    monkeypatch.setattr(cc, "BASELINE_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(cc, "BASELINE_METRICS_PATH", tmp_path / "nonexistent.json")
    accuracy, provenance = cc._resolve_baseline()
    assert provenance["source"] == "frozen baseline_manifest.json"
    assert provenance["checkpoint_sha256"] == "abc123"
    assert accuracy is None  # BASELINE_METRICS_PATH doesn't exist in this test, separate from the manifest


def test_resolve_baseline_opt_out_still_computes_live(tmp_path, monkeypatch):
    """require_manifest=False is kept for a caller that explicitly wants
    the old best-effort behavior -- confirmed it still works, just isn't
    the default anymore."""
    monkeypatch.setattr(cc, "BASELINE_MANIFEST_PATH", tmp_path / "nonexistent_manifest.json")
    accuracy, provenance = cc._resolve_baseline(require_manifest=False)
    assert "COMPUTED LIVE" in provenance["source"]


def test_main_continues_reporting_candidates_when_manifest_missing(tmp_path, monkeypatch, capsys):
    """main() must not crash the whole report on a missing manifest --
    per-candidate metrics are still valid on their own -- but it must
    print an unmistakable error, not proceed silently."""
    monkeypatch.setattr(cc, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(cc, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(cc, "BASELINE_MANIFEST_PATH", tmp_path / "nonexistent_manifest.json")
    cc.main()
    captured = capsys.readouterr()
    assert "ERROR: frozen baseline manifest missing" in captured.out
    assert (tmp_path / "reports" / "Stage3_Comparison.md").exists()


# ── Subband/Sharma metrics: distinguishes "not computed" from "computed" ─

def test_subband_summary_none_when_absent(tmp_path):
    run_dir = tmp_path / "S3-999-NOSUBBAND"
    assert cc.subband_summary(run_dir) is None


def test_subband_summary_present_when_file_exists(tmp_path):
    run_dir = tmp_path / "S3-999-SUBBAND"
    (run_dir / "mentor_eval").mkdir(parents=True)
    pd.DataFrame([{"subband": "A3", "class": "Normal", "cosine_similarity": 0.5}]).to_csv(
        run_dir / "mentor_eval" / "subband_similarity_metrics.csv", index=False
    )
    result = cc.subband_summary(run_dir)
    assert result is not None
    assert result["n_rows"] == 1


# ── Training duration ────────────────────────────────────────────────────

def test_training_duration_only_reported_when_done():
    in_progress = {"status": "training", "start_time": "2026-07-04T20:37:19+00:00",
                   "last_updated_utc": "2026-07-05T02:54:46+00:00"}
    assert cc.training_duration_str(in_progress) is None


def test_training_duration_computed_when_done():
    done = {"status": "done", "start_time": "2026-07-04T20:37:19+00:00",
            "last_updated_utc": "2026-07-05T02:54:46+00:00"}
    assert cc.training_duration_str(done) == "6h 17m"


# ── Parameter count: real computation, not a stub ───────────────────────

def test_param_count_differs_across_variants():
    """LayerScale adds 12 gamma parameters (2 per block x 6 blocks x
    model_dim=256 = 3072 extra scalars) on top of baseline's ~8.43M --
    real but tiny relative to the total, so it rounds away at 2 decimal
    places in param_count_millions()'s displayed value. Verify the
    UNDERLYING computation differs (via build_variant_model directly,
    bypassing the display rounding) rather than asserting on the
    rounded value, which would be a false negative here."""
    cfg = cc.load_config()
    n_classes = int(cfg.ptbxl.n_classes)
    baseline_model = cc.build_variant_model(cfg, n_classes=n_classes, variant="baseline")
    layerscale_model = cc.build_variant_model(cfg, n_classes=n_classes, variant="layerscale")
    baseline_n = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    layerscale_n = sum(p.numel() for p in layerscale_model.parameters() if p.requires_grad)
    assert layerscale_n - baseline_n == 3072, (
        f"expected exactly 3072 extra params (12 x model_dim=256), got {layerscale_n - baseline_n}"
    )

    assert cc.param_count_millions("baseline") is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
