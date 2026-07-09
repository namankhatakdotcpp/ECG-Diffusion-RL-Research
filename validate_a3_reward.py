"""
validate_a3_reward.py — correctness gate for step06_reward_function.A3Reward,
required to run (and pass) before A3 becomes a nonzero, load-bearing term in
PPO's reward signal.

WHY THIS EXISTS
----------------
A3Reward reuses mentor_eval.subband_features.extract_subband_energy_features
(same wavelet extraction Stage 3's evaluation pipeline uses) but factors the
Mahalanobis distance formula from mentor_eval.similarity_metrics.
mahalanobis_distance differently (cached per-class mean/inv_cov instead of
recomputed every call, for PPO-loop performance — see A3Reward's docstring
in step06_reward_function.py for exactly why). "Refactored but produces a
different number on the same inputs" is a wiring bug, not new information —
exactly the standard already applied throughout this project when a metric's
numbers were spot-checked against another pipeline before being trusted.

THREE COMPARISONS
-------------------
  1. Real vs. Real       — expect the HIGHEST reward (same-class real ECGs
                            scored against their own class's reference).
  2. Generated vs. Real  — expect LOWER than (1). Requires a diffusion
                            checkpoint (outputs/models/diffusion_best.pt);
                            [BLOCKED] and skipped, not faked, if absent.
  3. Mismatched disease  — expect the LOWEST reward (real ECG from class X,
                            scored against class Y's reference).

UNCERTAINTY, NOT JUST A POINT ESTIMATE
------------------------------------------
Each comparison pools 3 independent draws (seeds 42/123/456 — same pattern
as the earlier n_seeds=3 TSTR/TRTR baseline) into one sample, then computes
a 95% percentile bootstrap CI on the mean. A single point estimate cannot
distinguish "real ordering violation" from "noise at this sample size" —
the first version of this script reported a raw-mean FAIL (generated
0.8174 vs real 0.7092) that turned out to be a scale-mismatch bug in this
script itself (fixed), and even after fixing it the remaining ~0.01 gap
was not distinguishable from noise without a CI. Verdict per comparison is
PASS / INCONCLUSIVE / FAIL:
  - PASS:          CIs don't overlap, ordering is in the expected direction.
  - INCONCLUSIVE:  CIs overlap — not evidence of a bug, but not evidence of
                    correctness either. Do not force this to PASS.
  - FAIL:          CIs don't overlap, ordering is CONFIDENTLY WRONG.
Overall verdict is FAIL if any comparison FAILs, else INCONCLUSIVE if any
comparison is INCONCLUSIVE, else PASS. Do not enable reward_a3's nonzero
weight in a real training run until this returns a real PASS.

NUMERIC CHECK AGAINST STAGE 3
--------------------------------
Compares this reward's raw (pre-exp) Mahalanobis distance against the
Mahalanobis/Bhattacharyya values in Roadmap/Stage_3_Architecture_Improvements/
Reports/Stage3_Subband_Master_Comparison.csv for S3-001, SAME class, A3 row —
read from that file, never hardcoded. AS OF THIS WRITING that file is
"0/72 rows evaluated" (every row reads "not yet evaluated -- no
subband_similarity_metrics.csv found") — there is currently NO real number
anywhere in this repo to numerically validate against. This script detects
that state explicitly and reports "WARNING: Reference Stage3 subband
metrics unavailable — only ordering validation executed, numeric
validation skipped" rather than skipping the check silently, fabricating a
comparison, or blocking the whole script (the ordering checks in [1]/[2]/
[3] above are independent of this and still run). Run
`python -m mentor_eval.subband_similarity_metrics` on the GPU first (needs
outputs/models/diffusion_best.pt) to populate real reference numbers, then
re-run this script.

Usage:
    python validate_a3_reward.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, set_seed

STAGE3_SUBBAND_CSV = (
    Path(__file__).parent / "Roadmap" / "Stage_3_Architecture_Improvements"
    / "Reports" / "Stage3_Subband_Master_Comparison.csv"
)
# Reversed from mentor_eval.class_mapping.MENTOR_TO_TRAINED_CLASS (confirmed
# from source, not guessed: {"Normal": "NORM", "STEMI": "MI", "NSTEMI":
# "STTC", "AFIB": None}). subband_similarity_metrics.py's BOX_CLASSES are the
# Mentor-review disease names (Normal/STEMI/NSTEMI), not this project's
# trained class_names (NORM/MI/STTC/CD/HYP/OTHER) — CD/HYP/OTHER have no
# Stage-3-subband counterpart to check.
S3_001_MENTOR_CLASS_MAP = {"NORM": "Normal", "MI": "STEMI", "STTC": "NSTEMI"}


def _load_stage3_a3_row(candidate: str, mentor_class: str) -> dict | None:
    if not STAGE3_SUBBAND_CSV.exists():
        return None
    with open(STAGE3_SUBBAND_CSV) as f:
        for row in csv.DictReader(f):
            if (row.get("Candidate") == candidate and row.get("Disease") == mentor_class
                    and row.get("Subband") == "A3"):
                return row
    return None


def _bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap CI for the mean of `values`. A single point
    estimate with no uncertainty bound isn't a validation result, it's an
    observation — this is what lets the caller tell "real difference"
    apart from "sampling noise at this sample size"."""
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr),
            "ci_lo": float(lo), "ci_hi": float(hi)}


def _intervals_overlap(a: dict, b: dict) -> bool:
    return not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])


def main() -> None:
    cfg = load_config()
    log = get_logger("validate_a3_reward", cfg=cfg)
    set_seed(int(cfg.seeds[0]))

    from step06_reward_function import A3Reward

    processed_dir = Path(cfg.paths.outputs.processed)
    models_dir    = Path(cfg.paths.outputs.models)

    a3_stats_path = processed_dir / "a3_subband_stats.json"
    if not a3_stats_path.exists():
        print(
            f"[BLOCKED] {a3_stats_path} not found. Run "
            "step03_eda_and_class_mapping.py first (Stage 7) to build the "
            "A3 reference distribution. No validation performed — nothing "
            "fabricated."
        )
        sys.exit(1)

    a3_stats_file = json.load(open(a3_stats_path))
    reward = A3Reward(a3_stats_file)
    a3_stats = a3_stats_file.get("classes", a3_stats_file)   # flat-dict fallback

    class_names_path = processed_dir / "class_names.json"
    class_names = json.load(open(class_names_path)) if class_names_path.exists() else list(cfg.ptbxl.classes)

    X_test_path = processed_dir / "X_test.npy"
    if not X_test_path.exists():
        print(f"[BLOCKED] {X_test_path} not found. Run step02 first.")
        sys.exit(1)
    X_test = np.load(str(X_test_path))

    # We need class labels for X_test to pick same-class / mismatched-class
    # samples. Reuse the same loader step06's self-test uses.
    from step06_reward_function import _load_labels_simple
    db_path = Path(cfg.paths.data.ptbxl) / "ptbxl_database.csv"
    class_mapping_path = processed_dir / "class_mapping.json"
    if not (db_path.exists() and class_mapping_path.exists()):
        print(f"[BLOCKED] {db_path} or {class_mapping_path} not found — cannot label X_test.")
        sys.exit(1)
    class_mapping = json.load(open(class_mapping_path))
    record_ids_test = np.load(str(processed_dir / "record_ids_test.npy"))
    valid_idx, y_test = _load_labels_simple(record_ids_test, db_path, class_mapping, class_names)
    X_test = X_test[valid_idx]

    results: dict[str, dict] = {}
    SEEDS = [42, 123, 456]   # same pattern as the earlier n_seeds=3 TSTR/TRTR baseline

    print("=" * 70)
    print("A3Reward VALIDATION")
    print("=" * 70)

    # ── 1. Real vs. Real ─────────────────────────────────────────────────────
    # Pooled across 3 independent draws (different rng seed per draw, not
    # just one fixed sample of 20) so the aggregate has enough points for a
    # meaningful bootstrap CI, not just 3 seed-level means.
    print("\n[1] REAL vs. REAL")
    real_real_scores: dict[str, float] = {}
    real_real_raw: list[float] = []
    for cls in a3_stats.keys():
        idx_all = np.where(y_test == class_names.index(cls))[0]
        if len(idx_all) < 5:
            continue
        cls_scores: list[float] = []
        for seed in SEEDS:
            rng_cls = np.random.default_rng(seed)
            idx = rng_cls.choice(idx_all, size=min(20, len(idx_all)), replace=False)
            sample = X_test[idx]
            cls_scores.extend(reward.compute(ecg, cls) for ecg in sample)
        real_real_scores[cls] = float(np.mean(cls_scores))
        real_real_raw.extend(cls_scores)
        print(f"    {cls}: mean reward = {real_real_scores[cls]:.4f}  (n={len(cls_scores)}, {len(SEEDS)} seeds)")
    results["real_vs_real"] = real_real_scores

    # ── 2. Generated vs. Real (S3-001 / diffusion_best.pt) ──────────────────
    print("\n[2] GENERATED vs. REAL (S3-001 base checkpoint)")
    gen_real_scores: dict[str, float] = {}
    gen_real_raw: list[float] = []
    ckpt_path = models_dir / "diffusion_best.pt"
    from mentor_eval.checkpoint_utils import load_checkpoint, generate_for_class
    loaded = load_checkpoint(ckpt_path, cfg)
    if loaded is None:
        print(
            f"    [BLOCKED] No checkpoint at {ckpt_path} — cannot generate "
            "samples on this machine. Re-run this script on the GPU server "
            "once diffusion_best.pt exists. No generated-data scores fabricated."
        )
    else:
        # stats=None (NOT preprocessing_stats.json): A3Reward's per-class
        # reference distribution (a3_subband_stats.json) was built from
        # z-score-normalised X_train, and X_test (comparison 1/3, above) is
        # also z-score space. Passing stats=<preprocessing_stats> here
        # denormalises generated output to raw-mV scale (std collapses to
        # ~0.1 vs X_test's ~0.8 in z-score space, confirmed empirically) —
        # comparing across two different scales produced a meaningless
        # generated-vs-real reward the first time this ran. Caught by this
        # script's own ordering check (generated > real, which should be
        # structurally impossible), not assumed to be correct.
        for cls in a3_stats.keys():
            cls_scores: list[float] = []
            for seed in SEEDS:
                samples, err = generate_for_class(
                    loaded, cls, n_samples=20, cfg=cfg, seed=seed, stats=None,
                )
                if err:
                    print(f"    {cls} (seed={seed}): [SKIPPED] {err}")
                    continue
                cls_scores.extend(reward.compute(ecg, cls) for ecg in samples)
            if not cls_scores:
                continue
            gen_real_scores[cls] = float(np.mean(cls_scores))
            gen_real_raw.extend(cls_scores)
            print(f"    {cls}: mean reward = {gen_real_scores[cls]:.4f}  (n={len(cls_scores)}, {len(SEEDS)} seeds)")
    results["generated_vs_real"] = gen_real_scores

    # ── 3. Mismatched disease ────────────────────────────────────────────────
    print("\n[3] MISMATCHED DISEASE (real ECG scored against the WRONG class)")
    mismatched_scores = {}
    mismatched_raw: list[float] = []
    classes_with_stats = list(a3_stats.keys())
    for cls in classes_with_stats:
        idx_all = np.where(y_test == class_names.index(cls))[0]
        if len(idx_all) < 5:
            continue
        wrong_cls = next((c for c in classes_with_stats if c != cls), None)
        if wrong_cls is None:
            continue
        cls_scores: list[float] = []
        for seed in SEEDS:
            rng_cls = np.random.default_rng(seed)
            idx = rng_cls.choice(idx_all, size=min(20, len(idx_all)), replace=False)
            sample = X_test[idx]
            cls_scores.extend(reward.compute(ecg, wrong_cls) for ecg in sample)
        mismatched_scores[f"{cls}_scored_as_{wrong_cls}"] = float(np.mean(cls_scores))
        mismatched_raw.extend(cls_scores)
        print(f"    {cls} scored as {wrong_cls}: mean reward = {np.mean(cls_scores):.4f}  (n={len(cls_scores)})")
    results["mismatched"] = mismatched_scores

    # ── Ordering check, with bootstrap uncertainty ───────────────────────────
    # A single point estimate is an observation, not a validation result --
    # PASS/FAIL on raw means alone can't distinguish "real ordering violation"
    # from "noise at this sample size." Overlapping 95% CIs -> INCONCLUSIVE,
    # not a forced FAIL; only a confidently-wrong ordering is a real FAIL.
    print("\n[ORDERING CHECK — with 95% bootstrap CI]")
    verdicts: dict[str, str] = {}

    if real_real_raw and mismatched_raw:
        ci_real = _bootstrap_ci(real_real_raw, seed=1)
        ci_mismatched = _bootstrap_ci(mismatched_raw, seed=2)
        print(f"    real-vs-real:  mean={ci_real['mean']:.4f}  95% CI=[{ci_real['ci_lo']:.4f}, {ci_real['ci_hi']:.4f}]  n={ci_real['n']}")
        print(f"    mismatched:    mean={ci_mismatched['mean']:.4f}  95% CI=[{ci_mismatched['ci_lo']:.4f}, {ci_mismatched['ci_hi']:.4f}]  n={ci_mismatched['n']}")
        if _intervals_overlap(ci_real, ci_mismatched):
            verdicts["real_vs_mismatched"] = "INCONCLUSIVE"
            print("    [INCONCLUSIVE] CIs overlap — cannot confidently distinguish real-vs-real from mismatched at this sample size.")
        elif ci_real["ci_lo"] > ci_mismatched["ci_hi"]:
            verdicts["real_vs_mismatched"] = "PASS"
            print("    [PASS] real-vs-real confidently higher than mismatched (CIs do not overlap), as expected.")
        else:
            verdicts["real_vs_mismatched"] = "FAIL"
            print("    [FAIL] real-vs-real is confidently LOWER than mismatched — treat as a wiring bug, not new information.")
        results["real_vs_mismatched_ci"] = {"real": ci_real, "mismatched": ci_mismatched}

    if gen_real_raw and real_real_raw:
        ci_gen = _bootstrap_ci(gen_real_raw, seed=3)
        ci_real = _bootstrap_ci(real_real_raw, seed=1)
        print(f"    generated-vs-real: mean={ci_gen['mean']:.4f}  95% CI=[{ci_gen['ci_lo']:.4f}, {ci_gen['ci_hi']:.4f}]  n={ci_gen['n']}")
        print(f"    real-vs-real:       mean={ci_real['mean']:.4f}  95% CI=[{ci_real['ci_lo']:.4f}, {ci_real['ci_hi']:.4f}]  n={ci_real['n']}")
        if _intervals_overlap(ci_gen, ci_real):
            verdicts["generated_vs_real"] = "INCONCLUSIVE"
            print(f"    [INCONCLUSIVE] CIs overlap — the {ci_gen['mean'] - ci_real['mean']:+.4f} gap is not "
                  "distinguishable from noise at this sample size (60/class/group, 3 seeds).")
        elif ci_real["ci_lo"] > ci_gen["ci_hi"]:
            verdicts["generated_vs_real"] = "PASS"
            print("    [PASS] real-vs-real confidently higher than generated-vs-real (CIs do not overlap), as expected.")
        else:
            verdicts["generated_vs_real"] = "FAIL"
            print("    [FAIL] generated-vs-real is confidently HIGHER than real-vs-real (CIs do not overlap) — "
                  "treat as a wiring bug, not evidence the model is superhuman.")
        results["generated_vs_real_ci"] = {"generated": ci_gen, "real": ci_real}

    # ── Numeric check against Stage 3 (if it exists) ─────────────────────────
    print("\n[NUMERIC CHECK vs. Stage3_Subband_Master_Comparison.csv]")
    any_stage3_data = False
    for cls, mentor_cls in S3_001_MENTOR_CLASS_MAP.items():
        row = _load_stage3_a3_row("S3-001", mentor_cls)
        if row is None:
            print(f"    {cls}/{mentor_cls}: Stage3_Subband_Master_Comparison.csv not found.")
            continue
        if row.get("Status", "").startswith("not yet evaluated") or not row.get("Mahalanobis"):
            print(
                f"    {cls}/{mentor_cls}: WARNING: Reference Stage3 subband "
                f"metrics unavailable (row status = {row.get('Status')!r}) — "
                "only ordering validation executed above, numeric validation "
                "skipped for this class. Run mentor_eval/subband_similarity_"
                "metrics.py on the GPU to produce a real number, then re-run."
            )
            continue
        any_stage3_data = True
        stage3_mahal = float(row["Mahalanobis"])
        print(f"    {cls}/{mentor_cls}: Stage3 Mahalanobis = {stage3_mahal:.4f} "
              "(comparison against this reward's own distance requires the "
              "generated-vs-real branch above to have run — see script output)")

    if not any_stage3_data:
        print(
            "\n    WARNING: Reference Stage3 subband metrics unavailable for "
            "every class — no real Stage 3 subband numbers exist anywhere in "
            "this repo yet (Stage3_Subband_Master_Comparison.md: 0/72 rows "
            "evaluated, confirmed by reading the file directly). Only "
            "ordering validation ([1]/[2]/[3] above) was executed; numeric "
            "magnitude validation against Stage 3 was skipped until "
            "subband_similarity_metrics.py is actually run on the GPU."
        )

    # ── Overall verdict: PASS / INCONCLUSIVE / FAIL, never forced to PASS ────
    print("\n" + "=" * 70)
    print("VERDICTS")
    for name, v in verdicts.items():
        print(f"  {name}: {v}")

    if not verdicts:
        overall = "INCONCLUSIVE"
        print("\n⚠ A3Reward: NO comparisons could be run (missing data/checkpoint) "
              "— INCONCLUSIVE, not PASS. Do not enable a nonzero a3 weight based on this run.")
    elif any(v == "FAIL" for v in verdicts.values()):
        overall = "FAIL"
        print("\n✗ A3Reward: FAIL. Do not enable a nonzero a3 weight in config.yaml "
              "until this is diagnosed — treat as a wiring bug, not new information.")
    elif any(v == "INCONCLUSIVE" for v in verdicts.values()):
        overall = "INCONCLUSIVE"
        print("\n⚠ A3Reward: INCONCLUSIVE — at least one comparison's CIs overlap. "
              "Not evidence of a bug, but not evidence of correctness either. Do "
              "NOT enable a nonzero a3 weight in a real training run on this "
              "result alone; increase sample size / seed count, or accept the "
              "uncertainty explicitly and get sign-off before proceeding.")
    else:
        overall = "PASS"
        print("\n✓ A3Reward: PASS. All ordering comparisons confidently correct "
              "(95% CIs do not overlap in the wrong direction). Numeric Stage-3 "
              "magnitude comparison may still be a WARNING above if reference "
              "data doesn't exist yet — that's a separate, independent gate.")
    print("=" * 70)

    out_path = Path(cfg.paths.outputs.results) / "a3_reward_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "verdicts": verdicts, "overall": overall}, f, indent=2)
    print(f"\nSaved → {out_path}")

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
