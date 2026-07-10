"""
analyze_stage4_hyp_other.py -- cross-checks whether HYP/OTHER's r_diag
collapse in a Stage 4 RL fine-tuning run is corroborated by the
independent Mentor Classifier (mentor_eval.classification_validation),
which is retrained separately from the reward function's own TRTR
classifier, and whether it's a pre-existing classifier weakness rather
than something RL made worse.

WHY THIS EXISTS
----------------
rl_training_log.csv's own r_diag values come from the SAME TRTR
classifier used inside DiagnosticUtilityReward -- so a collapse there
could in principle reflect a quirk of that one classifier rather than a
real property of the generated ECGs. The full-eval checkpoints
(rl.full_eval_checkpoints) independently retrain a Mentor Classifier and
score it against both real and RL-generated data, written to
outputs/mentor_review/rl_checkpoint_iter{N:04d}_rep{R}/classifier_
{generated,real}_eval.json for each repeat R. This script reads those
files (never the reward log) to check the same question from an
independent angle.

METHOD
-------
1/2. Load classifier_generated_eval.json and classifier_real_eval.json
     for every requested rep; print a per-rep, per-class F1 table for
     the target classes plus mean/std across reps.
3.   Rank all classes by mean generated-data F1 (averaged across reps)
     to see whether the target classes are uniquely bad or just the
     lowest among a generally weak set.
4.   Cross-reference against the original TRTR per_class_f1
     (outputs/models/trtr_classifier_eval.json) -- a pre-existing
     weakness on a class would show up there too, before any RL run.
5.   Read checkpoint_metrics.json's lightweight-eval trend
     (trtr_diag_conf_mean, a3_reward_mean) across iterations and print
     it next to the full-eval numbers for a consistency check.

CAVEAT: with n_repeats reps (project convention: 3), per-class std
estimates are directional, not statistically powered -- same honesty
standard as diagnose_smoke_test_reward_trend.py's n=10 caveat.

Usage:
    python analyze_stage4_hyp_other.py --run-tag stage4_finetune_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

TARGET_CLASSES = ["HYP", "OTHER"]
TRTR_FIXED_CLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP", "OTHER"]


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="stage4_finetune_v1")
    parser.add_argument("--iteration", type=int, default=1000)
    parser.add_argument("--reps", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--trtr-eval", default="outputs/models/trtr_classifier_eval.json")
    args = parser.parse_args()

    print("=" * 70)
    print(f"STAGE 4 HYP/OTHER CROSS-CHECK  (run-tag={args.run_tag}, "
          f"iteration={args.iteration}, reps={args.reps})")
    print("=" * 70)

    # ── Load all reps ─────────────────────────────────────────────────────────
    rep_data: dict[int, dict] = {}
    missing = []
    for rep in args.reps:
        base = Path("outputs/mentor_review") / f"rl_checkpoint_iter{args.iteration:04d}_rep{rep}"
        gen = load_json(base / "classifier_generated_eval.json")
        real = load_json(base / "classifier_real_eval.json")
        if gen is None or real is None:
            missing.append((rep, base))
            continue
        rep_data[rep] = {"generated": gen, "real": real}

    if missing:
        print("\n[WARNING] missing rep data (not found on disk):")
        for rep, base in missing:
            print(f"  rep{rep}: {base}")

    if not rep_data:
        print(
            "\n[BLOCKED] No rep data found under "
            f"outputs/mentor_review/rl_checkpoint_iter{args.iteration:04d}_rep*/ "
            "-- nothing to analyze. Check --run-tag / --iteration / --reps match "
            "the actual directory names on disk."
        )
        return

    class_names = None
    for d in rep_data.values():
        pcf = d["generated"].get("per_class_f1", {})
        if pcf:
            class_names = list(pcf.keys())
            break
    if class_names is None:
        print("\n[BLOCKED] No per_class_f1 found in any generated_eval.json -- "
              "schema mismatch, inspect the files manually.")
        return
    print(f"\nClasses found: {class_names}")

    # ── 1/2. Per-rep, per-class table ────────────────────────────────────────
    print("\n[1/2. Per-rep, per-class F1/precision/recall — target classes]")
    class_gen_means: dict[str, float] = {}
    class_gen_stds: dict[str, float] = {}
    for cname in class_names:
        gen_vals, real_vals = [], []
        print(f"\n  Class: {cname}" + ("  <-- TARGET" if cname in TARGET_CLASSES else ""))
        for rep in sorted(rep_data):
            d = rep_data[rep]
            gf1 = d["generated"].get("per_class_f1", {}).get(cname)
            rf1 = d["real"].get("per_class_f1", {}).get(cname)
            gp = d["generated"].get("per_class_precision", {}).get(cname)
            gr = d["generated"].get("per_class_recall", {}).get(cname)
            excluded = cname in d["generated"].get("excluded_classes", [])
            tag = "  [EXCLUDED from generated eval]" if excluded else ""
            print(f"    rep{rep}: generated_f1={gf1}  real_f1={rf1}  "
                  f"precision={gp}  recall={gr}{tag}")
            if gf1 is not None:
                gen_vals.append(gf1)
            if rf1 is not None:
                real_vals.append(rf1)
        if gen_vals:
            m, s = float(np.mean(gen_vals)), float(np.std(gen_vals))
            class_gen_means[cname] = m
            class_gen_stds[cname] = s
            print(f"    -> generated_f1 across reps: mean={m:.4f} std={s:.4f} (n={len(gen_vals)})")
        if real_vals:
            print(f"    -> real_f1 across reps:      mean={np.mean(real_vals):.4f} "
                  f"std={np.std(real_vals):.4f} (n={len(real_vals)})")

    # ── 3. Rank all classes ───────────────────────────────────────────────────
    print("\n[3. All classes ranked by mean generated-data F1 (averaged across reps)]")
    all_class_means: dict[str, float] = {}
    for cname in class_names:
        vals = [
            rep_data[rep]["generated"].get("per_class_f1", {}).get(cname)
            for rep in rep_data
            if rep_data[rep]["generated"].get("per_class_f1", {}).get(cname) is not None
        ]
        if vals:
            all_class_means[cname] = float(np.mean(vals))
    ranked = sorted(all_class_means.items(), key=lambda kv: kv[1])
    for cname, m in ranked:
        marker = "  <-- TARGET" if cname in TARGET_CLASSES else ""
        print(f"  {cname:6s}: mean generated_f1={m:.4f}{marker}")
    non_target_vals = [m for c, m in all_class_means.items() if c not in TARGET_CLASSES]
    non_target_median = float(np.median(non_target_vals)) if non_target_vals else None

    # ── 4. Cross-reference against original TRTR per_class_f1 ───────────────
    print("\n[4. Cross-reference against original TRTR per_class_f1 "
          f"({args.trtr_eval})]")
    trtr = load_json(Path(args.trtr_eval))
    trtr_pcf: dict[str, float] = {}
    if trtr is None:
        print(f"  [WARNING] {args.trtr_eval} not found -- skipping this cross-check.")
    else:
        pcf = trtr.get("per_class_f1")
        if isinstance(pcf, dict):
            trtr_pcf = pcf
        elif isinstance(pcf, list):
            trtr_pcf = dict(zip(TRTR_FIXED_CLASS_ORDER, pcf))
        for cname in class_names:
            t = trtr_pcf.get(cname)
            m = all_class_means.get(cname)
            marker = "  <-- TARGET" if cname in TARGET_CLASSES else ""
            print(f"  {cname:6s}: TRTR per_class_f1={t}  vs. mentor generated_f1 this run={m}{marker}")

    # ── 5. checkpoint_metrics.json trend ──────────────────────────────────────
    print("\n[5. checkpoint_metrics.json trend]")
    ckpt_metrics_path = Path("outputs/results") / args.run_tag / "checkpoint_metrics.json"
    records = load_json(ckpt_metrics_path)
    mentor_macro_f1_std = None
    if records is None:
        print(f"  [WARNING] {ckpt_metrics_path} not found -- skipping.")
    else:
        for rec in records:
            print(f"  iter={rec.get('iteration')} kind={rec.get('kind')} "
                  f"a3_reward_mean={rec.get('a3_reward_mean')} "
                  f"trtr_diag_conf_mean={rec.get('trtr_diag_conf_mean')} "
                  f"mentor_macro_f1={rec.get('mentor_macro_f1')} "
                  f"mentor_macro_f1_std={rec.get('mentor_macro_f1_std')}")
            if rec.get("mentor_macro_f1_std") is not None:
                mentor_macro_f1_std = rec["mentor_macro_f1_std"]

    # ── VERDICT ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # a) Independent corroboration
    print("\n(a) Does the independent Mentor Classifier corroborate HYP/OTHER "
          "being poorly classified, independent of the reward function's own "
          "TRTR classifier?")
    if non_target_median is not None:
        for cname in TARGET_CLASSES:
            m = all_class_means.get(cname)
            if m is None:
                print(f"  {cname}: no generated_f1 data available -- cannot corroborate.")
            elif m < non_target_median:
                print(f"  {cname}: mean generated_f1={m:.4f} < median of the other four "
                      f"classes ({non_target_median:.4f}) -- YES, corroborated by the "
                      "independent Mentor Classifier, not just the reward function's TRTR eval.")
            else:
                print(f"  {cname}: mean generated_f1={m:.4f} >= median of the other four "
                      f"classes ({non_target_median:.4f}) -- NOT corroborated as uniquely bad "
                      "by the Mentor Classifier; reconsider whether the r_diag collapse is "
                      "specific to the TRTR classifier used in training.")
    else:
        print("  Insufficient data across classes to compute a comparison median.")

    # b) Pre-existing weakness vs. RL-induced
    print("\n(b) Is this a pre-existing classifier/data weakness, or did RL-generated "
          "data score worse than the original TRTR weakness alone would predict?")
    for cname in TARGET_CLASSES:
        t = trtr_pcf.get(cname)
        m = all_class_means.get(cname)
        if t is None or m is None:
            print(f"  {cname}: insufficient data (TRTR={t}, mentor={m}) -- cannot compare.")
            continue
        ratio = m / t if t > 0 else float("inf")
        if ratio >= 0.8:
            print(f"  {cname}: TRTR per_class_f1={t:.4f}, mentor generated_f1={m:.4f} "
                  f"(ratio={ratio:.2f}) -- consistent with a PRE-EXISTING classifier "
                  "weakness on this class, not something RL made meaningfully worse.")
        else:
            print(f"  {cname}: TRTR per_class_f1={t:.4f}, mentor generated_f1={m:.4f} "
                  f"(ratio={ratio:.2f}) -- RL-generated data scores MEANINGFULLY WORSE than "
                  "the pre-existing TRTR weakness alone would predict; this looks like RL "
                  "made it worse, not just inherited a weak classifier.")

    # c) Per-class stability across reps vs. the aggregate noise warning
    print("\n(c) Given the reps' spread, is the run's own \"mentor_macro_f1 too noisy\" "
          "warning also true at the per-class level for HYP/OTHER, or are those two "
          "classes stable across reps despite the aggregate metric being noisy?")
    if mentor_macro_f1_std is None:
        print("  No mentor_macro_f1_std found in checkpoint_metrics.json -- cannot compare "
              "per-class std against the aggregate noise figure directly.")
    for cname in TARGET_CLASSES:
        s = class_gen_stds.get(cname)
        if s is None:
            print(f"  {cname}: no per-rep std available.")
            continue
        if mentor_macro_f1_std is not None:
            if s < mentor_macro_f1_std * 0.5:
                print(f"  {cname}: per-class generated_f1 std={s:.4f}, well below the "
                      f"aggregate mentor_macro_f1_std={mentor_macro_f1_std:.4f} -- this class's "
                      "own signal is STABLE across reps despite the aggregate metric being noisy.")
            else:
                print(f"  {cname}: per-class generated_f1 std={s:.4f}, comparable to or larger "
                      f"than the aggregate mentor_macro_f1_std={mentor_macro_f1_std:.4f} -- this "
                      "class's own signal is ALSO noisy across reps, not just the aggregate.")
        else:
            print(f"  {cname}: per-class generated_f1 std={s:.4f} (no aggregate figure to compare against).")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
