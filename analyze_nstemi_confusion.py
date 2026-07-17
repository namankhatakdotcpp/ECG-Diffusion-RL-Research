"""
analyze_nstemi_confusion.py -- per-repetition confusion breakdown for the
Mentor Classifier's NSTEMI class on Stage 4 RL-generated data, to determine
whether NSTEMI's severe generated-F1 degradation (Finding 2,
Roadmap/Stage_4_Optimization/Decisions.md: mean generated F1 ~= 0.170,
rep0=0.000/rep1=0.378/rep2=0.131) is:

  A. collapse predominantly into one competing class,
  B. broad confusion spread across multiple classes, or
  C. qualitatively unstable behavior across the 3 mentor-eval repetitions.

WHY JSON, NOT THE _plain.txt FILES
------------------------------------
classifier_generated_eval.json's "confusion_matrix" field is the same
np.ndarray (via cm.tolist()) that _plain.txt is rendered from
(mentor_eval/classification_validation.py:227,244 write both from the same
`cm` variable) -- but the JSON is structured and doesn't need re-parsing a
fixed-width text table. Preferred source per investigation instruction
("if JSON artifacts contain cleaner raw confusion matrices, prefer those").

CONFUSION MATRIX ORIENTATION -- verified from source, not assumed
--------------------------------------------------------------------
`write_plain_confusion_table`'s own docstring
(mentor_eval/classification_validation.py:151) states "actual=rows,
predicted=columns", and `cm = confusion_matrix(y, pred, ...)`
(classification_validation.py:194) is sklearn's confusion_matrix with
its standard convention: cm[i, j] = number of samples with true label i
predicted as label j. Row i in this script always means "true class i".

Usage:
    python analyze_nstemi_confusion.py --run-tag stage4_finetune_v1 --iteration 1000 --reps 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from mentor_eval.class_mapping import MENTOR_CLASSES

TARGET_CLASS = "NSTEMI"


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
    args = parser.parse_args()

    print("=" * 70)
    print(f"NSTEMI CONFUSION BREAKDOWN  (run-tag={args.run_tag}, "
          f"iteration={args.iteration}, reps={args.reps})")
    print("=" * 70)

    if TARGET_CLASS not in MENTOR_CLASSES:
        print(f"[BLOCKED] '{TARGET_CLASS}' not in MENTOR_CLASSES={MENTOR_CLASSES} "
              "-- mentor_eval/class_mapping.py has changed since this script was written.")
        return
    nstemi_idx = MENTOR_CLASSES.index(TARGET_CLASS)

    per_rep_rows: dict[int, list[int]] = {}
    per_rep_metrics: dict[int, dict] = {}
    missing = []

    for rep in args.reps:
        base = Path("outputs/mentor_review") / f"rl_checkpoint_iter{args.iteration:04d}_rep{rep}"
        gen_path = base / "classifier_generated_eval.json"
        gen = load_json(gen_path)
        if gen is None:
            missing.append((rep, gen_path))
            continue

        cm = gen.get("confusion_matrix")
        if cm is None:
            print(f"\n[WARNING] rep{rep}: {gen_path} has no 'confusion_matrix' key -- "
                  "schema mismatch or this run predates the field being added. Skipping.")
            continue
        if len(cm) != len(MENTOR_CLASSES) or any(len(row) != len(MENTOR_CLASSES) for row in cm):
            print(f"\n[WARNING] rep{rep}: confusion_matrix shape {len(cm)}x"
                  f"{len(cm[0]) if cm else 0} does not match len(MENTOR_CLASSES)="
                  f"{len(MENTOR_CLASSES)} -- class list has likely changed since this run. Skipping.")
            continue

        nstemi_row = cm[nstemi_idx]  # true=NSTEMI, predicted-as-<col> counts
        per_rep_rows[rep] = nstemi_row

        true_count = sum(nstemi_row)
        tp = nstemi_row[nstemi_idx]
        # column sum for precision: how many samples (any true class) were predicted NSTEMI
        predicted_nstemi_count = sum(row[nstemi_idx] for row in cm)
        # zero_division=0 convention, matching sklearn's precision_score/recall_score/f1_score
        # (same convention the JSON's own "reported" values were computed with) -- keeps
        # from_cm directly comparable to reported instead of using None for an edge case
        # the reported field never uses.
        precision = tp / predicted_nstemi_count if predicted_nstemi_count > 0 else 0.0
        recall = tp / true_count if true_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # cross-check against the JSON's own reported per_class_precision/recall/f1
        reported_p = gen.get("per_class_precision", {}).get(TARGET_CLASS)
        reported_r = gen.get("per_class_recall", {}).get(TARGET_CLASS)
        reported_f1 = gen.get("per_class_f1", {}).get(TARGET_CLASS)

        per_rep_metrics[rep] = {
            "true_count": true_count,
            "precision_from_cm": precision,
            "recall_from_cm": recall,
            "f1_from_cm": f1,
            "precision_reported": reported_p,
            "recall_reported": reported_r,
            "f1_reported": reported_f1,
        }

    if missing:
        print("\n[WARNING] missing rep data (not found locally -- expected, GPU-server-only artifact):")
        for rep, path in missing:
            print(f"  rep{rep}: {path}")

    if not per_rep_rows:
        print(
            "\n[BLOCKED] No usable confusion_matrix data found under "
            f"outputs/mentor_review/rl_checkpoint_iter{args.iteration:04d}_rep*/ -- "
            "nothing to analyze locally. This is expected if these are GPU-server-only "
            "artifacts (per this project's environment split). Run this same script on "
            "the GPU server after `git pull`:\n\n"
            f"    python analyze_nstemi_confusion.py --run-tag {args.run_tag} "
            f"--iteration {args.iteration} --reps {' '.join(str(r) for r in args.reps)}\n"
        )
        return

    # ── Per-rep breakdown ────────────────────────────────────────────────────
    print(f"\n[Per-rep breakdown -- true {TARGET_CLASS} predicted as:]")
    for rep in sorted(per_rep_rows):
        row = per_rep_rows[rep]
        m = per_rep_metrics[rep]
        print(f"\n  rep{rep} (n_true_{TARGET_CLASS}={m['true_count']}):")
        for j, cname in enumerate(MENTOR_CLASSES):
            frac = row[j] / m["true_count"] if m["true_count"] else 0.0
            marker = "  <-- TARGET (correct)" if j == nstemi_idx else ""
            print(f"    predicted {cname:8s}: {row[j]:4d}  ({frac:5.1%}){marker}")
        p_cm, r_cm, f_cm = m["precision_from_cm"], m["recall_from_cm"], m["f1_from_cm"]
        p_rep, r_rep, f_rep = m["precision_reported"], m["recall_reported"], m["f1_reported"]
        print(f"    precision: from_cm={p_cm}  reported={p_rep}"
              f"{'  [MISMATCH]' if p_cm is not None and p_rep is not None and abs(p_cm - p_rep) > 1e-6 else ''}")
        print(f"    recall:    from_cm={r_cm}  reported={r_rep}"
              f"{'  [MISMATCH]' if r_cm is not None and r_rep is not None and abs(r_cm - r_rep) > 1e-6 else ''}")
        print(f"    f1:        from_cm={f_cm}  reported={f_rep}"
              f"{'  [MISMATCH]' if f_cm is not None and f_rep is not None and abs(f_cm - f_rep) > 1e-6 else ''}")

    # ── A vs B vs C classification ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERDICT: single-class collapse (A) vs. broad confusion (B) vs. unstable-across-reps (C)")
    print("=" * 70)

    dominant_wrong_class_per_rep: dict[int, Optional[str]] = {}
    for rep in sorted(per_rep_rows):
        row = per_rep_rows[rep]
        true_count = per_rep_metrics[rep]["true_count"]
        if true_count == 0:
            dominant_wrong_class_per_rep[rep] = None
            continue
        wrong = [(MENTOR_CLASSES[j], row[j]) for j in range(len(MENTOR_CLASSES)) if j != nstemi_idx]
        wrong.sort(key=lambda kv: -kv[1])
        top_cls, top_n = wrong[0] if wrong else (None, 0)
        top_frac = top_n / true_count
        # A: one wrong class holds a clear majority (>=60%) of misclassified samples
        misclassified = true_count - row[nstemi_idx]
        top_frac_of_errors = top_n / misclassified if misclassified > 0 else 0.0
        dominant_wrong_class_per_rep[rep] = top_cls if top_frac_of_errors >= 0.6 else None
        print(f"  rep{rep}: top misclassification target = {top_cls} "
              f"({top_n}/{misclassified} of errors = {top_frac_of_errors:.1%}) "
              f"-- {'single-class-dominant (A)' if top_frac_of_errors >= 0.6 else 'spread across classes (B)'}")

    non_none = [c for c in dominant_wrong_class_per_rep.values() if c is not None]
    if len(per_rep_rows) < 2:
        print("\n  Only 1 rep available -- cannot assess cross-rep stability (C). Need >=2 reps.")
    elif len(non_none) == len(per_rep_rows) and len(set(non_none)) == 1:
        print(f"\n  -> CONSISTENT single-class collapse (A): every rep's dominant "
              f"misclassification target is '{non_none[0]}'. Not (C) -- stable across reps.")
    elif len(non_none) == 0:
        print("\n  -> CONSISTENT broad confusion (B): no rep shows a single dominant "
              "wrong class holding >=60% of errors. Not (C) -- stable pattern across reps.")
    else:
        print(f"\n  -> UNSTABLE ACROSS REPS (C): dominant-wrong-class pattern differs "
              f"rep-to-rep ({dominant_wrong_class_per_rep}), or some reps show single-class "
              "collapse while others show broad confusion. This qualitative instability is "
              "itself evidence, consistent with NSTEMI's already-noted high generated-F1 "
              "run-to-run variance (rep0=0.000, rep1=0.378, rep2=0.131).")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
