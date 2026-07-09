"""
diagnose_smoke_test_reward_trend.py -- checks whether the smoke test's
"reward_did_not_degrade_first_vs_second_half" FAIL is a class-rotation
confound or a real decline.

WHY THIS EXISTS
----------------
The smoke test's class is sampled randomly each iteration
(`class_idx = int(rng.integers(0, n_classes))` in step07_rl_finetuning.py).
Different classes have shown different baseline reward magnitudes
throughout this project (e.g. HYP/OTHER-type classes scoring lower than
NORM/CD in the earlier A3Reward validation). A raw first-half-vs-second-
half split of reward_total can therefore partly reflect "which classes
happened to land in which half," not whether the SAME policy scores worse
on the SAME class over time. This script re-slices the already-collected
rl_training_log.csv (no retraining) to check which one it is.

METHOD
-------
1. Raw split (reproduces the smoke test's own check): mean(reward_total)
   for iterations [1..n/2] vs [n/2+1..n].
2. Class-adjusted split: for each iteration, subtract that class's OWN
   mean reward_total across the whole run (i.e. remove the class-level
   mean, keeping only the within-class deviation), then redo the
   first/second-half split on the adjusted values. If the adjusted decline
   is much smaller than the raw one, class rotation was a real confound.
   If it's similar, the decline is not explained by class rotation alone.
3. Per-class trend: for each class with >=2 occurrences, report its own
   reward_total values in order of occurrence (a class with a real within-
   class decline would still show it here even after adjustment above).

CAVEAT, stated explicitly, not glossed over: at n=10 iterations spread
across up to 6 classes, most classes will have only 1-3 occurrences. This
adjustment is directional evidence, not a statistically powered test --
same honesty standard already applied to the A3 bootstrap CI work. A
"class-adjusted decline persists" result here is suggestive, not proof;
report it as such.

Usage:
    python diagnose_smoke_test_reward_trend.py [--csv PATH]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="logs/rl_training_log.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[BLOCKED] {csv_path} not found. Run "
              "`python step07_rl_finetuning.py --smoke-test` first -- this "
              "script only re-analyses that run's existing log, it doesn't "
              "generate new data.")
        return

    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "iter": int(row["iter"]),
                "class": row["class"],
                "reward_total": float(row["reward_total"]),
            })
    rows.sort(key=lambda r: r["iter"])
    n = len(rows)

    print("=" * 70)
    print(f"SMOKE TEST REWARD TREND DIAGNOSIS  ({csv_path}, n={n} iterations)")
    print("=" * 70)

    print("\n[per-iteration sequence]")
    for r in rows:
        print(f"  iter={r['iter']:02d}  class={r['class']:6s}  reward_total={r['reward_total']:.4f}")

    class_counts = defaultdict(list)
    for r in rows:
        class_counts[r["class"]].append(r["reward_total"])
    print("\n[class occurrence counts this run]")
    for cls, vals in sorted(class_counts.items(), key=lambda kv: -len(kv[1])):
        print(f"  {cls:6s}: n={len(vals)}  values={[round(v, 4) for v in vals]}  mean={np.mean(vals):.4f}")

    half = max(1, n // 2)
    raw_first  = [r["reward_total"] for r in rows[:half]]
    raw_second = [r["reward_total"] for r in rows[half:]]
    print("\n[1. RAW first/second-half split -- reproduces the smoke test's own check]")
    print(f"  first half  (iters 1-{half}):  mean={np.mean(raw_first):.4f}  values={[round(v,4) for v in raw_first]}")
    print(f"  second half (iters {half+1}-{n}): mean={np.mean(raw_second):.4f}  values={[round(v,4) for v in raw_second]}")
    raw_delta = np.mean(raw_second) - np.mean(raw_first)
    print(f"  raw delta (second - first) = {raw_delta:+.4f}")

    class_mean = {cls: float(np.mean(vals)) for cls, vals in class_counts.items()}
    adjusted = [r["reward_total"] - class_mean[r["class"]] for r in rows]
    adj_first, adj_second = adjusted[:half], adjusted[half:]
    print("\n[2. CLASS-ADJUSTED first/second-half split -- removes each class's own mean]")
    print(f"  first half  adjusted mean = {np.mean(adj_first):+.4f}")
    print(f"  second half adjusted mean = {np.mean(adj_second):+.4f}")
    adj_delta = np.mean(adj_second) - np.mean(adj_first)
    print(f"  class-adjusted delta (second - first) = {adj_delta:+.4f}")

    print("\n[3. Per-class within-class trend (classes with >=2 occurrences)]")
    any_multi = False
    for cls, vals in class_counts.items():
        if len(vals) >= 2:
            any_multi = True
            trend = "declining" if vals[-1] < vals[0] else "not declining"
            print(f"  {cls}: {[round(v,4) for v in vals]}  (first->last: {trend})")
    if not any_multi:
        print("  No class occurred >=2 times in this run -- within-class trend not checkable "
              "at this iteration count.")

    print("\n" + "=" * 70)
    print("VERDICT")
    if abs(adj_delta) < abs(raw_delta) * 0.5:
        print(f"  Class-adjusted delta ({adj_delta:+.4f}) is much smaller than the raw delta "
              f"({raw_delta:+.4f}) -- consistent with class rotation being a real confound in "
              "the raw first/second-half check, not evidence of policy degradation. NOT proof "
              "(n=10, see caveat in module docstring) -- directional evidence only.")
    elif adj_delta < 0:
        size_word = "larger than" if abs(adj_delta) > abs(raw_delta) else "not meaningfully smaller than"
        print(f"  Class-adjusted delta ({adj_delta:+.4f}) is still negative and {size_word} "
              f"the raw delta ({raw_delta:+.4f}) -- class rotation does NOT fully "
              "explain the decline. This is NOT yet proof of a real PPO problem either (n=10, "
              "see caveat) -- but it rules out the confound as a full explanation. Worth a "
              "longer validation run before trusting this direction.")
    else:
        print(f"  Class-adjusted delta ({adj_delta:+.4f}) is non-negative -- no decline once "
              "class is controlled for.")
    print("=" * 70)


if __name__ == "__main__":
    main()
