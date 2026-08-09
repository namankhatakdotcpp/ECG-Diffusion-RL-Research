"""
analysis/rl_significance_power_check.py — Task 3 statistical power re-analysis.

Re-analyzes the Stage 3 (S3-001) vs Stage 4 (stage4_finetune_v1) 10-seed
generated-data macro-F1 comparison reported in
Roadmap/Stage_4_Optimization/Decisions.md (the "Corrected 10-Seed
Comparison" section, Welch's t = 1.221, p ≈ 0.241).

Raw per-seed data (CONFIRMED — copied verbatim from the table in
Decisions.md, seeds 42..51, both computed via classification_validation.py
with excluded_classes == ["AFIB"] confirmed uniform across all 20 runs):

    seed | Stage 3 (S3-001) | Stage 4 (stage4_finetune_v1)
    42   | 0.3421            | 0.3422
    43   | 0.4612            | 0.6268
    44   | 0.4547            | 0.5424
    45   | 0.5185            | 0.5443
    46   | 0.5025            | 0.4894
    47   | 0.4994            | 0.5611
    48   | 0.4751            | 0.4004
    49   | 0.4622            | 0.5101
    50   | 0.5044            | 0.4993
    51   | 0.5201            | 0.6245

This script:
  1. Recomputes Welch's t and p from the raw data and checks it reproduces
     the reported t=1.221, p=0.241 (CRITICAL finding if it doesn't).
  2. Reports n.
  3. Computes Cohen's d for the observed effect (pooled-variance definition,
     standard for an independent two-sample comparison).
  4. Runs a post-hoc power analysis (statsmodels TTestIndPower) at the
     observed effect size and n.
  5. If power < 0.8, computes the n required to reach power 0.8 at the
     observed effect size.

Run: python3 analysis/rl_significance_power_check.py
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

STAGE3 = np.array([0.3421, 0.4612, 0.4547, 0.5185, 0.5025,
                    0.4994, 0.4751, 0.4622, 0.5044, 0.5201])
STAGE4 = np.array([0.3422, 0.6268, 0.5424, 0.5443, 0.4894,
                    0.5611, 0.4004, 0.5101, 0.4993, 0.6245])

REPORTED_T = 1.221
REPORTED_P = 0.241
REPORTED_DF = 14.49

ALPHA = 0.05
POWER_TARGET = 0.8


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-variance Cohen's d for two independent samples."""
    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    return float((b.mean() - a.mean()) / pooled_std)


def main() -> None:
    n3, n4 = len(STAGE3), len(STAGE4)
    assert n3 == n4 == 10, f"Expected n=10 per group, got n3={n3}, n4={n4}"

    t_stat, p_val = stats.ttest_ind(STAGE4, STAGE3, equal_var=False)

    # Welch-Satterthwaite df, for direct comparison against the reported df≈14.49
    var3, var4 = STAGE3.var(ddof=1), STAGE4.var(ddof=1)
    df = ((var3 / n3 + var4 / n4) ** 2) / (
        (var3 / n3) ** 2 / (n3 - 1) + (var4 / n4) ** 2 / (n4 - 1)
    )

    print("=" * 72)
    print("TASK 3 — RL result statistical power re-analysis")
    print("=" * 72)

    print(f"\nn per group: Stage 3 = {n3}, Stage 4 = {n4}")
    print(f"Stage 3 mean = {STAGE3.mean():.4f}, std = {STAGE3.std(ddof=1):.4f}")
    print(f"Stage 4 mean = {STAGE4.mean():.4f}, std = {STAGE4.std(ddof=1):.4f}")
    print(f"Mean difference (Stage 4 - Stage 3) = {STAGE4.mean() - STAGE3.mean():.4f}")

    print(f"\nRecomputed Welch's t = {t_stat:.3f}  (reported: {REPORTED_T})")
    print(f"Recomputed df        = {df:.2f}  (reported: {REPORTED_DF})")
    print(f"Recomputed two-tailed p = {p_val:.3f}  (reported: {REPORTED_P})")

    t_matches = abs(t_stat - REPORTED_T) < 0.01
    p_matches = abs(p_val - REPORTED_P) < 0.01
    if t_matches and p_matches:
        print("\n[CONFIRMED] Recomputed t and p match the reported values in Decisions.md.")
    else:
        print("\n[CRITICAL] Recomputed t/p do NOT match the reported values — "
              "investigate before trusting either the original or this result.")

    d = cohens_d(STAGE3, STAGE4)
    print(f"\nCohen's d (pooled-variance) = {d:.3f}")

    power_analysis = TTestIndPower()
    observed_power = power_analysis.power(effect_size=d, nobs1=n3, ratio=n4 / n3, alpha=ALPHA)
    print(f"\nPost-hoc power at observed d={d:.3f}, n={n3}, alpha={ALPHA}: {observed_power:.3f}")

    print("\n" + "-" * 72)
    if observed_power < POWER_TARGET:
        n_required = power_analysis.solve_power(
            effect_size=d, power=POWER_TARGET, ratio=1.0, alpha=ALPHA
        )
        print(f"[FINDING] Observed power ({observed_power:.3f}) is below the "
              f"conventional {POWER_TARGET} threshold.")
        print(f"[FINDING] n required per group to reach power={POWER_TARGET} at "
              f"this same effect size (d={d:.3f}): {n_required:.1f} "
              f"(≈{int(np.ceil(n_required))} seeds per checkpoint).")
        verdict = "INCONCLUSIVE due to low power"
    else:
        print(f"[FINDING] Observed power ({observed_power:.3f}) meets the "
              f"conventional {POWER_TARGET} threshold — the null result is "
              f"well-powered, not just under-sampled.")
        verdict = "CONFIRMED no statistically distinguishable effect (adequately powered)"
    print("-" * 72)

    print(f"\nVERDICT: {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
