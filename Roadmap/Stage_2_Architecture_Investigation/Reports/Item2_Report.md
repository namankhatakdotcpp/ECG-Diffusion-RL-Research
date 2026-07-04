# Item 2 (Localized Gain) -- Phase A Report

Stage 2 / Tier 0 Item 2, localized-gain variant (`k=0`, block1->2 transition),
per the locked pre-registration (`Reports/Tier0_Findings.md`, Item 2 v3,
commit `e84c54c`). Analytical (hook-substitution) phase only, no
gradient-descent retrain -- per Sec. 10, retrain is only triggered if at
least one variant reaches SUPPORTED or partial support, which this result
does trigger, but the retrain decision itself is a separate, later step,
not executed in this report.

## Identity-regression test (Sec. 6)

**PASSED**, twice independently:
1. Standalone A2 test (`identity_regression_test.json`): layer 1
   bit-identical, layers 2-6 within 9.54e-07 of baseline (floating-point
   roundoff from `cached_A + (out - cached_A)` arithmetic).
2. Built-in re-check inside the A3 sweep itself (override-hook code path,
   independently implemented from A2's cached-mode hook): max|diff| across
   all 15 cells x 20 draws x 6 layers at g=1.0 = 7.63e-06.

Both confirm the substitution mechanism reduces exactly to the unmodified
forward pass at g=1.0, to floating-point tolerance, before any g != 1 result
is trusted.

## Summary table

| Gain | Block 6 Recovery% | Direction Consistency (min, layers 2-6) | Propagation Efficiency | Verdict |
|---|---|---|---|---|
| 1.0 | -0.00% | 0.995580 | n/a (g=1.0) | REJECTED |
| 1.25 | 8.64% | 0.995783 | 0.1698 | REJECTED |
| 1.5 | 19.94% | 0.995839 | 0.1958 | REJECTED |
| 2.0 | 46.92% | 0.995762 | 0.2306 | PARTIAL SUPPORT |
| 3.0 | 107.77% | 0.995639 | 0.2666 | SUPPORTED |
| 5.0 | 248.72% | 0.996063 | 0.3113 | SUPPORTED |

Baseline pooled block1->2 drop used as the recovery-fraction denominator:
0.0635 (Item 1's own pooled n=15 statistic, `Tier0_Findings.md` Sec. 9) --
not recomputed independently in this report, per the pre-registration's
explicit instruction to use that pooled figure.

## Monotonicity check (authorized gate addition)

Recovery% is **monotonically non-decreasing** across the full grid
{1.0, 1.25, 1.5, 2.0, 3.0, 5.0} -- no dip observed, no flag raised.

## Sanity gates (per standing authorization)

- Direction consistency floor (0.989): never violated -- minimum observed
  across the entire sweep is 0.995580 (at g=1.0), well above the floor.
- Propagation efficiency bounds (0-100%): all in-range (efficiency
  increases mildly from ~0.17 at g=1.25 to ~0.31 at g=5.0 -- roughly
  20-31% of the injected correction survives to block 6, the rest is
  re-absorbed by the frozen downstream blocks, consistent with a real but
  partial, not total, propagation).
- No NaN, no timeout.

## Decision-table verdict (Sec. 9)

Applied per gain value independently, per the closing clarification
(no monotonicity/unanimity requirement across the grid):

**Overall: SUPPORTED (driven by g=3.0)**

g=3.0 is the smallest gain in the locked grid that clears the 70%
recovery threshold (107.77% recovery, direction consistency 0.9956 --
comfortably above the 0.989 floor). g=2.0 lands at 46.92% (partial
support only). g=5.0 also clears SUPPORTED (248.72% recovery) with
direction consistency intact (0.9961) -- overshoot past 100% recovery
at high gain is itself informative (see propagation-efficiency note
above: the correction overshoots block 6's baseline delta once injected
strongly enough, rather than saturating at 100%).

Per Item 2 v3 Sec. 10, this SUPPORTED result at the analytical stage is
what would trigger escalation consideration to a gradient-descent retrain
-- flagged here as the decision point, not executed as part of this report.

## Artifacts

- Raw per-gain JSON: `Outputs/stage2_tier0_item2_localized_gain/gain_{1.00,1.25,1.50,2.00,3.00,5.00}.json`
- Sweep summary: `Outputs/stage2_tier0_item2_localized_gain/sweep_summary.json`
- Summary table: `Outputs/stage2_tier0_item2_localized_gain/summary.csv`
- Figures: `Figures/stage2_tier0_item2_localized_gain/{recovery_vs_gain,direction_vs_gain,propagation_efficiency}.png`
