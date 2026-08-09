# RL Result Statistical Power Re-Analysis — Summary

**Script**: [`analysis/rl_significance_power_check.py`](rl_significance_power_check.py)
**Source data**: `Roadmap/Stage_4_Optimization/Decisions.md`, "Stage 3 vs Stage 4 — Corrected 10-Seed Comparison" table (seeds 42–51, n=10 per group)

## CONFIRMED

- Recomputing Welch's t directly from the 20 raw per-seed values reproduces the paper's reported statistic exactly: **t = 1.221, df = 14.49, p = 0.241**. The original test was computed correctly — this is not a recomputation error.
- Cohen's d for the observed effect (pooled-variance) = **0.546** — a "medium" effect by conventional benchmarks, not a negligible one.
- Post-hoc power to detect an effect of that size at n=10 per group, α=0.05: **0.212** — roughly a 1-in-5 chance of detecting a real effect of this magnitude with this sample size, well below the conventional 0.8 threshold.
- n required per group to reach power=0.8 at the same observed effect size: **≈54 seeds** (vs. the 10 actually run).

## Verdict

**This is INCONCLUSIVE due to low power, not a confirmed null result.**

The non-significant p-value (0.241) does not mean "Stage 4 shows no real advantage over Stage 3." It means the test, at n=10, had only a ~21% chance of detecting an effect this size even if it is real. The point estimate (Stage 4 higher by 0.0401 macro-F1, d=0.546) is consistent with a genuine moderate effect that the sample size simply cannot resolve from noise.

## Implication for the paper's Discussion section

Do **not** write either of these:
- "Stage 4 shows no improvement over Stage 3" (implies confirmed null — not supported)
- "Stage 4 confirmed to outperform Stage 3" (implies confirmed effect — not supported either)

The accurate sentence is closer to: *"The observed mean difference (Stage 4 higher by 0.040 macro-F1, d≈0.55) did not reach statistical significance (Welch's t=1.221, p=0.241) at n=10 seeds per checkpoint; a post-hoc power analysis shows this test had only ~21% power to detect an effect of the observed size, so the result should be reported as inconclusive rather than as evidence of no effect. Reaching conventional power (0.8) at this effect size would require ~54 seeds per checkpoint."*

## Feasibility note (not requested, but relevant to whether "collect more seeds" is a real option)

54 seeds per checkpoint is ~5.4x the seeds already run (10). Each seed is one `classification_validation.py` invocation — if that's cheap (no GPU sampling involved, matches "computed locally... no GPU needed" per Decisions.md), scaling to 54 seeds may be practical. If each seed instead requires fresh DDIM sample generation from the diffusion model, 54 seeds is a much heavier ask and worth flagging as a cost/benefit call before committing.
