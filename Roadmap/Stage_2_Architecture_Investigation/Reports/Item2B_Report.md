# Item 2B (Uniform Gain) -- Phase A-D Report
Stage 2 / Tier 0 Item 2, uniform-gain variant (blocks 1-5, cumulative
substitution, budget-matched via `g_k = g_L^(1/sqrt(5))`), per the locked
pre-registration (`Reports/Tier0_Findings.md`, Item 2 v3, commit
`e84c54c`, budget-matching rule carried forward unchanged from v2).
Analytical (hook-substitution) phase only, no gradient-descent retrain.

## Identity-regression test (Sec. 6)
**PASSED**, independently of Item 2A's own test (the uniform hook's
cumulative bookkeeping is structurally different -- 5 chained
substitutions vs. 1 -- so Item 2A's test result was not assumed to carry
over, per Sec. 6's explicit instruction):
- Standalone test (single draw, pair 0->1, t=500): layer 1 bit-identical,
  layers 2-6 within 9.54e-07 of baseline -- the SAME roundoff ceiling as
  Item 2A's single-substitution test, confirming the cumulative bookkeeping
  does not compound floating-point error across the 5 chained hooks.
- Built-in re-check inside the full sweep (15 cells x 20 draws x 6 layers):
  max|diff| at nominal_gain=1.0 = 1.53e-05
  (larger than the single-cell test's max, as expected from a much larger
  sample -- still floating-point-roundoff scale, not a bug signature).

## 1. Item 2B verdict -- decided ONLY on its own fixed criteria
Per Sec. 9's decision table, applied against the fixed pooled baseline
(0.0635 block1->2 drop) and the fixed 0.989 direction floor -- **independent
of any comparison to Item 2A** (a methodology issue in the *comparison*
does not retroactively weaken a verdict that does not depend on that
comparison):

| Nominal Gain | Per-block Gain | Recovery% | Direction (min, L2-6) | Verdict |
|---|---|---|---|---|
| 1.0 | 1.0000 | -0.00% | 0.995580 | REJECTED |
| 1.25 | 1.1049 | 31.73% | 0.995908 | PARTIAL SUPPORT |
| 1.5 | 1.1988 | 75.69% | 0.996061 | SUPPORTED |
| 2.0 | 1.3634 | 196.33% | 0.996129 | SUPPORTED |
| 3.0 | 1.6345 | 569.36% | 0.996469 | SUPPORTED |
| 5.0 | 2.0539 | 1871.24% | 0.997014 | SUPPORTED |

**Item 2B verdict: SUPPORTED (driven by nominal_gain=1.5 -- the smallest gain in the locked grid that clears the SUPPORTED threshold, same convention as Item 2A's g=3.0)**

Recovery% is monotonically non-decreasing across the full grid -- no dip observed, no flag raised. No NaN, no negative efficiency. With the corrected (Sec. 2) denominator -- the **L2 combination**
of the 5 per-hook injected magnitudes, NOT the plain sum -- efficiency ranges 0.95-1.22, much closer to a sane ~1.0 ceiling than the
uncorrected single-hook formula's 1.4-2.3. It still modestly exceeds 1.0 at gains 1.5-5.0, and
this residual is NOT fully resolved by the L2 fix: unlike the plain-sum alternative (which treats
the 5 sequential corrections as independent additive contributions -- the same linearity
assumption that broke the original budget-matching formula), the L2 combination does not carry
that specific flaw, so a >1.0 reading under L2 is a more genuine signal that slightly more block-6
magnitude was recovered than the L2-combined injected delta predicts, not a residual of the same
additive-assumption bug. The g=1.0 identity test rules out a hook-mechanism bug independently.
This is flagged here as an open question, not resolved: RecoveredMagnitude (measured at block 6,
via the standard magnitude metric) and the L2-combined InjectedMagnitude (measured at the 5
injection points, via the same metric applied per-block) are still two different measurement
paths through a nonlinear network, and nothing in this analysis proves they should sum to exactly
1.0 even under a perfectly matched budget -- whoever revisits Item 2's efficiency metric should
treat this residual as unresolved, not as evidence the L2 fix under- or over-corrects.

## 2. Propagation-efficiency denominator, corrected
Item 2A's `InjectedDelta(g) = (g-1)*mag1_baseline` assumes a single
injection point and does not fit the uniform variant's 5 cumulative
injections. This report uses the ACTUAL per-hook injected magnitude
instead -- `(g_k-1)*layer_magnitude_avg[k]` at each of blocks 1-5, already
captured by `common/hooks.py`'s mean-pool hook (which fires BEFORE each
block's CorrectionHook and therefore records the real pre-correction
delta at that block, not a formula-derived proxy) -- combined via L2
across the 5 hooks (consistent with the budget-matching formula's own
squared-log framing; a plain sum was also computed and shows the same
trend, so this choice does not drive the conclusion below).

| Nominal Gain | Recovery% | Direction (min) | Efficiency (corrected, L2) | Verdict |
|---|---|---|---|---|
| 1.0 | -0.00% | 0.995580 | n/a (g=1.0) | REJECTED |
| 1.25 | 31.73% | 0.995908 | 0.9505 | PARTIAL SUPPORT |
| 1.5 | 75.69% | 0.996061 | 1.0746 | SUPPORTED |
| 2.0 | 196.33% | 0.996129 | 1.1971 | SUPPORTED |
| 3.0 | 569.36% | 0.996469 | 1.2164 | SUPPORTED |
| 5.0 | 1871.24% | 0.997014 | 1.0898 | SUPPORTED |

## 3. Localized vs. Uniform comparison -- CONFOUNDED
**This section's numbers are real, but the comparison is not clean.**
The budget-matching formula (`g_k = g_L^(1/sqrt(5))`, an additive-log-gain
heuristic) was intended to hold total injected correction magnitude equal
between the localized and uniform variants at matched nominal gain `g`.
A post-processing audit (this session, no rerun -- using the raw
pre-correction magnitudes each hook already captured) found it does not:

| Nominal g | Localized Injected | Uniform Injected (sum) | Uniform Injected (L2) | Ratio (sum) | Ratio (L2) |
|---|---|---|---|---|---|
| 1.0 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a |
| 1.25 | 0.0337 | 0.0457 | 0.0212 | 1.357 | 0.630 |
| 1.5 | 0.0673 | 0.0981 | 0.0447 | 1.457 | 0.665 |
| 2.0 | 0.1346 | 0.2283 | 0.1041 | 1.696 | 0.774 |
| 3.0 | 0.2693 | 0.6106 | 0.2972 | 2.268 | 1.104 |
| 5.0 | 0.5385 | 1.9506 | 1.0903 | 3.622 | 2.025 |

The ratio **grows with gain rather than staying flat near 1.0** -- from ~1.36x (sum) / ~0.63x (L2) at g=1.25 to ~3.62x (sum) / ~2.03x (L2) at g=5.0.
The mechanism is visible in the per-hook breakdown: at g=5.0, blocks 4 and 5's
own raw (pre-correction) delta magnitude is 0.4647 and 0.9267 respectively -- several
times block 1's 0.1419 -- even though all 5 blocks
nominally received the same budget-matched `g_k`. Later hooks are correcting a
signal that earlier hooks have already amplified, so the linear/additive
log-gain budgeting formula underestimates actual injected magnitude at high
gain -- a real confound arising from nonlinear compounding, not a units error
or implementation bug (the g=1.0 identity test rules that out independently).

**Two claims, explicitly separated:**

- **SURVIVES the confound:** under the tested budget formula, distributed
  correction achieved higher raw recovery than concentrated correction, at
  every non-identity gain in the locked grid. This is a description of what
  was observed, not a causal claim -- it stands regardless of the confound.
- **DOES NOT SURVIVE the confound:** "distribution is inherently more
  effective than concentration, at matched injection strength." This causal
  claim is blocked -- injected magnitude was not actually held constant
  between variants (see ratio table above), so the recovery gap cannot yet
  be attributed to WHERE the correction landed vs. HOW MUCH correction
  actually landed, since those two things covaried instead of being held
  fixed the way the pre-registration's budget-matching rule assumed.

**Qualifier (carry forward to any future citation of this comparison):**
> Recovery advantage of uniform over localized is confounded by budget-matching breakdown under nonlinear compounding -- see Item2B_Report.md Sec. 3 before citing this as an architectural preference.

## Artifacts
- Raw per-gain JSON: `Outputs/stage2_tier0_item2_uniform_gain/gain_{1.00,1.25,1.50,2.00,3.00,5.00}.json`
- Sweep summary: `Outputs/stage2_tier0_item2_uniform_gain/sweep_summary.json`
- Budget-matching audit: `Outputs/stage2_tier0_item2_uniform_gain/budget_matching_audit.{json,csv}`
- Summary table: `Outputs/stage2_tier0_item2_uniform_gain/summary.csv`
- Figures: `Figures/stage2_tier0_item2_uniform_gain/{recovery_vs_gain,direction_vs_gain,propagation_efficiency,localized_vs_uniform_recovery}.png`
