# Item 3 (Residual-Path Attenuation) -- Report

Stage 2 / Tier 0 Item 3, per the locked pre-registration
(`Reports/Item3_PreRegistration.md`, commit `3101f28`). Measurement only
-- no intervention in the main sweep (the block-6 ablation check below is
a targeted causal follow-up on one specific claim, not a re-opening of
Item 3's own design).

## Architectural question (restated)

Does conditioning attenuation occur primarily inside the residual branch
itself (visible as a declining `R_k`), or does it emerge only after
residual addition? **Answer: neither cleanly -- `R_k` is non-monotonic
(a U-shape), which is itself the finding** (see Finding 1 below).

## Finding 1: `R_k` varies systematically across blocks -- a U-shape, not attenuation or amplification

| Block | R_k (class A) | R_k (class B) | R_k (combined) |
|---|---|---|---|
| 1 | 0.2516 | 0.2809 | 0.2662 |
| 2 | 0.1248 | 0.1180 | 0.1214 |
| 3 | 0.0943 | 0.0950 | 0.0947 |
| 4 | 0.1287 | 0.1311 | 0.1299 |
| 5 | 0.2174 | 0.2128 | 0.2151 |
| 6 | 0.7812 | 0.7419 | 0.7616 |

**Shape: non-monotonic** -- `R_k` declines from block 1 to a minimum at
block 3 (0.0947), then rises
to a peak at block 6 (0.7616),
a ~8.0x
range. Per the Interpretation Framework's non-monotonic row: this
requires checking whether it reflects a real architecture effect or a
methodological difference from Item 1's own measurement -- **it does not
contradict Item 1's two-drop shape**, because the two items measure
different quantities (Item 3's `R_k` is a within-pass update magnitude;
Item 1's finding is a cross-class output-magnitude delta). The two are
complementary, not required to match in shape.

**Candidate statistical test result (Wilcoxon signed-rank, block 1 vs.
block 6, n=30):** statistic=0.0,
**p=1.67e-06**. Null hypothesis (flat/noise-like) is
**REJECTED** -- `R_k`'s
variation across blocks is statistically significant, not noise. Note
this test characterizes the ENDPOINTS, not the full U-shape; the shape
itself (valley + late-block spike) is the more informative pattern, not
folded into a single p-value.

## Finding 2: class-independence -- `R_k` barely differs between class A and class B

At every block, `R_k` for class A and class B are nearly identical (e.g.
block 6: 0.7812 vs. 0.7419).
**This is a finding in its own right, not a footnote:** it corroborates
Item 1's own conclusion -- that class-conditioning signal lives in
*direction* rather than *magnitude* -- from a completely different,
within-pass measurement angle, independent of Item 1's cross-class
methodology. Two independent measurement approaches landing on the same
qualitative conclusion (magnitude is not where the class signal is
carried) is stronger evidence than either alone.

## Finding 3: block 6's residual update genuinely has outsized post-normalization influence -- confirmed by causal ablation, not inferred from pre-norm ratios

An earlier draft of this finding compared PRE-FinalNorm residual/output
ratios between block 6 and the valley block (3)
and concluded the disparity "survives" FinalNorm's compression. **That
claim was corrected before being locked here** -- a pre-normalization
ratio disparity does not by itself establish what survives a
whole-tensor normalization (LayerNorm normalizes the entire tensor's
variance, not any one block's specific contribution). The corrected,
causally-grounded test: ablate block 6's residual update
(override its output with its own input, skipping its contribution
entirely), pass the result through `final_norm`, and measure how much
`final_norm`'s output changes, compared against doing the identical
ablation at the valley block.

| Quantity | Value |
|---|---|
| Baseline `final_norm` output norm (no ablation) | 13.492 |
| Mean change from ablating block 6 | 5.555 |
| Mean change from ablating block 3 (valley) | 1.714 |
| Ratio | 3.24x |

**Ratio formula, stated explicitly (pool first, then ratio -- not a mean
of per-observation ratios):** each of the 600
individual (pair, timestep, draw, class) observations contributes one
`delta_6(i) = ||final_norm(baseline_i) - final_norm(ablated_block_6_i)||`
and one `delta_3(i)` (same formula, block 3). The two
columns above are `mean_i(delta_6(i))` and `mean_i(delta_3(i))` --
flat means across all 600 observations, not
per-cell -- and the ratio is `mean_i(delta_6(i)) / mean_i(delta_3(i))`,
**not** `mean_i(delta_6(i)/delta_3(i))` (the latter would be unstable
wherever `delta_3(i)` is near zero for a specific draw).

**CONFIRMED by causal ablation: ablating block 6's residual update changes final_norm's output by 5.555 on average, vs. 1.714 for ablating the valley block (3) -- a 3.24x larger effect. Block 6's larger pre-normalization residual DOES translate to proportionally greater post-normalization influence -- this is measured directly via ablation, not inferred from pre-norm ratios.**

(Pre-normalization context, for reference only -- not the basis for the
claim above: block 6's residual/output ratio was
0.454 vs.
0.085 for the valley
block, and `final_norm` compresses block 6's output to
0.183x its
pre-normalization magnitude.)

## Decision criteria (locked now, from this real distribution, per the pre-registration's explicit deferral)

Item 3's hypothesis ("`R_k` varies systematically across blocks") is
evaluated against: Wilcoxon signed-rank test, block 1 vs. block
6, alpha=0.05, matching Item 1's own paired-design precedent.

**Verdict: SUPPORTED (systematic variation across blocks)**

This criterion was NOT chosen before the sweep ran (per the
pre-registration's explicit "Decision criteria: TBD from Item 3's own
fresh sweep" -- no threshold existed prior to seeing this distribution).

## Stage 2 Follow-up Questions (deferred, not blocking Item 3's closure)

Per explicit instruction not to expand Item 3's scope further:
- Attention-sublayer vs. FFN-sublayer decomposition of `R_k` (Item 3's
  granularity is block-level only, per its own scope statement).
- Timestep-dependent subgroup analysis of the U-shape (t=100 vs. t=500
  vs. t=900 shown separately, not just pooled).
- Whether the block 6 spike is specific to this checkpoint
  or a general property of this architecture family.

None of these block Item 3's closure -- they are recorded here for a
future item or the eventual Stage 2 Decision Report, not investigated
further in this item.

## Artifacts

- Raw sweep: `Outputs/stage2_tier0_item3_residual_attenuation/residual_probe_raw.json`
- Pooled summary: `Outputs/stage2_tier0_item3_residual_attenuation/residual_probe.csv`
- Block 6 investigation (pre-norm ratios, superseded by the ablation for the causal claim): `Outputs/stage2_tier0_item3_residual_attenuation/block6_investigation.json`
- Block 6 causal ablation (the evidence actually cited above): `Outputs/stage2_tier0_item3_residual_attenuation/block6_ablation.json`
- Figure: `Figures/stage2_tier0_item3_residual_attenuation/residual_ratio_vs_block.png`
