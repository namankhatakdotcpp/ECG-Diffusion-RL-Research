# Item 5 (AdaLN/FiLM Parameter Statistics) -- Report

## Executive Summary

Per-block adaLN weight-matrix Frobenius-norm decomposition into
scale (`scale1`+`scale2`) vs. shift (`shift1`+`shift2`) capacity.
**VERIFIED**: allocation is non-uniform across blocks (0.553 to 0.761,
a 20.8-percentage-point range). The largest single-step change (block
1->2, +12.4pp) coincides with Item 1's dominant conditioning-magnitude
drop; the only decrease (block 5->6, -5.5pp) coincides with Item 1's
smaller secondary drop -- a notable correlation, stated as correlation,
not causation.

## Methodology

Pure weight inspection, no forward pass, no data, no draws (n=1, one
frozen checkpoint). Confirmed by direct source read
(`step04_transformer_diffusion.py:169,171-172`): `adaLN` is
`nn.Linear(2*model_dim, 4*model_dim)`, and `torch.chunk(4, dim=-1)` on
its output splits into 4 contiguous row-blocks of `adaLN.weight`, in
order `shift1, scale1, shift2, scale2`. Per block: `||W||_F` (full),
`||W_chunk||_F` for each of the 4 chunks,
`scale_fraction = (scale1^2+scale2^2)/||W||_F^2`,
`shift_fraction = 1 - scale_fraction`.

## Verification (Phase D -- actual values, not just PASS strings)

**Internal consistency audit:** all 6 blocks' `adaLN.weight`/`bias`
shapes matched the expected `(4*256, 2*256)`/`(4*256,)` pattern exactly
-- PASS, no shape mismatches.

**Reconstruction check** (verifies the quadrature-sum claim --
`sum(chunk_norm^2) == full_norm^2` for disjoint row-blocks -- is
actually true numerically, not just algebraically asserted):
**max relative difference across all 6 blocks = 1.72e-07** (floating-
point roundoff scale, not a real discrepancy). This confirms the
scale/shift fraction decomposition is numerically exact, not an
approximation.

## Results

| Block | \|\|W\|\|_F | Scale fraction | Shift fraction | Δ scale_fraction (vs. prev block) |
|---|---|---|---|---|
| 1 | 18.341 | 0.5527 | 0.4473 | -- |
| 2 | 19.571 | 0.6766 | 0.3234 | **+0.1239** (largest jump) |
| 3 | 17.046 | 0.6923 | 0.3077 | +0.0157 |
| 4 | 16.750 | 0.7230 | 0.2770 | +0.0307 |
| 5 | 16.192 | 0.7605 | 0.2395 | +0.0375 |
| 6 | 12.721 | 0.7053 | 0.2947 | **-0.0552** (only decrease) |

Scale-fraction range: 0.2079 (>2pp threshold locked in the
pre-registration) -> **VERIFIED**, non-uniform allocation.

Pattern: block 1 is the most balanced (55/45 shift/scale) -- closest to
an even split. Blocks 2-5 climb steadily toward scale-dominance
(68%->76%). Block 6 (the final block, immediately before `final_norm`)
reverses partway back toward shift, the only block-to-block decrease
in the whole sequence.

## Interpretation

The mechanism does not allocate capacity evenly between additive
(shift) and multiplicative (scale) modulation -- it is scale-dominant
almost everywhere except block 1, and increasingly so through the
middle blocks. This is consistent with (not proof of) a "conditioning
signal that acts increasingly through rescaling the residual stream's
existing content, rather than adding a fresh signal" story for blocks
2-5, while block 1's near-even split suggests the initial
conditioning injection is closer to a balanced additive+multiplicative
adjustment.

## Cross-validation (Phase H)

- **Item 1** (two-drop shape: dominant block1->2, smaller block5->6):
  the largest scale_fraction jump occurs at exactly the block1->2
  transition (+12.4pp, more than 3x the next-largest step), and the
  ONLY decrease occurs at exactly the block5->6 transition (-5.5pp).
  Both of Item 1's identified transition points show the two most
  unusual scale_fraction changes in the sequence. **Stated precisely:
  this is a correlation between two independently-measured quantities
  (Item 1's forward-pass cross-class delta magnitude; Item 5's static
  weight-capacity allocation) -- it is not evidence that one causes
  the other**, and no causal mechanism is asserted here. **This is
  observational correlation only. No intervention was performed** --
  unlike Item 2's gain-substitution experiments, nothing was manipulated
  here to test whether changing scale/shift allocation would change
  Item 1's magnitude-decay pattern; both quantities were simply measured
  independently on the same frozen checkpoint. It is, however, exactly
  the kind of convergent signal from independent measurement types
  (Item 5's weights, Item 1's activations) that strengthens confidence
  the block1->2 and block5->6 transitions are architecturally
  distinguished, not noise.
- **Item 4** (gradient competitiveness): found `adaLN` parameters
  receive the largest mean gradient norm of any parameter-type bucket
  (~0.013, ~6x `class_emb.weight`'s mean). Item 5 does not directly
  explain WHY adaLN's gradients are large, but confirms adaLN carries
  substantial, non-trivial, unevenly-distributed weight capacity across
  blocks -- consistent with (not proof of) it being an active, high-
  capacity conditioning pathway rather than a vestigial one.
- **Item 3** (residual-update U-shape: valley at block 3, spike at
  block 6): scale_fraction does NOT show a corresponding valley/spike
  at blocks 3/6 (it climbs smoothly 1->5, only reversing slightly at
  6) -- these two measurements do NOT show the same shape. Stated
  plainly as a non-match, not glossed over: static weight-capacity
  allocation and within-pass residual-update magnitude are different
  quantities and there is no a priori reason they must track each
  other block-for-block.

## Limitations

n=1 (one frozen checkpoint) -- no confidence interval is meaningful;
every number here is an exact, deterministic property of the trained
weights, not a sampled estimate. Cannot establish causation with any
of Items 1/3/4's findings, only the correlations stated above.

## Decision

**VERIFIED** -- non-uniform scale/shift allocation exists across
blocks (0.2079 range, well above the 0.02 threshold locked before
running). The block1->2 and block5->6 correlations with Item 1 are
noted as a real, worth-tracking pattern for the eventual Stage 2
Decision Report, explicitly flagged as correlational.

## Next Steps

No further investigation needed for Item 5 itself. The block1->2/
block5->6 correlation with Item 1 is a candidate line for the eventual
Decision Report's synthesis section, not a new sub-experiment.

## Artifacts

- `Outputs/stage2_tier0_item5_adaln_statistics/adaln_statistics.json`
- `Outputs/stage2_tier0_item5_adaln_statistics/adaln_statistics.csv`
- `Figures/stage2_tier0_item5_adaln_statistics/scale_shift_fraction_vs_block.png`
