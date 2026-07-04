# Item 6 (Attention Entropy and Attention-Map Inspection) -- Report

## Executive Summary

**Late transformer blocks exhibit measurably higher class-dependence
than early/middle blocks, despite the network-wide average remaining
below the predefined class-blindness threshold.** Blocks 5 and 6's
point estimates (0.0643, 0.0547) exceed the locked 0.05 threshold,
while blocks 1-4 are all comfortably below it (0.016-0.031) -- a clean
separation in point estimates. **Updated after computing confidence
intervals across the 15 pooled pair x timestep cells (not done in the
original draft of this report): the separation is real in direction,
but blocks 5/6's exceedance of the specific 0.05 threshold is NOT
statistically airtight** -- both blocks' 95% CIs cross 0.05 (block 5:
[0.0396, 0.0890]; block 6: [0.0417, 0.0676]), while blocks 1-4's CIs sit
entirely below it (upper bounds 0.027-0.040, no overlap with 0.05).
**Correct reading: blocks 5-6 are measurably, qualitatively less
class-blind than blocks 1-4 (the point-estimate separation is real and
the CIs don't overlap each other's typical range), but whether blocks
5/6 individually "exceed the 0.05 threshold" specifically is uncertain
given cell-to-cell variance -- stated with the correct amount of
confidence, not overclaimed.** The overall network-wide verdict
(VERIFIED, pooled mean 0.0352 < 0.05) is unaffected by this correction.

## Methodology

**Correction applied before running:** the master prompt's "STEMI"
class does not exist in this project's real taxonomy (`['NORM', 'MI',
'STTC', 'CD', 'HYP', 'OTHER']`, confirmed against the loaded checkpoint)
-- substituted MI (class 1), matching every other item's 0-vs-1
convention.

No cross-attention exists in this architecture (confirmed,
`step04_transformer_diffusion.py:157-160,174`) -- self-attention only,
conditioned via `adaLN` modulation of attention's input, never as a
query/key/value. `H(head, query) = -sum_key(p*log(p+eps))`, averaged
over heads and query positions, computed by replaying
`block.attn(h, h, h, need_weights=True, average_attn_weights=False)`
(per-head weights, NOT pre-averaged -- entropy is non-linear, so these
are not interchangeable) using the exact adaLN-modulated input each
block's attention actually received (captured via a new
`register_attention_input_hooks` pre-hook, `common/hooks.py`). Same
5-pairs x 3-timesteps x 20-draws design as Item 1/3, `model.eval()`
mode (no dropout).

## Verification (Phase D)

- 5 pairs x 3 timesteps x 20 draws x 6 blocks x 2 classes all completed
  without error; no NaN, no exceptions.
- Entropy values are all well within `[0, log(600)=6.9078]` (the
  theoretical max for a uniform distribution over 600 keys) -- sanity
  bound satisfied at every block/class/cell.
- Runtime: 42s total (CPU), matching the pre-registration's "seconds to
  ~1-2 minutes" estimate for Item 1/3-style synthetic-noise designs --
  no runtime correction needed (unlike Item 4).

## Results

| Block | H(class A, NORM) | H(class B, pooled over MI/STTC/CD/HYP/OTHER) | \|diff\| (point est.) | 95% CI (across 15 cells) |
|---|---|---|---|---|
| 1 | 5.8252 | 5.8070 | 0.0182 | [0.0140, 0.0278] |
| 2 | 5.7161 | 5.6847 | 0.0314 | [0.0230, 0.0398] |
| 3 | 5.5864 | 5.5604 | 0.0259 | [0.0130, 0.0388] |
| 4 | 5.6057 | 5.5891 | 0.0166 | [0.0084, 0.0269] |
| 5 | 4.4262 | 4.3619 | **0.0643** | [0.0396, 0.0890] -- crosses 0.05 |
| 6 | 3.3365 | 3.2818 | **0.0547** | [0.0417, 0.0676] -- crosses 0.05 |

95% CIs computed as `mean +/- 1.96*SE` across the n=15 pooled
(pair, timestep) cells per block (the same n=15 evidentiary unit Item
1 used for its own pooled statistics). **Blocks 1-4's CIs sit entirely
below the 0.05 threshold with no overlap; blocks 5-6's point estimates
exceed it but their CIs include sub-threshold values** -- the
"exceeds 0.05" claim for blocks 5/6 specifically is directionally
suggestive, not statistically confirmed at conventional confidence.

Pooled mean |diff| across all 6 blocks: **0.03519** (< 0.05 threshold
-> VERIFIED, class-blind, per the locked criterion) -- this network-
wide verdict does not depend on the blocks-5/6 CI question above and
is unaffected by it.

**Entropy itself declines steadily from block 1 (~5.83, close to the
6.91 theoretical max -- broad, near-uniform attention) to block 6
(~3.32, well below max -- attention has become substantially more
peaked/focused).** This progressive sharpening is itself a notable,
independent architectural observation, separate from the class-blindness
question.

## Interpretation

The network-wide verdict (class-blind, pooled mean under threshold)
is real and matches the master prompt's stated interpretation --
attention's *distributional shape* barely differs by class label
almost everywhere. But blocks 5 and 6 are the exception: both
individually cross the 0.05 threshold, meaning the *late* blocks'
attention pattern is measurably (if still small in absolute terms
relative to the ~6.9 max) more class-dependent than the early/middle
blocks'. This should not be smoothed over by reporting only the pooled
number -- it is a real, per-block pattern in point estimates (blocks
1-4 all below 0.032, blocks 5-6 both above 0.054, a clean separation
across the six point estimates). **However, per the CI computation
above, the specific claim that blocks 5/6 "exceed the 0.05 threshold"
should be read as directionally suggestive, not statistically
confirmed** -- their confidence intervals, computed across the same
15 pooled cells, include values below 0.05.

## Cross-validation (Phase H)

This is the **fourth independent measurement type** (after Item 1's
cross-class output-magnitude delta, Item 3's within-pass residual-
update ratio, Item 5's static adaLN weight-capacity allocation) to
flag blocks 5/6 as architecturally distinctive:

- **Item 1:** smaller, real secondary conditioning-magnitude drop at
  block5->6.
- **Item 3:** residual-update-ratio spike at block 6 (~8x the valley),
  confirmed by causal ablation to have outsized post-normalization
  influence.
- **Item 5:** the only scale_fraction decrease across the whole
  sequence occurs at block5->6.
- **Item 6 (this item):** the only two blocks whose attention entropy
  measurably differs by class label are blocks 5 and 6.

**Stated precisely, per this project's standing discipline: this is
convergent correlation across four independently-measured quantities
(forward-pass activation delta, within-pass residual magnitude, static
weight capacity, attention entropy), not proof of a single causal
mechanism** -- and Item 6's own contribution to that convergence carries
the CI caveat above (point-estimate separation is real; the specific
0.05-threshold crossing is not statistically airtight). Four different
measurement types agreeing on DIRECTION (blocks 5-6 stand out) is still
a stronger signal than any one alone, but the reader should weight
Item 6's contribution accordingly -- as "directionally consistent,"
not as an independently-confirmed threshold breach.

## Limitations

Entropy differences, even at blocks 5/6, are small in absolute terms
(0.05-0.06 out of a ~6.9 max scale, ~0.8-0.9%) -- "measurably different"
is not "large." The pooled network-wide verdict (class-blind) is not
overturned by this finding; the per-block pattern is a refinement, not
a contradiction, of the headline verdict. **The blocks-5/6 exceedance
of the 0.05 threshold specifically has wide enough cell-to-cell
variance (n=15) that its CI crosses the threshold -- this is a real
limitation of the finding's precision, not just a formality**, and
should be carried into any citation of this result in the Stage 2
Decision Report.

## Decision

**VERIFIED** (network-wide, pooled criterion) -- attention is
substantially class-blind, consistent with the master prompt's
interpretation that adding cross-attention would not obviously help.
**Refinement, not overturning the verdict:** blocks 5-6 show
directionally distinctive (higher) class-dependence than blocks 1-4 in
point estimates, converging in DIRECTION with three other independent
measurement types (Items 1/3/5) on the same late-block region -- but
the specific claim that blocks 5/6 individually cross the 0.05
threshold is not statistically confirmed given cell-to-cell variance
(both CIs include sub-threshold values).

## Next Steps

No further investigation needed for Item 6 itself. The blocks-5/6
convergence across four measurement types is a candidate headline
finding for the Stage 2 Decision Report's synthesis, not a new
sub-experiment.

## Artifacts

- `Outputs/stage2_tier0_item6_attention_entropy/attention_entropy_raw.json`
- `Outputs/stage2_tier0_item6_attention_entropy/attention_entropy.csv`
- `Figures/stage2_tier0_item6_attention_entropy/attention_entropy_vs_block.png`
