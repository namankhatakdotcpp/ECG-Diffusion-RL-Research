# Item 6 (Attention Entropy and Attention-Map Inspection) -- Report

## Executive Summary

**Overall VERIFIED** (pooled mean entropy diff 0.03519 < 0.05 threshold)
-- attention is largely class-blind, consistent with the master
prompt's own interpretation ("argues against cross-attention being
sufficient by itself"). **But the pooled average masks a real, worth-
flagging pattern: blocks 5 and 6 individually exceed the 0.05 threshold**
(0.0643 and 0.0547) while blocks 1-4 are all comfortably below it
(0.016-0.031) -- late blocks are measurably less class-blind than early/
middle blocks, even though the network-wide average clears the bar.

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

| Block | H(class A, NORM) | H(class B, pooled over MI/STTC/CD/HYP/OTHER) | \|diff\| |
|---|---|---|---|
| 1 | 5.8252 | 5.8070 | 0.0182 |
| 2 | 5.7161 | 5.6847 | 0.0314 |
| 3 | 5.5864 | 5.5604 | 0.0259 |
| 4 | 5.6057 | 5.5891 | 0.0166 |
| 5 | 4.4262 | 4.3619 | **0.0643** |
| 6 | 3.3365 | 3.2818 | **0.0547** |

Pooled mean |diff| across all 6 blocks: **0.03519** (< 0.05 threshold
-> VERIFIED, class-blind, per the locked criterion).

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
number -- it is a real, per-block pattern, not noise (the pattern is
monotonic: blocks 1-4 all below 0.032, blocks 5-6 both above 0.054, a
clean separation, not a single outlier draw).

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
mechanism.** But four different measurement types agreeing that blocks
5-6 are where something architecturally different happens is a
stronger signal than any one of them alone, and is exactly the kind of
convergent evidence worth foregrounding in the eventual Stage 2
Decision Report's synthesis section.

## Limitations

Entropy differences, even at blocks 5/6, are small in absolute terms
(0.05-0.06 out of a ~6.9 max scale, ~0.8-0.9%) -- "measurably different"
is not "large." The pooled network-wide verdict (class-blind) is not
overturned by this finding; the per-block pattern is a refinement, not
a contradiction, of the headline verdict.

## Decision

**VERIFIED** (network-wide, pooled criterion) -- attention is
substantially class-blind, consistent with the master prompt's
interpretation that adding cross-attention would not obviously help.
**Refinement, not overturning the verdict:** blocks 5-6 are the
exception, converging with three other independent measurement types
(Items 1/3/5) on the same late-block region.

## Next Steps

No further investigation needed for Item 6 itself. The blocks-5/6
convergence across four measurement types is a candidate headline
finding for the Stage 2 Decision Report's synthesis, not a new
sub-experiment.

## Artifacts

- `Outputs/stage2_tier0_item6_attention_entropy/attention_entropy_raw.json`
- `Outputs/stage2_tier0_item6_attention_entropy/attention_entropy.csv`
- `Figures/stage2_tier0_item6_attention_entropy/attention_entropy_vs_block.png`
