# Item 8 (Representation Collapse Analysis) -- Report

## Executive Summary

**VERIFIED at every block, every timestep** -- but the headline number
(100% linear-probe test accuracy, everywhere, with zero train/test gap)
is more important for *what it refines* than for confirming "no
collapse." **Class label remains perfectly linearly decodable at every
block, all the way through block 6, even where Fisher ratio has
declined substantially** (e.g. t=900: 25.17 at block 1 -> 2.17 at block
6, an 11.6x decline). This is not a contradiction of Item 1's magnitude-
decay finding -- it is the exact pattern `representation_metrics.py`'s
own docstring predicted: a trace-based statistic can under-report
separability that a linear decoder finds trivially. **The refinement
this adds to the Stage 2 narrative: class information is never
information-theoretically lost anywhere in this network -- if
downstream generation is poor, it is not because the class label
becomes undecodable, but because of how the (still-decodable, but
proportionally shrinking, per Item 1/3) signal gets used by what comes
after.**

## Methodology

Per the master prompt's explicit instruction (deliberately overriding
`representation_metrics.py`'s own cost-saving suggestion to run
`linear_probe_accuracy()` at only 1-2 Fisher-flagged blocks): both
`fisher_ratio()` and `linear_probe_accuracy()` run at **every** block.
Same cheap single-timestep synthetic-noise forward-pass convention as
Item 1/3/6 (mean-pooled per-block activations via `register_layer_hooks`,
NOT full reverse-diffusion generation). Per timestep (`t in {100, 500,
900}`), `n=120` samples (20 per class x 6 classes), all classes present
simultaneously (unlike Item 1/3/6's pairwise design -- Fisher
ratio/linear probe need multiple classes at once).

## Verification (Phase D)

- All 18 (timestep x block) cells: no classes excluded by either
  function's `min_class_count` guard (`n_gen=20`/class clears both
  the Fisher-ratio floor of 5 and the linear-probe floor of 10).
- Linear probe's PCA-reduction guard triggered as expected
  (`n_train=84 < 5*256=1280` -> reduced to 16 components) at every
  cell -- reported explicitly, not hidden, per the module's own
  contract.
- **Critical check, not assumed:** is 100% test accuracy the
  overfitting/interpolation-regime artifact the module's docstring
  warns about (train high, test collapses)? **No** -- `train_accuracy`
  and `test_accuracy` are BOTH exactly 1.0 at every cell, zero gap. The
  module's own overfitting flag (`train_acc - test_acc > 0.25`) never
  fires anywhere. This is a genuinely different pattern from the
  module's documented failure mode, not a mislabeled instance of it.
- `n_test=36` (30% of 120, stratified across 6 classes, ~6/class) --
  small enough that "100%" should be read as "zero errors observed on
  a modest held-out set," not as an asymptotically precise estimate;
  stated as a limitation below, not hidden.

## Statistical validity check (added after review -- must-fix, not editorial)

A 100% accuracy claim deserves the same suspicion as a printed PASS
string. Reviewer flagged a specific, concrete risk the original
verification pass missed: **`n_train=84` and raw hidden dimension
`d=256` -- `n_train <= d` at literally every one of the 18 cells**
(confirmed explicitly, not assumed). A linear classifier can achieve
100% TRAIN accuracy by construction whenever `n_train` is on the order
of `d`, regardless of whether classes are separable at all (VC
dimension of a linear separator in `d` dimensions is `d+1`). Confusion
matrices/macro-F1/balanced accuracy (an alternative check considered)
would NOT have caught this -- they are monotonic functions of the same
already-perfect confusion matrix and carry no additional information
at 100% accuracy. Three checks were run instead:

1. **n vs. d, reported explicitly:** `n_train=84 <= d=256` at every
   cell -- confirmed as a real, universal condition, not a corner case.
2. **Multiple random train/test splits (n=3, not just the one default
   seed):** accuracy was `[1.0, 1.0, 1.0]` at every cell, range=0.0 --
   the 100% result is not a single lucky partition.
3. **Label-permutation control (the decisive check):** shuffle the true
   labels, refit the IDENTICAL probe (same PCA-reduction logic, same
   code path), report train AND test accuracy on the nonsense labels.
   **Result: shuffled-label test accuracy stayed near chance (range
   0.083-0.278, chance=0.167) at every cell, and -- critically --
   shuffled-label TRAIN accuracy stayed at only ~0.40-0.57, nowhere
   near 1.0.** This is the direct evidence against memorization: if the
   post-PCA-reduced classifier had enough effective capacity to fit any
   arbitrary labeling (the actual risk the n<=d condition raises), it
   would ALSO hit ~100% train accuracy on shuffled labels. It does not.
   The true-label 100% train+test accuracy reflects genuine linearly-
   exploitable class structure, not a dimensionality artifact.

**Conclusion: the finding survives the validity check.** The naive
n-vs-d concern is real and worth having checked, but the permutation
control directly rules out the failure mode it predicts. This
strengthens confidence in the original claim rather than requiring a
downgrade -- stated as an outcome, not assumed in advance.

## Results (abbreviated -- full table in `representation_collapse.csv`)

| Timestep | Block 1 Fisher | Block 6 Fisher | Block 1-6 probe accuracy |
|---|---|---|---|
| 100 | 16.31 | 1.72 | 1.0000 at every block |
| 500 | 21.53 | 2.62 | 1.0000 at every block |
| 900 | 25.17 | 2.17 | 1.0000 at every block |

Fisher ratio declines substantially and fairly smoothly from block 1 to
block 6 at every timestep (roughly monotonic, largest at block 1,
smallest at block 6 or nearby). Linear-probe accuracy is 1.0000 at
literally every one of the 18 cells -- flat, no decline anywhere.

## Interpretation

Two genuinely different questions, with two different answers:
**"Is the class-conditioning signal still information-theoretically
present and linearly recoverable?"** -- yes, everywhere, undiminished
in the classification sense. **"Is the class-conditioning signal's
proportional magnitude/influence on the residual stream declining?"**
-- yes (Item 1's own finding, and Fisher ratio's decline here is
consistent with it). These are not in tension: a signal can shrink in
magnitude relative to the growing residual stream (Item 1/3) while
remaining perfectly linearly separable in direction (Item 8), exactly
because a linear classifier only needs *some* separating direction,
not a large one. This is precisely the failure mode
`representation_metrics.py`'s docstring was written to guard against
being missed.

## Cross-validation (Phase H)

- **Item 1** (two-drop magnitude shape): Fisher ratio's decline here is
  broadly consistent with Item 1's own magnitude-decay finding (both
  decline block1->6), though Item 8's Fisher-ratio decline looks more
  smoothly monotonic here than Item 1's specific two-drop shape --
  worth noting as a difference in shape detail, not a contradiction in
  direction.
- **Items 1/3/5/6's convergent "blocks 5-6 are distinctive" finding:**
  Fisher ratio IS lowest at blocks 5-6 in 2 of 3 timesteps (t=500:
  lowest at block 6; t=900: lowest at block 6; t=100: lowest at block
  3, close to block 6) -- broadly consistent with, though not as
  sharply localized as, the other four items' convergence. Linear-probe
  accuracy shows NO block-5/6-specific pattern at all (flat 100%
  everywhere) -- this measurement does NOT corroborate the blocks-5/6
  story, stated plainly rather than selectively cited only where it agrees.
- **This is the fifth independent measurement type** in Stage 2 (after
  Item 1's activation delta, Item 3's residual ratio, Item 5's weight
  allocation, Item 6's attention entropy) -- and the first one whose
  headline metric (linear-probe accuracy) does NOT single out any
  particular block, providing a useful counterweight: not every
  measurement converges on blocks 5-6, and that should be represented
  honestly in the eventual Decision Report's synthesis, not smoothed
  into "everything points to blocks 5-6."

## Limitations

`n_test=36` per cell is small -- "100%, zero errors" is a real,
clean result but should not be read as a tight statistical estimate;
a handful of errors on a larger held-out set would still be a strong
result, just not literally 100%. PCA reduction to 16 components means
the reported accuracy is on reduced features, not the raw 256-dim
representation, per the module's own required disclosure.

## Decision

**VERIFIED** (linear-probe accuracy, the confirmatory test per the
module's own hierarchy) at every block and timestep -- representation
never collapses in the classification sense. Fisher ratio's decline is
a real, complementary pattern (broadly consistent with Item 1's
magnitude-decay finding) but is NOT, by the module's own documented
behavior, evidence of collapse on its own -- exactly the discipline the
master prompt's "never use Fisher ratio alone to skip the probe"
instruction was designed to enforce, and it mattered here: Fisher ratio
alone would have suggested block 6 is "worse," while the probe shows
class information is fully intact there.

## Next Steps

No further investigation needed for Item 8 itself. The genuine tension
between "signal is fully decodable everywhere" (Item 8) and "signal's
magnitude/influence shrinks" (Item 1/3/5/6, blocks 5-6 particularly) is
a substantive open question for the Stage 2 Decision Report: does
downstream generation quality track magnitude/influence (Item 1/3/5/6's
story) or decodability (Item 8's story)? This is exactly the kind of
contradiction-adjacent finding this project's discipline says to
investigate, not ignore -- flagged here for the Decision Report's
synthesis, not resolved within Item 8's own scope.

## Artifacts

- `Outputs/stage2_tier0_item8_representation_collapse/representation_collapse_raw.json`
- `Outputs/stage2_tier0_item8_representation_collapse/representation_collapse.csv`
- `Figures/stage2_tier0_item8_representation_collapse/representation_collapse_vs_block.png`
