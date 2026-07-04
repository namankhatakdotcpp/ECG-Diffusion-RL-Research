# Stage 2 / Tier 0 -- Evidence Matrix

Maps each cross-cutting claim tested across Stage 2's Tier 0 to every
item that produced evidence bearing on it. Built to prevent narrative
bias in the eventual `Stage_2_Decision_Report.md` -- makes explicit how
(and whether) each conclusion is actually supported across independent
experiments, rather than letting a single item's framing carry into the
synthesis unchallenged.

**Legend:** Supports / Weakly supports / Neutral / Contradicts /
Confounded / Not testable / Pending (verification incomplete).

**Do not write `Stage2_Decision_Report.md` from this matrix yet** --
Item 4 is still Pending (blocked on the GPU rerun), and this matrix
should be re-checked once it closes.

## Evidence matrix

| Claim | Item 1 | Item 2A | Item 2B | Item 3 | Item 4 | Item 5 | Item 6 | Item 7 | Item 8 |
|---|---|---|---|---|---|---|---|---|---|
| Conditioning magnitude/influence attenuates through the network | **Supports** (two-drop shape, dominant block1->2) | Supports (recovers magnitude when injected) | Supports (own criteria) | **Supports, refines** (non-monotonic U-shape, not simple decay) | -- | Weakly supports (scale/shift reallocation correlates with attenuation points) | Weakly supports (entropy declines block1->6, separate from class-blindness) | Not testable | **Contradicts a magnitude=information equivalence reading** (Fisher declines but linear separability never collapses) -- refines, does not contradict Item 1's own claim |
| Class-conditioning information is preserved/decodable throughout the network | Neutral (measures magnitude/direction, not decodability) | Neutral | Neutral | Neutral | -- | Neutral | Neutral | Not testable | **Supports** (100% linear-probe accuracy at every block, permutation-verified) |
| Blocks 5-6 are architecturally distinctive vs. blocks 1-4 | **Supports** (secondary drop at block5->6) | Neutral | Neutral | **Supports** (residual-ratio spike at block 6, causally confirmed) | Pending | **Supports** (only scale_fraction decrease at block5->6) | **Weakly supports** (point estimates higher at 5/6; CIs cross the 0.05 threshold -- directionally consistent, not independently confirmed) | Not testable | **Does not corroborate** (linear-probe accuracy flat at 100% everywhere, no block-5/6 signal) |
| LayerScale-style gain correction can restore block-6 conditioning magnitude | Motivates the hypothesis | **Supports** (own criteria, g=3.0) | **Supports** (own criteria, nominal_gain=1.5) | Neutral | Neutral | Neutral | Neutral | Not testable | Neutral |
| Localized vs. uniform gain correction -- which is more effective | -- | Compared against | **Confounded** (budget-matching formula breaks down under nonlinear compounding; "uniform wins" not a clean finding without this qualifier) | -- | -- | -- | -- | -- | -- |
| Class-embedding gradient was ever competitive during training | Motivates the hypothesis (independent signal) | -- | -- | -- | **Pending** (GPU rerun required; preliminary N=10 showed ~61% percentile rank, declining with timestep -- not yet locked) | Weakly supports (adaLN receives largest gradient bucket, per preliminary Item 4 data) | -- | -- | -- |
| AdaLN allocates scale/shift capacity evenly across blocks | -- | -- | -- | -- | -- | **Contradicts (falsified as stated)** -- allocation is non-uniform, range 0.208 | -- | -- | -- |
| Attention already varies by class label (cross-attention would not obviously help) | -- | -- | -- | -- | -- | -- | **Supports, network-wide** (pooled diff 0.035 < 0.05); **weakly contradicts at blocks 5-6** (point estimates exceed threshold, CI-uncertain) | -- | -- |
| Class embedding differentiates over the course of training | -- | -- | -- | -- | -- | -- | -- | **Not testable** (BLOCKED, permanent -- no per-epoch checkpoint history exists) | -- |

## Notes on specific cells

- **Item 2B row:** its own verdict (SUPPORTED) is listed separately from
  the localized-vs-uniform comparison row, which is explicitly marked
  **Confounded** -- per the standing project rule not to merge a clean
  per-variant verdict with a confounded cross-variant comparison into
  one hedged mark.
- **Item 7:** marked **Not testable**, not Neutral or Contradicts --
  the claim was never evaluated at all, for a documented, permanent
  reason (no data exists), not because the evidence was ambiguous.
- **Item 8's "does not corroborate" mark** on the blocks-5/6 row is
  reported as-is, post-validity-check -- the permutation control
  confirmed Item 8's underlying finding (100% accuracy is real, not a
  dimensionality artifact), so this is a confirmed non-convergence, not
  a provisional one awaiting further checking.
- **Item 4's row is Pending throughout** -- do not treat the preliminary
  (N=10, unverified) ~61% percentile rank as a locked number in any
  synthesis; it is cited above only to show what a locked Item 4 might
  look like, not as evidence.

## Cross-item finding table: the blocks-5/6 pattern specifically

| Item | Evidence re: blocks 5-6 | Confidence |
|---|---|---|
| 1 | Conditioning magnitude shows a secondary, smaller, real drop at block5->6 (statistically confirmed, Wilcoxon) | High |
| 3 | Residual-update ratio spikes at block 6 (~8x the block-3 valley); causal ablation confirms outsized post-normalization influence (3.24x vs. control block) | High |
| 4 | Pending -- preliminary secondary-design data (N=10, unverified) shows gradient percentile rank declining with timestep, not yet block-localized to 5/6 specifically | Pending |
| 5 | The ONLY scale_fraction decrease across all 6 blocks occurs at block5->6; also the largest single jump at block1->2 | High (n=1, exact deterministic weight property) |
| 6 | Point estimates for entropy class-dependence are highest at blocks 5-6, but 95% CIs (n=15 cells) cross the 0.05 threshold -- directionally consistent, not independently significant | Medium |
| 8 | Linear-probe accuracy is flat 100% at every block -- does NOT single out blocks 5-6 | High (permutation-verified) |

**Four of six testable items (1, 3, 5, 6) show directional convergence
on blocks 5-6; one (8) explicitly does not; one (4) is pending.** This
should be reported in the eventual Decision Report as a real but
qualified convergence -- strong on Items 1/3/5, present but statistically
softer on Item 6, absent on Item 8 -- not smoothed into a uniform "all
evidence points to blocks 5-6" claim.
